"""
core/compute_backends/slurm_backend.py  -  Slurm 后端（SSH 到登录节点）

跟 K8s 后端的一个**真实的、不该被抹平的差异**：很多生产 HPC 环境里，计算节点
没有出站公网访问（防火墙/内网隔离是常态），所以远端任务不一定能 HTTP 回调
这台服务器。因此 Slurm 这边的作业状态以 SSH 轮询 squeue/sacct 为**主要**
机制，webhook 回调只是"网络恰好通的话顺带更新得更及时"，不是唯一依赖。
K8s 那边则相反（Pod 通常能出站），以回调为主、轮询兜底。

私钥是从数据库里读出来的 PEM 文本，不是读服务器上的 ~/.ssh/id_rsa——
每个用户连的是各自的集群，用各自的身份。
"""
from __future__ import annotations

import io
from typing import Optional

from core.compute_backends import ComputeBackendError

_DEFAULT_PORT = 22
_CONNECT_TIMEOUT = 15


def _load_pkey(pem_text: str):
    """paramiko 需要知道私钥的具体类型才能加载，但用户不会（也不该）告诉我们
    这是 RSA 还是 Ed25519——挨个试一遍，这是 paramiko 生态里的常规做法。

    注意用 getattr 动态取类而不是直接引用：paramiko 5.x 已经删掉了 DSSKey
    （DSA 早就不安全、被 OpenSSH 默认禁用了），直接写 paramiko.DSSKey 会在
    新版本上抛 AttributeError。这里按"有就试、没有就跳过"处理，同时兼容新旧版本。"""
    import paramiko

    key_classes = [getattr(paramiko, n, None)
                   for n in ("Ed25519Key", "RSAKey", "ECDSAKey", "DSSKey")]
    for key_cls in [k for k in key_classes if k is not None]:
        try:
            return key_cls.from_private_key(io.StringIO(pem_text))
        except Exception:
            continue
    raise ComputeBackendError(
        "无法解析这份 SSH 私钥——支持 Ed25519/RSA/ECDSA 格式的 PEM。"
        "如果私钥设了密码短语（passphrase），需要换一份没有密码短语的密钥"
        "（服务端无人值守，没法在连接时输入密码短语）"
    )


def _connect(profile):
    import paramiko

    if not (profile.slurm_host or "").strip():
        raise ComputeBackendError("这个配置里没有填 Slurm 登录节点地址")
    if not (profile.slurm_username or "").strip():
        raise ComputeBackendError("这个配置里没有填 SSH 用户名")
    if not (profile.slurm_ssh_private_key or "").strip():
        raise ComputeBackendError("这个配置里没有 SSH 私钥，无法登录")

    pkey = _load_pkey(profile.slurm_ssh_private_key)
    ssh = paramiko.SSHClient()
    # 这里用 AutoAddPolicy：服务端无人值守，没有交互确认 host key 的机会。
    # 代价是首次连接不校验主机身份（理论上可被中间人攻击）——真实的 HPC 场景里
    # 登录节点通常在可信网络内，这个取舍可接受，但要在文档里说清楚。
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=profile.slurm_host.strip(),
        port=profile.slurm_port or _DEFAULT_PORT,
        username=profile.slurm_username.strip(),
        pkey=pkey,
        timeout=_CONNECT_TIMEOUT,
        auth_timeout=_CONNECT_TIMEOUT,
        banner_timeout=_CONNECT_TIMEOUT,
        look_for_keys=False,     # 不要偷偷用服务器上 ~/.ssh 里的别的密钥
        allow_agent=False,       # 同理，不要用服务器进程的 ssh-agent
    )
    return ssh


def _run(ssh, cmd: str, timeout: int = 20) -> tuple[int, str, str]:
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode("utf-8", errors="replace")
    err = stderr.read().decode("utf-8", errors="replace")
    return stdout.channel.recv_exit_status(), out, err


def test_connection(profile) -> tuple[bool, str, Optional[str]]:
    """分三步验证，每步失败给不同的提示：能不能 SSH 上去 → 上去之后是谁 →
    这台机器上到底有没有 Slurm（登录节点上没有 sinfo 是很常见的配置错误，
    比如填成了一台普通跳板机）。"""
    import paramiko

    try:
        ssh = _connect(profile)
    except ComputeBackendError as e:
        return False, str(e), None
    except paramiko.AuthenticationException as e:
        return False, "SSH 认证失败——用户名或私钥不对，或者这个公钥没加进目标账号的 authorized_keys", str(e)
    except paramiko.SSHException as e:
        return False, f"SSH 连接失败：{e}", None
    except (OSError, TimeoutError) as e:
        # 只有真正的网络层错误才说"连不上"——把所有异常都归因成网络不通会误导用户
        # （曾经把一个 paramiko 版本兼容问题报成"网络不可达"，排查方向完全跑偏）
        return False, f"连不上登录节点——检查地址/端口是否正确、网络是否可达：{e}", None
    except Exception as e:
        return False, f"SSH 连接时发生意外错误（{type(e).__name__}）：{e}", None

    try:
        rc, whoami, _ = _run(ssh, "whoami")
        if rc != 0:
            return False, "SSH 连上了，但执行命令失败（账号可能被限制为不可交互登录）", None
        user = whoami.strip()

        rc, sinfo_out, sinfo_err = _run(ssh, "sinfo --version 2>&1 || scontrol --version 2>&1")
        if rc != 0 or not sinfo_out.strip():
            return (False,
                    f"SSH 登录成功（用户 {user}），但这台机器上找不到 Slurm 命令——"
                    f"确认填的是 Slurm 登录节点，而不是普通跳板机",
                    (sinfo_out or sinfo_err).strip()[:200])
        slurm_version = sinfo_out.strip().splitlines()[0]

        partition = (profile.slurm_partition or "").strip()
        if partition:
            rc, part_out, _ = _run(ssh, f"sinfo -h -p {partition} -o '%P' 2>&1")
            if rc != 0 or not part_out.strip():
                return (False,
                        f"登录成功（{user}，{slurm_version}），但集群里没有分区「{partition}」",
                        None)

        return (True,
                f"连接成功：以 {user} 登录，{slurm_version}"
                + (f"，分区「{partition}」存在" if partition else ""),
                None)
    finally:
        try:
            ssh.close()
        except Exception:
            pass
