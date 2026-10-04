"""
core/compute_backends/k8s_backend.py  -  Kubernetes 后端

用官方 kubernetes Python client 调 REST API，不 shell 出去调 kubectl——
kubectl 不一定装在 API 服务器上，而且解析它的文本输出远不如直接读结构化响应可靠。

kubeconfig 是按用户存在数据库里的一整份 YAML 文本（见
api/accounts/models_db.py::ComputeResourceProfile），不是读服务器上的
~/.kube/config——每个用户连的是各自的集群，不能共用进程级的全局配置。
"""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

from core.compute_backends import ComputeBackendError


@contextmanager
def _client_from_profile(profile):
    """kubernetes 的 config.load_kube_config() 只接受文件路径，不接受字符串内容，
    所以把库里存的 kubeconfig 文本落成一个临时文件再加载，用完立刻删。
    临时文件权限交给 tempfile 默认的 0600，不放在可预测的路径上。"""
    from kubernetes import client, config

    if not (profile.k8s_kubeconfig or "").strip():
        raise ComputeBackendError("这个配置里没有 kubeconfig 内容，无法连接集群")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=True) as f:
        f.write(profile.k8s_kubeconfig)
        f.flush()
        try:
            cfg = client.Configuration()
            config.load_kube_config(config_file=f.name, client_configuration=cfg)
        except Exception as e:
            raise ComputeBackendError(f"kubeconfig 解析失败（内容可能不是合法的 kubeconfig YAML）：{e}")
        api_client = client.ApiClient(configuration=cfg)
        try:
            yield api_client
        finally:
            api_client.close()


def test_connection(profile) -> tuple[bool, str, Optional[str]]:
    """真实打一次 API server：先读版本（不需要任何 RBAC 权限，纯连通性），
    再尝试列 namespace 里的 Job（这才是真正要用到的权限）——两步分开，
    这样"连得上但没权限"和"根本连不上"能给出不同的提示。"""
    from kubernetes import client
    from kubernetes.client.exceptions import ApiException

    try:
        with _client_from_profile(profile) as api_client:
            try:
                version = client.VersionApi(api_client).get_code()
            except ApiException as e:
                if e.status in (401, 403):
                    return False, "连上了集群，但认证被拒绝——检查 kubeconfig 里的凭据是否过期", str(e.reason)
                return False, f"无法连接集群 API Server（HTTP {e.status}）", str(e.reason)
            except Exception as e:
                return False, "无法连接集群 API Server——检查地址是否可达、网络/VPN 是否通", str(e)

            ns = (profile.k8s_namespace or "default").strip() or "default"

            # namespace 存在性必须单独用 read_namespace 查——不能靠下面的
            # list_namespaced_job 顺带发现：K8s 对"列出一个不存在的 namespace 里的资源"
            # 返回的是 200 + 空列表，不是 404（实测确认）。只测列表的话，namespace
            # 名字打错会显示"连接成功"，等到真正提交训练任务时才炸，排查成本很高。
            try:
                client.CoreV1Api(api_client).read_namespace(name=ns)
            except ApiException as e:
                if e.status == 404:
                    return (False,
                            f"集群里没有 namespace「{ns}」——检查拼写，或者先在集群里创建它",
                            str(e.reason))
                if e.status == 403:
                    # 没权限读 namespace 对象不代表不能在里面跑 Job（RBAC 可以配得很细），
                    # 所以这里不直接判失败，继续往下测真正需要的 Job 权限
                    pass
                else:
                    return False, f"检查 namespace 时出错（HTTP {e.status}）", str(e.reason)

            try:
                client.BatchV1Api(api_client).list_namespaced_job(namespace=ns, limit=1)
            except ApiException as e:
                if e.status == 403:
                    return (False,
                            f"能连上集群，但当前凭据没有在 namespace「{ns}」里读写 Job 的权限——"
                            f"提交训练任务需要这个权限，请让集群管理员授予",
                            str(e.reason))
                return False, f"检查 Job 权限时出错（HTTP {e.status}）", str(e.reason)

            return (True,
                    f"连接成功：Kubernetes {version.git_version}，namespace「{ns}」可读写 Job",
                    f"platform={version.platform}")
    except ComputeBackendError as e:
        return False, str(e), None
    except Exception as e:
        return False, f"连接失败：{e}", None
