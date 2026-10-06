<div align="center">

<img src="assets/logo-512.png" alt="VibeML" width="170">

# VibeML

**对话式 AutoML —— 说清楚任务，剩下的交给闭环**

从一句自然语言和十几条样本出发，自动完成数据准备、模型选择、训练、评估与部署。<br>
**并且会告诉你，这个结果到底值不值得信。**

<p>
<img alt="Python" src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white">
<img alt="License" src="https://img.shields.io/badge/License-MIT-green.svg">
<img alt="Backends" src="https://img.shields.io/badge/后端-sklearn%20%7C%20HF%20%7C%20LoRA%20%7C%20RL%20%7C%20VLM-8b5cf6">
<img alt="LLM" src="https://img.shields.io/badge/LLM-Ollama%20%7C%20vLLM%20%7C%20Anthropic%20%7C%20OpenAI兼容-f59e0b">
</p>

</div>

---

## 为什么是另一个 AutoML

市面上的 AutoML 和 LLM Agent 都在比谁的分数更高。但在真实的小数据场景里，**分数本身往往不可信**：

> 在 20 条验证样本上，70% 和 75% 的准确率，统计上无法区分。
> 而几乎所有工具都会把这 5 个百分点报告成"提升"。

VibeML 的出发点不是把分数做得更高，而是**不让你被分数骗了**。

| | 常见 AutoML / Agent | VibeML |
|---|---|---|
| 报告指标 | 一个数字 | 数字 + 置信区间 + **"这次改动是否真实有效"的判定** |
| 迭代决策 | 分数变高就采纳 | 超过噪声下界才采纳，**否则如实说"没有提升"** |
| 小数据 | 样本不足时静默给出乐观估计 | 自动切换交叉验证，并给出噪声水平 |
| 中文 | 多为英文优先，中文常静默劣化 | 按 CJK 占比自动切换字符 n-gram（见下方实测） |
| 使用门槛 | 需要懂特征、模型、超参 | 用大白话描述任务即可 |

---

## 效果预览

<div align="center">
<img src="assets/screenshot.png" alt="VibeML 界面" width="100%">
</div>

左侧是对话推进，右侧是实时训练过程与**逐轮归因**。注意诊断不是泛泛而谈，而是能直接执行的：

> **迭代 1 · 42.9%** —— 模型目前像个只懂说好话的"马屁精"，训练时没学到多少"差评"的例子，所以遇到新评论一律猜成好评。
> **迭代 2 · 69.2%** —— 建议给"负面"类别补至少 40 条样本、权重设为 3.0，**并切换到 `weighted_ce` 损失函数**。

这一轮真实运行里，第 3 轮的单次切分指标升到了 76.9%，但 5 折交叉验证显示 `73.1% → 70.8%`，系统判定**"没有提升"**并停止——这正是噪声门禁在起作用。

---

## 快速开始

```bash
# 1. 装依赖
pip install -r requirements.txt

# 2. 用本地模型跑（免费，推荐）—— 先装好 Ollama 并拉一个模型
ollama pull qwen3:30b

# 3. 启动
python -m uvicorn api.main:app --port 8000
```

打开 <http://localhost:8000> ，在对话框里描述你的任务，例如：

> 帮我做一个酒店评论情感分类模型，判断评论是正面还是负面

系统会主动追问缺失信息（数据从哪来、用什么后端、训几轮），补齐后自动开跑。

<details>
<summary><b>不想用本地模型？支持四种 LLM 来源</b></summary>

| 来源 | 说明 |
|---|---|
| **Ollama** | 本地免费，无需联网，隐私数据不出内网 |
| **OpenAI 兼容协议** | 自建 vLLM / LM Studio，或任意兼容端点 |
| **Anthropic** | 自备 API Key |
| **系统托管** | 登录后使用平台额度，按次计量并留存用量明细 |

在页面右上角「设置 → 配置」里切换，互斥选择、凭据隔离。
</details>

<details>
<summary><b>命令行方式</b></summary>

```bash
python run.py \
  --task "帮我把客服工单按问题类型分类：账号、支付、物流、产品质量、其他" \
  --data examples/data/customer_tickets.jsonl

# 训练完直接预测
python run.py --demo --predict "付款成功但订单显示未支付"
```
</details>

---

## 核心能力

### 🎯 可信评估 —— 这是本项目的立身之本

`core/robust_eval.py` 给每个指标配一个**零成本的解析噪声下界**：

```
二项标准误 σ = √(p(1-p)/n)
n=20, p=0.7  →  σ = ±10.2%   # 70% 与 75% 不可区分
```

改动带来的提升必须**超过 1σ** 才会被采纳。实测行为（真实中文数据集）：

```
150 条训练集 : 66.0% ±8.6%（5 折交叉验证）
600 条训练集 : 75.5% ±3.9%（5 折交叉验证）

[真改进] 600 vs 150   → ✅ 判为真实提升  (+9.5% > ±3.9%)
[空对照] 同数据换顺序  → ✅ 判为噪声      (+0.8% < ±8.6%)
```

既认得出真提升，也不会把重排序造成的波动当成进步。样本量够时自动切换 5 折交叉验证，且向量化器放在 Pipeline 内部，避免折间词表泄漏。

### 💬 对话式任务澄清

不需要懂"特征工程""超参搜索"。描述不完整时系统逐个追问，一次只问一件事；
一次性把信息给全则直接开跑，不会多余追问。

### 🔁 自动迭代与归因

每轮训练后用 LLM 做诊断并给出**可执行**建议，不是泛泛而谈：

> **迭代 1 · 42.9%** —— 模型目前是个"只会说好话"的偏科生，把所有评论都当成正面处理了。
> **建议**：为"负面"类别补充 20–30 条不同角度的差评样本（噪音、异味、服务冷漠、设施故障）。

诊断信号来自真实的 per-class F1 与混淆对，不是凭空生成。

### 🧩 七种后端，一套对话

| 任务类型 | 后端 |
|---|---|
| 文本分类 | sklearn / 预训练模型微调 / **LLM 生成自定义网络结构** |
| 强化学习 | LLM 生成 Gym 环境 + stable-baselines3 |
| 指令微调 | 任意 HF causal LM + LoRA |
| 图像分类 | CLIP 类视觉编码器 + 分类头 |
| 看图说话 / VQA | 端到端微调（BLEU / ROUGE-L 驱动迭代） |

文本分类自带三级降级链：`custom_nn → pretrained_nn → sklearn`，上层失败自动退回，不会整个任务崩掉。

### 🛡️ LLM 生成代码的安全边界

`custom_nn` 和 RL 环境由 LLM 现场生成并**真实执行**，因此有三道防线：

1. **AST 白名单静态门禁**（`core/nn_sandbox.py` / `core/rl_sandbox.py`）：禁危险内置、禁 dunder 逃逸、限定可导入模块
2. **子进程隔离**（`core/subprocess_runner.py`）：`spawn` 独立进程 + 墙钟超时 + terminate→kill 升级
3. **只信任经审计的训练库**：sklearn / transformers / stable-baselines3，LLM 只产出结构定义，不碰训练算法本身

### 🀄 中文不是二等公民

修复过一个影响深远的问题：sklearn 默认 `token_pattern` 靠空格切词，**中文整句会变成一个 token**，样本间零特征重叠，模型退化成查表。

真实数据集实测（`Chinese_sentiment` 1000 条，5 折 f1_macro，1σ≈0.015）：

| 分词方式 | f1_macro |
|---|---|
| 默认 word(1,2) | **0.388 ±0.000** ← 折间方差为 0，即恒定预测多数类 |
| **char(1,3)（现方案）** | **0.629** ±0.041 |

现按 CJK 字符占比自动切换，英文路径逐参数不变。

### 🔌 工程配套

- **账号体系**：注册登录 / Google OAuth / JWT / 按次计量的 LLM 用量审计
- **API 服务**：长效 API Token，每个 Token 自带独立 provider 配置，API 发起的会话可在网页只读查看
- **附件**：PDF / Word / Excel / Markdown / 图片 / 压缩包，支持拖拽与粘贴长文本转附件
- **计算资源**：可配置 Slurm / Kubernetes 连接（⚠ 见下方限制）
- **产物导出**：模型包与代码包一键打包下载

---

## 架构

```
用户对话
   │
   ▼
core/conversation/        对话编排：澄清 → 数据 → 选型 → 配置 → 训练 → 播报
   │                      （Multi-Agent 驱动，工具调用失败自动降级为确定性状态机）
   ▼
core/pipeline.py          迭代闭环：训练 → 评估 → 噪声门禁 → 诊断 → 调整
   │                          │
   │                          ├── core/robust_eval.py      交叉验证 + 噪声下界
   │                          ├── core/explainer.py        LLM 归因与建议
   │                          ├── core/loss_factory.py     按错误模式选损失函数
   │                          └── core/augmentor.py        样本外预测找错标
   ▼
core/*_trainer.py         七种后端，统一经 subprocess_runner 隔离
   ▼
core/*_deployer.py        导出独立可运行的部署包（权重 + inference.py + README）
```

前端 `web/index.html` 通过 WebSocket 实时接收训练事件，`web/app.js::reduceEvent` 负责状态归约（28 个单元测试覆盖）。

---

## 已知限制（如实记录）

这一节不是待办清单，是**现在就成立的事实**，避免你按错误预期使用。

| 限制 | 说明 |
|---|---|
| **训练始终在本地执行** | Slurm / K8s 目前只能配置与测试连接，**尚未真正提交远程作业**。远程执行入口、作业脚本生成、事件回调尚未接通 |
| **发布到 HF / 魔搭 / GitHub 未实现** | 设计约定已定（默认私有、每次显式确认、后端拒绝无确认标志的请求），但代码未落地 |
| **难例检索未证明有效** | 实测定向筛选 **+0.008**，低于 1σ 噪声 0.011，且各种子符号反复变号。当前只保留"难例诊断"与零成本去重，未做超量生成 |
| **凭据明文存库** | `ApiToken.llm_api_key`、`ComputeResourceProfile` 的 kubeconfig / SSH 私钥均为明文（必须可反解才能真正调用）。生产部署建议改接密钥管理服务 |
| **Slurm 用 AutoAddPolicy** | 首次连接不校验主机 host key，理论上可被中间人攻击 |
| **导出下载无额外鉴权** | 拿到 `task_id`(UUID) 即可下载产物，与现有端点行为一致 |
| **sklearn 的 epoch 是模拟的** | 通过"逐步增大训练数据比例"模拟学习曲线，不是真实梯度下降轮次。神经网络后端的 epoch 是真实的 |

---

## 环境要求

- Python 3.10+
- 无需 GPU（sklearn 路径纯 CPU；神经网络后端支持 Apple MPS / CUDA，也可 CPU 兜底）
- LLM：本地 Ollama 即可，无需任何付费 API
- 可选：PostgreSQL（启用账号体系时）

---

## 开发者

**Zhongjiang Yao**

## 许可

本项目采用 [MIT License](LICENSE) 开源。
