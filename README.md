<div align="center">
  <img src="assets/logo.svg" alt="VibeML" width="128" height="128">

  <h1>VibeML</h1>

  <p><b>对话式 AutoML —— 说清楚任务，剩下的交给闭环</b></p>
  <p>用一句自然语言，从 10 条样本出发，跑完整个 ML 闭环。</p>
</div>

---

## 快速开始（3 步）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 设置 API Key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. 运行 Demo（30 条客服工单 → 完整闭环）
python examples/run_demo.py
```

或者用命令行指定任务和数据：

```bash
python run.py \
  --task "帮我把客服工单按问题类型分类：账号、支付、物流、产品质量、其他" \
  --data examples/data/customer_tickets.jsonl
```

训练完成后直接预测：

```bash
python run.py --demo --predict "付款成功但订单显示未支付" "快递三天没有更新"
```

---

## 项目结构

```
automl_agent/
├── run.py                        # CLI 入口
├── config.py                     # 所有共享数据结构（TaskSpec / LoopState ...）
├── requirements.txt
│
├── core/
│   ├── task_parser.py            # Phase 1.1 · 会话解析器
│   ├── data_engine.py            # Phase 1.2 · 小数据引导器
│   ├── trainer.py                # Phase 1.3 · 训练执行引擎
│   ├── explainer.py              # Phase 1.4 · 可解释迭代器
│   └── loop.py                   # Phase 1.5 · 闭环决策代理
│
├── utils/
│   └── io_utils.py               # 结果序列化 / JSONL 读写
│
└── examples/
    ├── run_demo.py               # 开箱即用的演示脚本
    └── data/
        └── customer_tickets.jsonl  # 30 条客服工单（故意很少）
```

---

## 三个差异化能力

本项目针对文献验证的三个真实市场空白构建，每个能力对应一个核心模块。

### 空白 1 · 零门槛会话（`task_parser.py`）

**现状**：AutoML-GPT 等系统要求用户懂 ML 术语才能使用。  
**本系统**：用户用完全自然的语言描述任务，系统自动推断任务类型、标签体系、评估指标。

```
用户输入 → "帮我识别客服工单是什么类型的投诉"
系统输出 → TaskSpec(
               task_type=CLASSIFICATION,
               label_schema=["账号问题", "支付问题", "物流问题", ...],
               evaluation_metric="f1",
               domain="电商客服"
           )
```

如果描述不清晰，系统会自动追问一个具体问题，不会让用户面对技术参数。

---

### 空白 2 · 可解释中间迭代（`explainer.py`）

**现状**：几乎所有 Agent 式 AutoML 只报告最终指标，迭代过程黑盒。  
**本系统**：每轮训练后生成人类可读的诊断，不是数字，是顾问级分析。

```
第 3 轮结果：F1 = 0.61

诊断：模型在"支付问题"和"账号问题"之间经常混淆，拖低了整体指标。

原因：就像学生做题时把两道相似的应用题搞混了——这两类投诉的语言
     风格太接近（都在说"登不上去""付不了钱"），需要更多区分性样本。

建议：给"支付问题"补充 15 条含有金额/银行卡/支付宝关键词的例子。

决策：📥 追加训练数据
```

---

### 空白 3 · 小数据上游决策（`data_engine.py`）

**现状**：现有 AutoML 工具对任务定义、数据构建的支持极弱，默认数据充足。  
**本系统**：从 10 条样本出发，自动完成三件事：

| 功能 | 说明 |
|------|------|
| **边界样本识别** | 找出标注存在争议的样本，提示用户复查 |
| **均衡数据增强** | 按标签自动增强到可训练规模（10→200+），同时修正类别不平衡 |
| **质量评分** | 综合样本量、类别分布、标签覆盖给出 0-1 质量分 |

---

## 数据格式

训练数据为 **JSONL 格式**，每行一条：

```jsonl
{"text": "我的账号突然无法登录，密码也没错", "label": "账号问题"}
{"text": "付款成功但订单状态还是待支付", "label": "支付问题"}
```

最少 **10 条**即可启动（系统会自动增强）。建议每个标签至少 3 条种子样本。

---

## 完整 CLI 参数

```bash
python run.py [参数]

  --demo              使用内置客服 Demo 数据
  --task   TEXT       自然语言任务描述
  --data   PATH       训练数据文件（JSONL）
  --max-iter  N       最大闭环迭代次数，默认 3
  --target    F       目标指标值（达到后自动停止），默认 0.80
  --predict   TEXT…   训练完成后对指定文本做预测
  --api-key   KEY     Anthropic API Key（也可用环境变量）
  --no-augment        跳过数据增强（快速测试用）
```

---

## 在代码里使用

```python
from core.loop import AutoMLLoop

loop  = AutoMLLoop(api_key="sk-ant-...")
state = loop.run(
    user_description = "帮我把法律合同按风险等级分类：高风险、中风险、低风险",
    examples         = [
        {"text": "乙方须在 3 日内赔偿全部损失", "label": "高风险"},
        {"text": "双方协商解决争议",             "label": "低风险"},
        # ...
    ],
    max_iterations = 3,
    target_metric  = 0.80,
)

# 用最终模型预测
labels = state.final_model.predict(["新合同条款文本"])

# 保存训练记录
from utils.io_utils import save_state
save_state(state, output_dir="outputs/")
```

---

## 自动模型选择逻辑

用户不需要选择模型，系统根据数据量自动决策：

| 数据量 | 自动选择 | 原因 |
|--------|----------|------|
| < 100 条 | TF-IDF + Logistic Regression | 快速收敛，小数据不易过拟合 |
| 100–500 条 | TF-IDF + Linear SVM | 更强的间隔最大化，中等数据效果更好 |
| > 500 条 | TF-IDF + SGD | 支持在线学习，可扩展 |

---

## Phase 2 升级路径（HuggingFace PEFT）

Phase 1 使用 sklearn 验证闭环逻辑。Phase 2 只需替换 `trainer.py` 中的 `_build_clf` 方法，其余模块（解析器、数据引擎、解释器、闭环代理）**无需改动**。

```python
# Phase 2 替换点（trainer.py 第 ~90 行）
# 当前：
clf = LogisticRegression(...)

# 替换为：
from transformers import AutoModelForSequenceClassification
from peft import get_peft_model, LoraConfig
model = AutoModelForSequenceClassification.from_pretrained("bert-base-chinese")
peft_model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, ...))
```

安装 Phase 2 依赖：

```bash
pip install torch transformers peft datasets accelerate
```

---

## 竞品对比定位

| 能力 | Weco AIDE | AutoML-Agent | Karpathy AutoResearch | **本系统** |
|------|-----------|--------------|----------------------|------------|
| 零门槛会话（无需术语） | ❌ | ❌ | ❌ | ✅ |
| 中间过程可解释 | ❌ | ❌ | ❌ | ✅ |
| 小数据专项（<100条） | ❌ | ❌ | ❌ | ✅ |
| 边界样本识别 | ❌ | ❌ | ❌ | ✅ |
| 本地运行（无需云GPU） | ❌ | ❌ | ❌ | ✅ |
| 完整闭环自动化 | ✅ | ✅ | ✅ | ✅ |

---

## 环境要求

- Python 3.10+
- Anthropic API Key（用于任务解析、数据增强、迭代解释）
- 无需 GPU（Phase 1 全部在 CPU 上运行）

---

## Phase 3 · API 服务启动

```bash
pip install fastapi uvicorn websockets pydantic

# 启动 API 服务（含队列调度器 + 前端界面）
cd automl_agent
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# 打开浏览器访问 http://localhost:8000 即可使用完整产品界面
# API 交互文档：http://localhost:8000/docs
```

### Web 前端（真实连接后端，非模拟数据）

`web/index.html` 是一个零构建步骤的单文件前端，通过 WebSocket 直连
`api/routes/tasks.py` 的真实事件流，不是演示用的模拟数据。启动后端后
访问 `http://localhost:8000/` 即可看到：

- 左侧：任务描述 + 训练样本输入表单（`文本 | 标签` 每行一条）
- 右侧：实时学习曲线、每轮迭代的可读诊断（"训练日志"卡片）、部署包信息、
  预测测试框

前端的事件处理逻辑（`web/app.js`）是纯函数、无 DOM 依赖，与渲染完全解耦，
可以直接用 Node 跑单元测试，不需要启动浏览器或后端：

```bash
cd web
node test_app_logic.mjs
# 23/23 tests passed —— 测试用例严格对照 core/pipeline.py 里
# _emit(on_event, {...}) 实际发出的字段构造，不是凭空编的示例数据
```

如果前端要跑在和后端不同的源（比如本地开发时前端用 `python -m http.server`
单独起在别的端口），把 `web/index.html` 里的 `API_BASE` 常量改成后端地址即可，
`api/main.py` 已经开了 CORS。

### API 快速使用

```bash
# 1. 提交训练任务（立即返回 task_id）
curl -X POST http://localhost:8000/api/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "description": "帮我把客服工单按问题分类",
    "examples": [
      {"text": "账号无法登录", "label": "账号问题"},
      {"text": "付款失败",     "label": "支付问题"}
    ]
  }'

# 2. 查询任务状态
curl http://localhost:8000/api/tasks/{task_id}

# 3. 查看队列状态（限流情况）
curl http://localhost:8000/api/tasks/queue/stats

# 4. 部署后反馈（检查是否需要重训）
curl -X POST http://localhost:8000/api/tasks/{task_id}/feedback \
  -H "Content-Type: application/json" \
  -d '{"production_metric": 0.71, "drift_threshold": 0.05}'
```

### WebSocket 实时订阅

```javascript
const ws = new WebSocket('ws://localhost:8000/api/tasks/{task_id}/stream');
ws.onmessage = (e) => {
  const event = JSON.parse(e.data);
  console.log(event.type, event);            // epoch_done / iteration_done / finished
  if (event.type === 'finished') ws.close();
};
```

---

## Phase 4 · 多模态扩展

### 表格数据（CSV）

```python
from core.modalities.tabular import TabularTrainer

trainer, report = TabularTrainer.from_csv("sales.csv", target_col="category")
print(f"F1: {trainer.pipeline.score(X_test, y_test):.4f}")
```

### 持续学习 · 漂移检测

```python
from core.drift_detector import DriftDetector

detector = DriftDetector()
detector.fit_reference(train_texts, train_labels, trainer.vectorizer)

# 每天/每周对生产数据做检测
report = detector.detect(production_texts, production_labels)
if report.is_drifted:
    print(f"⚠  {report.recommendation}")
    # 触发 run_pipeline() 重新训练
```

### 模型市场

```python
from core.marketplace import ModelMarketplace

mp = ModelMarketplace()

# 注册训练好的模型
model_id = mp.register(task_spec, metrics={"f1": 0.85},
                        description="电商客服工单分类 5类", tags=["客服","中文"])

# 搜索可复用模型
results = mp.search(domain="客服", language="zh", min_f1=0.75)

# 查看排行榜
top = mp.top_models(n=5)
```

---

## Phase 5 · 外部计算资源 + 训练产物导出

### 计算资源配置（Slurm / Kubernetes）

头像 →「计算资源」tab 里可以配置外部集群，之后训练任务可以提交到集群上跑，
而不是在 API 服务器本机跑。

```
POST   /api/compute-profiles           创建（k8s: kubeconfig+namespace+镜像；slurm: host+用户名+私钥+分区）
GET    /api/compute-profiles           列表（不回传凭据本身，只回传"已保存/未配置"）
DELETE /api/compute-profiles/{id}      撤销（软删除）
POST   /api/compute-profiles/{id}/test 真实连通性测试
```

`/test` 是真的去连集群，不是格式校验：
- **Kubernetes**：读集群版本 → 查 namespace 是否存在 → 检查有没有读写 Job 的 RBAC 权限，
  三步分开报错（"连不上" / "认证过期" / "namespace 不存在" / "没有 Job 权限" 是四种不同的提示）
- **Slurm**：SSH 登录 → `whoami` → `sinfo --version` → 检查分区是否存在，
  能区分"密钥不对"、"连的不是 Slurm 登录节点"、"分区名写错了"

**当前状态（重要）**：这一期只做到「配置能存下来 + 能测通连接」。
**训练任务还没有真正提交到集群执行**——不管配没配计算资源，训练目前仍然在
API 服务器本机的线程池里跑（`api/worker.py::_run_job` 的 `asyncio.to_thread` 路径）。
真正的远程执行（Dockerfile、`remote_worker/run_remote_job.py`、进度事件回调端点、
`_run_job` 的分发分支）是明确的后续工作。

### 训练产物打包下载

训练完成后，部署卡片上有两个下载出口：

```
GET /api/tasks/{task_id}/export/model   模型包：权重 + inference.py + README + model_card.md
GET /api/tasks/{task_id}/export/code    代码包：inference.py + 生成的架构源码 + train.py + requirements.txt
```

两者是同一个 `deploy/<task_id>/` 目录的两个视图，不重新生成产物：
- **模型包**含权重，是拿去部署、或者上传到 HuggingFace / 魔搭 的
- **代码包不含权重**（代码仓库里塞二进制是反模式），是拿去归档、或者推 GitHub 的
- 打包时会过滤 macOS 的 `._*` AppleDouble 伴生文件（这个仓库挂在不支持原生
  resource fork 的外部卷上，`deploy/` 里真实存在这些文件）
- `task_id` 只接受 UUID 格式，且拼出的路径 `resolve()` 后必须仍在 `deploy/` 内
  （防路径穿越，写法照抄 `core/data_sources.py::LocalUploadSource`）

`train.py` **刻意不内联训练样本**：代码包是可能被推到公开 GitHub 仓库的，
如果原始数据是手动粘贴的敏感业务样本，写进去等于公开泄露。脚本里只描述数据来源，
要复现的人自己提供数据。

---

## ⚠ 发布到 HuggingFace / 魔搭 / GitHub 的注意事项

这部分**尚未实现**，但设计约定先写在这里，避免以后实现时被悄悄绕过：

1. **发布是对外公开动作，且实际不可逆**。即使发布后删除仓库，内容也可能已经被
   搜索引擎 / 第三方镜像缓存。因此：
   - 每次发布必须有**显式确认**（明确列出：推到哪个平台、哪个仓库名、公开还是私有）
   - 仓库可见性**默认私有**，要公开必须用户主动勾选，不能是默认值
   - 后端不提供"静默发布"路径——请求体里没有确认标志就直接拒绝
2. **不会代替用户擅自建仓库**。真实上传验证需要用户自己提供 token、并明确同意做一次
   真实上传、指定目标仓库名和可见性。在那之前只验证到"请求校验 / 参数错配拒绝 /
   无效 token 的错误提示是否清晰"这一档。
3. **三个平台的 token 明文存库**（跟 `ApiToken.llm_api_key` 同一基线）。
   GitHub PAT 尤其敏感——建议创建**只对单个仓库有写权限的 fine-grained token**，
   不要用 classic 全权限 token。
4. **代码包可能带出数据来源描述**。见上面 `train.py` 的说明——默认不内联样本，
   但仍会写"来自哪个数据集 / 多少条手动样本"，介意的话发布前自己检查一遍。

## 已知的安全风险（如实记录，不是待办）

- **计算资源凭据明文存库**：`ComputeResourceProfile` 的 kubeconfig / SSH 私钥
  跟 `ApiToken.llm_api_key` 一样明文存储——必须能反解出原文才能真正拿去连集群，
  不能像密码那样单向哈希。但风险等级不同：泄露 LLM Key 损失的是额度，
  泄露 kubeconfig / SSH 私钥损失的是**整个计算集群的访问权**。
  本项目目前没有字段级加密机制，生产部署强烈建议改接密钥管理服务。
- **Slurm SSH 使用 `AutoAddPolicy`**：服务端无人值守，没有交互确认 host key 的机会，
  首次连接不校验主机身份（理论上可被中间人攻击）。真实 HPC 场景里登录节点通常在
  可信网络内，这个取舍可接受，但要知道它存在。
- **导出下载不做额外鉴权**：拿到 `task_id`(UUID) 就能下载产物，跟现有
  `/api/tasks/{id}/events` 等端点的既有行为一致，这次没有收紧也没有放宽。

---

## 开发者

**Zhongjiang Yao**

## 许可

本项目尚未声明开源许可证。在作者补充 LICENSE 文件之前，默认保留所有权利。
