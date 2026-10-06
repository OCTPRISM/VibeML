# 参与贡献

## 开发环境

```bash
git clone https://github.com/OCTPRISM/VibeML.git
cd VibeML
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # 核心
pip install -r requirements-nn.txt         # 需要神经网络后端时
cp .env.example .env
python -m uvicorn api.main:app --reload --port 8000
```

LLM 建议用本地 Ollama，开发期间不产生任何 API 费用。

## 提交前请跑

```bash
node web/test_app_logic.mjs       # 前端状态归约，应为 30/30
python -m py_compile $(git ls-files '*.py')
```

## 这个项目最在意的一件事

**不要让系统报告自己无法支撑的结论。**

本项目的核心主张是「指标可信」，所以涉及评估、指标、迭代判定的改动，
请一并给出验证：改动前后的真实数字、样本量、以及该差异是否超过噪声下界
（`core/robust_eval.py::binomial_standard_error` 可直接调用）。

一个具体的反例：曾有一版把「类别不平衡」映射到 focal loss，听起来合理，
但实测只有 +0.004（噪声下界 0.011），等于没有作用。这类改动不应合入。

## 代码约定

- 注释说明**为什么**这么写，特别是绕过某个坑的地方——
  仓库里很多注释记录了真实踩过的问题，请保持这个习惯
- 新增事件字段是安全的；修改或删除 `web/app.js::reduceEvent` 已消费的字段
  会破坏前端契约，必须同步更新测试
- LLM 生成代码的执行路径（`core/nn_sandbox.py` / `core/rl_sandbox.py`）
  改动需格外谨慎，这是安全边界
