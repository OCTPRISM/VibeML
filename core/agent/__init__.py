"""
core/agent/  -  Multi-Agent 编排模式（"传统方式"以外的可选项）。

设计原则（详见实施计划）：训练/评估/部署/回滚 100% 复用 core/pipeline.py 等
现有确定性实现，这里的 Agent 只通过工具调用去驱动它们，从不重新实现或绕过
沙盒校验（core/nn_sandbox.py/core/rl_sandbox.py）、子进程隔离
（core/subprocess_runner.py）或真实指标计算——"实验结果由独立、不可修改的
评估系统裁决"，不是 Agent 自己认定成功与否。
"""
