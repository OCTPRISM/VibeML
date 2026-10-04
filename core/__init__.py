# core/__init__.py
# 懒加载包——不在此处 import 各模块，避免 anthropic 等 API 依赖
# 在需要使用时按需 import：from core.trainer import Trainer
__all__ = ["task_parser","data_engine","trainer","explainer","loop",
           "augmentor","iteration_tree","deployer"]
