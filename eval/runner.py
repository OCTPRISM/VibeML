"""
eval/runner.py  -  Phase 3：基准测试主运行器

论文实验流程：
  1. 加载所有数据集
  2. 对每个数据集按 SAMPLE_SIZES 子采样
  3. 每个系统 × 每个数据集规模运行 N_REPEATS 次（控制方差）
  4. 输出 CSV + Markdown 格式的对比表

运行方式：
  cd automl_agent
  python -m eval.runner                         # 快速模式（无 API）
  python -m eval.runner --full --api-key sk-... # 完整模式（含 LLM 增强）
  python -m eval.runner --dataset newsgroups_4cls --sizes 10 50 100
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from eval.datasets  import load_all, SAMPLE_SIZES, BenchmarkDataset
from eval.baselines import EvalResult, get_all_systems


N_REPEATS = 3   # 每个配置重复次数（控制随机方差）


# ── 主运行逻辑 ────────────────────────────────────────────────────────────────

class BenchmarkRunner:
    """
    论文实验的核心运行器。

    用法：
        runner = BenchmarkRunner(output_dir="outputs/benchmark")
        runner.run(sample_sizes=[10, 50, 100], n_repeats=3)
        runner.save_results()
        runner.print_table()
    """

    def __init__(self, output_dir: str = "outputs/benchmark"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[EvalResult] = []

    def run(
        self,
        datasets:     Dict[str, BenchmarkDataset] | None = None,
        sample_sizes: List[int] = SAMPLE_SIZES,
        n_repeats:    int = N_REPEATS,
        include_autogluon: bool = False,
    ):
        """
        运行完整基准测试。

        Args:
            datasets:     数据集字典（None 则自动加载所有）
            sample_sizes: 每类样本数列表
            n_repeats:    重复次数
            include_autogluon: 是否包含 AutoGluon 基线
        """
        if datasets is None:
            print("📦 加载数据集...")
            datasets = load_all()

        systems = get_all_systems(include_autogluon=include_autogluon)
        print(f"🔬 系统：{[s.name for s in systems]}")
        print(f"📊 样本档位：{sample_sizes}")
        print(f"🔁 重复次数：{n_repeats}\n")

        total = len(datasets) * len(sample_sizes) * len(systems) * n_repeats
        done  = 0

        for ds_name, dataset in datasets.items():
            for n_per_class in sample_sizes:
                # 子采样一次只是为了检查训练集是否够大；真正参与训练的子采样在下面
                # repeat 循环内部按 seed=42+repeat 重新做一次，见下方注释
                sub = dataset.subsample(n_per_class)

                if sub.n_train < dataset.n_classes:
                    continue   # 训练集太小，跳过

                for system in systems:
                    run_results = []

                    for repeat in range(n_repeats):
                        done += 1
                        pct = done / total * 100
                        print(
                            f"  [{pct:5.1f}%] {ds_name:25s} n={n_per_class:3d}  "
                            f"{system.name:20s} repeat={repeat+1}",
                            end="  ",
                        )

                        # 关键修复：之前这里复用循环外那一份固定 seed=42 的子采样，
                        # 而 Trainer/sklearn 内部的 random_state 全部硬编码——导致
                        # n_repeats 次跑的是完全相同的数据、完全相同的模型初始化，
                        # std_f1 恒为 0，"重复实验"名不副实。这里按 repeat 变化
                        # 子采样的 seed，让每次 repeat 抽到不同的具体样本，
                        # 从而产生真实方差（Trainer 内部固定 random_state=42 本身
                        # 不动，如实写进论文 Limitations，不是这里能改的范围）。
                        sub_r = dataset.subsample(n_per_class, seed=42 + repeat)

                        try:
                            t0 = time.time()
                            result = system.fit_predict(
                                sub_r.train_texts, sub_r.train_labels,
                                sub_r.test_texts,  sub_r.test_labels,
                            )
                            result.dataset_name = ds_name
                            result.n_per_class  = n_per_class
                            run_results.append(result)
                            print(f"F1={result.f1_weighted:.4f}  ({time.time()-t0:.1f}s)")

                        except Exception as e:
                            print(f"ERROR: {e}")

                    # 平均多次重复的结果
                    if run_results:
                        avg = self._average_results(run_results)
                        self.results.append(avg)

        print(f"\n✅ 完成 {len(self.results)} 个配置")

    # ── 输出 ──────────────────────────────────────────────────────────────────

    def save_results(self):
        """保存原始结果为 JSON 和 CSV"""
        # JSON
        json_path = self.output_dir / "raw_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump([self._result_to_dict(r) for r in self.results], f,
                      ensure_ascii=False, indent=2)

        # CSV
        csv_path = self.output_dir / "results.csv"
        if self.results:
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._result_to_dict(self.results[0]).keys())
                writer.writeheader()
                for r in self.results:
                    writer.writerow(self._result_to_dict(r))

        print(f"💾 结果已保存：{json_path}, {csv_path}")
        return json_path, csv_path

    def _render_table(self, metric: str = "f1_weighted") -> str:
        """构建 Markdown 表格文本（纯字符串拼接，供 print_table 打印到控制台/写入文件共用，
        不做"重定向 stdout 再调用自己"这种会无限递归的写法）"""
        if not self.results:
            return "没有结果可展示"

        systems  = sorted(set(r.system_name for r in self.results))
        datasets = sorted(set(r.dataset_name for r in self.results))
        sizes    = sorted(set(r.n_per_class  for r in self.results))

        lines = []
        for ds_name in datasets:
            lines.append(f"\n### {ds_name}\n")

            header = "| System | " + " | ".join(f"n={n}" for n in sizes) + " |"
            sep    = "|--------|" + "|".join(["--------"] * len(sizes)) + "|"
            lines.append(header)
            lines.append(sep)

            for sys_name in systems:
                row = f"| {sys_name:20s} |"
                for n in sizes:
                    match = [
                        r for r in self.results
                        if r.system_name == sys_name
                        and r.dataset_name == ds_name
                        and r.n_per_class == n
                    ]
                    if match:
                        val = getattr(match[0], metric)
                        row += f" {val:.4f} |"
                    else:
                        row += "    —   |"
                lines.append(row)

        return "\n".join(lines)

    def print_table(self, metric: str = "f1_weighted"):
        """打印 Markdown 格式的对比表到控制台（直接可以粘贴进论文），同时保存到文件"""
        text = self._render_table(metric)
        print(text)

        if not self.results:
            return
        md_path = self.output_dir / "table.md"
        md_path.write_text(text, encoding="utf-8")
        print(f"\n📄 Markdown 表格已保存：{md_path}")

    def gain_over_baseline(
        self,
        our_system:   str = "Ours-SMOTE",
        baseline:     str = "TF-IDF + LogReg",
        metric:       str = "f1_weighted",
    ) -> Dict:
        """
        计算我们的系统相对于基线的平均提升。
        用于填写论文 Abstract 中的数字。
        """
        gains = []
        for r_ours in self.results:
            if r_ours.system_name != our_system:
                continue
            match = [
                r for r in self.results
                if r.system_name == baseline
                and r.dataset_name == r_ours.dataset_name
                and r.n_per_class  == r_ours.n_per_class
            ]
            if match:
                ours_val = getattr(r_ours, metric)
                base_val = getattr(match[0], metric)
                gains.append(ours_val - base_val)

        if not gains:
            return {}

        return {
            "mean_gain":    round(float(np.mean(gains)),  4),
            "std_gain":     round(float(np.std(gains)),   4),
            "max_gain":     round(float(np.max(gains)),   4),
            "n_configs":    len(gains),
            "positive_pct": round(sum(g > 0 for g in gains) / len(gains), 3),
        }

    # ── 私有辅助 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _average_results(results: List[EvalResult]) -> EvalResult:
        """对多次重复的结果取均值"""
        r0 = results[0]
        return EvalResult(
            system_name  = r0.system_name,
            dataset_name = r0.dataset_name,
            n_per_class  = r0.n_per_class,
            f1_weighted  = round(float(np.mean([r.f1_weighted  for r in results])), 4),
            accuracy     = round(float(np.mean([r.accuracy     for r in results])), 4),
            train_time_s = round(float(np.mean([r.train_time_s for r in results])), 3),
            n_train      = r0.n_train,
            n_test       = r0.n_test,
            extra        = {"std_f1": round(float(np.std([r.f1_weighted for r in results])), 4)},
        )

    @staticmethod
    def _result_to_dict(r: EvalResult) -> dict:
        return {
            "system":      r.system_name,
            "dataset":     r.dataset_name,
            "n_per_class": r.n_per_class,
            "n_train":     r.n_train,
            "n_test":      r.n_test,
            "f1_weighted": r.f1_weighted,
            "accuracy":    r.accuracy,
            "train_time_s": r.train_time_s,
            "std_f1":      r.extra.get("std_f1", 0),
        }


# ── CLI 入口 ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Vibe ML Studio 基准测试")
    parser.add_argument("--dataset",  nargs="+", default=None,
                        help="指定数据集（默认全部）")
    parser.add_argument("--sizes",    nargs="+", type=int, default=SAMPLE_SIZES,
                        help="每类样本数档位")
    parser.add_argument("--repeats",  type=int, default=N_REPEATS)
    parser.add_argument("--autogluon", action="store_true",
                        help="包含 AutoGluon 基线（需另行安装）")
    parser.add_argument("--output",   default="outputs/benchmark")
    args = parser.parse_args()

    runner = BenchmarkRunner(output_dir=args.output)

    datasets = load_all()
    if args.dataset:
        datasets = {k: v for k, v in datasets.items() if k in args.dataset}

    runner.run(
        datasets          = datasets,
        sample_sizes      = args.sizes,
        n_repeats         = args.repeats,
        include_autogluon = args.autogluon,
    )
    runner.save_results()
    runner.print_table()

    gain = runner.gain_over_baseline()
    if gain:
        print(f"\n📈 Ours-SMOTE vs TF-IDF+LogReg：平均 F1 提升 {gain['mean_gain']:+.4f} "
              f"（{gain['positive_pct']:.0%} 的配置有正向提升）")


if __name__ == "__main__":
    main()
