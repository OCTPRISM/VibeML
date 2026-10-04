/**
 * web/test_app_logic.mjs
 *
 * 纯 Node 单元测试，零依赖（不需要 npm install，不需要浏览器，不需要
 * 真的启动后端）。直接跑：
 *
 *   node web/test_app_logic.mjs
 *
 * 每个事件 fixture 都是从 core/pipeline.py 的 _emit() 调用和
 * api/routes/tasks.py 的 WebSocket 处理逻辑里逐字段抄出来的，不是猜的
 * 示例数据 —— 这样测试真正验证的是"前端认不认识后端实际会发的消息"，
 * 而不是"前端认不认识我自己瞎编的消息"。
 */

import { initialState, reduceEvent, reduceAll } from "./app.js";
import assert from "node:assert/strict";

let passed = 0, failed = 0;
function test(name, fn) {
  try {
    fn();
    console.log(`PASS: ${name}`);
    passed++;
  } catch (e) {
    console.log(`FAIL: ${name}\n  ${e.message}`);
    failed++;
  }
}

// ── 完整真实事件流（严格来自 core/pipeline.py 各 _emit 调用的字段）──────────

const REAL_EVENT_SEQUENCE = [
  { type: "snapshot", task_id: "abc-123", status: "queued", progress: {}, created_at: "2026-07-17T10:00:00Z" },
  { type: "step_start", step: "parsing", message: "正在理解任务描述…" },
  { type: "task_parsed", task_type: "classification", domain: "电商客服",
    labels: ["账号问题", "支付问题", "物流问题", "产品质量", "其他"], metric: "f1" },
  { type: "step_start", step: "data", message: "正在分析和增强数据…" },
  { type: "data_ready", total_samples: 200, augmented: 170, quality_score: 0.87,
    warnings: ["类别轻微不平衡"], boundary_count: 3,
    label_dist: { "账号问题": 42, "支付问题": 40, "物流问题": 39, "产品质量": 41, "其他": 38 } },
  { type: "model_selected", model_name: "TF-IDF + Logistic Regression（小数据专用）", n_epochs: 6 },
  { type: "iteration_start", iteration: 1, n_data: 200 },
  { type: "augment_done", original: 160, augmented: 200, balance_before: 0.82, balance_after: 0.97 },
  { type: "epoch_done", iteration: 1, epoch: 1, val_metric: 0.41, train_loss: 0.89, val_loss: 0.95,
    per_class: { "账号问题": 0.3 }, confusion: ["'支付问题' 被误判为 '账号问题'（3 次）"] },
  { type: "epoch_done", iteration: 1, epoch: 2, val_metric: 0.53, train_loss: 0.71, val_loss: 0.80,
    per_class: { "账号问题": 0.5 }, confusion: [] },
  { type: "iteration_done", iteration: 1, val_metric: 0.74,
    diagnosis: "模型初步学到了基本区分能力", root_cause: "样本还不够区分细粒度差异",
    recommendation: "补充支付问题的样本", next_action: "collect_more_data", confidence: 0.78 },
  { type: "iteration_start", iteration: 2, n_data: 260 },
  { type: "flywheel_done", removed: 12, remaining: 248, avg_confidence: 0.71 },
  { type: "augment_done", original: 198, augmented: 248, balance_before: 0.89, balance_after: 0.98 },
  { type: "epoch_done", iteration: 2, epoch: 1, val_metric: 0.85, train_loss: 0.17, val_loss: 0.22,
    per_class: { "账号问题": 0.86 }, confusion: [] },
  { type: "iteration_done", iteration: 2, val_metric: 0.85,
    diagnosis: "模型表现优秀！F1 已达 85%", root_cause: "追加样本让模型学到了关键区分词",
    recommendation: "可以停止训练", next_action: "stop_success", confidence: 0.91 },
  { type: "step_start", step: "deploy", message: "正在导出部署包（含 INT8 量化）…" },
  { type: "deploy_done", format: "onnx", model_path: "./deploy/task-abc/model_int8.onnx",
    size_kb: 234.5, labels: ["账号问题", "支付问题", "物流问题", "产品质量", "其他"],
    usage_example: "from inference import predict\nprint(predict(['示例文本']))" },
  { type: "feedback_check", action: "baseline_recorded", metric_name: "f1",
    baseline_metric: 0.85, recorded_at: "2026-07-17T10:05:00Z", n_train_samples: 248 },
  { type: "finished", status: "completed", best_metric: 0.85, metric_name: "f1",
    labels: ["账号问题", "支付问题", "物流问题", "产品质量", "其他"], domain: "电商客服",
    n_samples: 248, epoch_history: [
      { iteration: 1, epoch: 1, val_metric: 0.41, train_loss: 0.89 },
      { iteration: 2, epoch: 1, val_metric: 0.85, train_loss: 0.17 },
    ], deploy_path: "./deploy/task-abc" },
];

// ── 测试 ──────────────────────────────────────────────────────────────────

test("initialState 结构完整且状态为 idle", () => {
  const s = initialState();
  assert.equal(s.status, "idle");
  assert.deepEqual(s.iterations, []);
  assert.equal(s.bestMetric, 0);
});

test("snapshot 事件正确写入 taskId 和 status", () => {
  const s = reduceEvent(initialState(), REAL_EVENT_SEQUENCE[0]);
  assert.equal(s.taskId, "abc-123");
  assert.equal(s.status, "queued");
});

test("step_start 事件更新 step/stepMessage 并把 status 置为 running", () => {
  const s = reduceEvent(initialState(), { type: "step_start", step: "parsing", message: "x" });
  assert.equal(s.step, "parsing");
  assert.equal(s.status, "running");
});

test("task_parsed 事件正确解析 domain/labels/metric（含中文标签）", () => {
  const s = reduceEvent(initialState(), REAL_EVENT_SEQUENCE[2]);
  assert.equal(s.domain, "电商客服");
  assert.equal(s.labels.length, 5);
  assert.equal(s.labels[0], "账号问题");
  assert.equal(s.metric, "f1");
});

test("data_ready 事件正确映射 snake_case -> camelCase 且保留 label_dist", () => {
  const s = reduceEvent(initialState(), REAL_EVENT_SEQUENCE[4]);
  assert.equal(s.dataReport.totalSamples, 200);
  assert.equal(s.dataReport.augmented, 170);
  assert.equal(s.dataReport.qualityScore, 0.87);
  assert.equal(s.dataReport.warnings.length, 1);
  assert.equal(s.dataReport.labelDist["账号问题"], 42);
});

test("data_ready 事件在 warnings/label_dist 缺失时不崩溃（防御性）", () => {
  const s = reduceEvent(initialState(), {
    type: "data_ready", total_samples: 10, augmented: 0, quality_score: 1.0,
    boundary_count: 0,
    // warnings 和 label_dist 故意不传，模拟未来后端字段变化
  });
  assert.deepEqual(s.dataReport.warnings, []);
  assert.deepEqual(s.dataReport.labelDist, {});
});

test("iteration_start 创建新的迭代记录并把 currentIterationIdx 指向它", () => {
  let s = reduceEvent(initialState(), REAL_EVENT_SEQUENCE[6]); // iteration_start iter=1
  assert.equal(s.iterations.length, 1);
  assert.equal(s.iterations[0].iteration, 1);
  assert.equal(s.currentIterationIdx, 0);
});

test("augment_done 事件正确挂载到当前迭代，不影响其他迭代", () => {
  let s = reduceAll(REAL_EVENT_SEQUENCE.slice(0, 8)); // 到第一个 augment_done 为止
  assert.equal(s.iterations[0].augment.original, 160);
  assert.equal(s.iterations[0].augment.balanceAfter, 0.97);
});

test("epoch_done 事件同时更新 iterations[].epochs 和全局 epochHistory", () => {
  let s = reduceAll(REAL_EVENT_SEQUENCE.slice(0, 10)); // 到第二个 epoch_done
  assert.equal(s.iterations[0].epochs.length, 2);
  assert.equal(s.epochHistory.length, 2);
  assert.equal(s.epochHistory[1].valMetric, 0.53);
});

test("bestMetric 在多轮 epoch_done 中正确取最大值（不会被后面更低的值覆盖）", () => {
  let s = reduceAll(REAL_EVENT_SEQUENCE); // 跑完整个序列
  assert.equal(s.bestMetric, 0.85);
});

test("iteration_done 事件正确挂载 explanation（含 next_action 映射）", () => {
  let s = reduceAll(REAL_EVENT_SEQUENCE.slice(0, 11)); // 到第一个 iteration_done
  assert.equal(s.iterations[0].explanation.diagnosis, "模型初步学到了基本区分能力");
  assert.equal(s.iterations[0].explanation.nextAction, "collect_more_data");
  assert.equal(s.iterations[0].explanation.confidence, 0.78);
});

test("第二轮 iteration_start 后，第一轮的数据保持不变（隔离性）", () => {
  let s = reduceAll(REAL_EVENT_SEQUENCE.slice(0, 12)); // 到第二个 iteration_start
  assert.equal(s.iterations.length, 2);
  assert.equal(s.iterations[0].epochs.length, 2); // 第一轮的仍是 2 个 epoch，没被清空
  assert.equal(s.iterations[1].epochs.length, 0); // 第二轮刚开始，还没有 epoch
  assert.equal(s.currentIterationIdx, 1);
});

test("hyperparams_adjusted 事件正确挂载到当前迭代（自动调参闭环，core/pipeline.py 新增事件）", () => {
  const seq = [
    { type: "iteration_start", iteration: 1, n_data: 200 },
    { type: "iteration_done", iteration: 1, val_metric: 0.62,
      diagnosis: "训练损失下降但验证指标不上升", root_cause: "模型开始死记硬背训练数据",
      recommendation: "收紧正则强度", next_action: "adjust_hyperparams", confidence: 0.7 },
    { type: "hyperparams_adjusted", iteration: 1, changes: { C: 0.35, class_boost: { "支付问题": 1.3 } } },
  ];
  let s = reduceAll(seq);
  assert.equal(s.iterations.length, 1);
  assert.deepEqual(s.iterations[0].hyperparamChanges, { C: 0.35, class_boost: { "支付问题": 1.3 } });
});

test("parallel_prep_done 事件正确记录数据准备/模型设计并行耗时（Part C 新增事件）", () => {
  const seq = [
    { type: "parallel_prep_done", data_seconds: 12.34, design_seconds: 12.5, design_reused: true },
  ];
  let s = reduceAll(seq);
  assert.equal(s.parallelPrep.dataSeconds, 12.34);
  assert.equal(s.parallelPrep.designSeconds, 12.5);
  assert.equal(s.parallelPrep.designReused, true);
});

test("arch_designed 事件正确挂载 custom_nn 生成代码信息（core/pipeline.py _select_and_train_backend 新增事件）", () => {
  const seq = [
    { type: "arch_designed", mode: "custom_nn", class_name: "SimpleClassifier",
      code: "import torch\nclass SimpleClassifier: pass", loss_fn: "cross_entropy",
      rationale: "样本少，用精简两层网络防止过拟合" },
  ];
  let s = reduceAll(seq);
  assert.equal(s.archInfo.mode, "custom_nn");
  assert.equal(s.archInfo.className, "SimpleClassifier");
  assert.equal(s.archInfo.lossFn, "cross_entropy");
  assert.ok(s.archInfo.code.includes("SimpleClassifier"));
});

test("arch_designed 事件正确挂载 pretrained_nn 选型信息", () => {
  const seq = [
    { type: "arch_designed", mode: "pretrained_nn", model_id: "hfl/chinese-macbert-base",
      use_lora: true, rationale: "中文任务优先选 MacBERT，样本少用 LoRA 省算力" },
  ];
  let s = reduceAll(seq);
  assert.equal(s.archInfo.mode, "pretrained_nn");
  assert.equal(s.archInfo.modelId, "hfl/chinese-macbert-base");
  assert.equal(s.archInfo.useLora, true);
});

test("nn_codegen_fallback 事件累积记录每一次降级（不覆盖前一次）", () => {
  const seq = [
    { type: "nn_codegen_fallback", stage: "custom_nn", attempt: 1, reason: "语法错误" },
    { type: "nn_codegen_fallback", stage: "custom_nn", attempt: 2, reason: "参数量超限" },
    { type: "nn_codegen_fallback", stage: "pretrained_nn", attempt: 1, reason: "下载超时" },
  ];
  let s = reduceAll(seq);
  assert.equal(s.nnFallbacks.length, 3);
  assert.equal(s.nnFallbacks[0].stage, "custom_nn");
  assert.equal(s.nnFallbacks[2].stage, "pretrained_nn");
});

test("flywheel_done 事件正确映射字段", () => {
  let s = reduceAll(REAL_EVENT_SEQUENCE.slice(0, 13));
  assert.equal(s.flywheel.removed, 12);
  assert.equal(s.flywheel.remaining, 248);
  assert.equal(s.flywheel.avgConfidence, 0.71);
});

test("deploy_done 正常情况：格式/路径/标签/用法示例全部正确映射", () => {
  let s = reduceAll(REAL_EVENT_SEQUENCE.slice(0, 18)); // 含下标17 deploy_done
  assert.equal(s.deploy.format, "onnx");
  assert.equal(s.deploy.sizeKb, 234.5);
  assert.equal(s.deploy.labels.length, 5);
  assert.ok(s.deploy.usageExample.includes("predict"));
});

test("deploy_done 失败情况：只有 error 字段时正确识别为失败，不误读其他字段", () => {
  const s = reduceEvent(initialState(), { type: "deploy_done", error: "磁盘空间不足" });
  assert.equal(s.deploy.error, "磁盘空间不足");
  assert.equal(s.deploy.format, undefined);
});

test("feedback_check 事件正确映射部署后反馈基线字段", () => {
  let s = reduceAll(REAL_EVENT_SEQUENCE.slice(0, 19)); // 含下标18 feedback_check
  assert.equal(s.feedbackBaseline.action, "baseline_recorded");
  assert.equal(s.feedbackBaseline.baselineMetric, 0.85);
  assert.equal(s.feedbackBaseline.nTrainSamples, 248);
});

test("finished 事件把 status 置为 completed 并完整保存最终结果", () => {
  let s = reduceAll(REAL_EVENT_SEQUENCE); // 完整跑完
  assert.equal(s.status, "completed");
  assert.equal(s.finishedResult.bestMetric, 0.85);
  assert.equal(s.finishedResult.epochHistory.length, 2);
  assert.equal(s.finishedResult.deployPath, "./deploy/task-abc");
});

test("error 事件正确处理 message 字段（worker.py 失败路径）", () => {
  const s = reduceEvent(initialState(), { type: "error", message: "未提供 ANTHROPIC_API_KEY" });
  assert.equal(s.status, "failed");
  assert.equal(s.error, "未提供 ANTHROPIC_API_KEY");
});

test("error 事件在只有 error 字段（而非 message）时也能兜住（WS 路由的另一种形状）", () => {
  const s = reduceEvent(initialState(), { type: "error", error: "connection lost" });
  assert.equal(s.status, "failed");
  assert.equal(s.error, "connection lost");
});

test("heartbeat 事件是纯粹的 no-op，不改变除 log 外的任何状态", () => {
  const before = reduceEvent(initialState(), REAL_EVENT_SEQUENCE[2]); // 先塞点真实状态
  const after = reduceEvent(before, { type: "heartbeat" });
  assert.equal(after.domain, before.domain);
  assert.equal(after.status, before.status);
  assert.equal(after.log.length, before.log.length + 1); // 只有 log 变长
});

test("未知事件类型不抛异常，且被记录到 log 供调试", () => {
  const s = reduceEvent(initialState(), { type: "some_future_event_type", foo: "bar" });
  assert.equal(s.log.length, 1);
  assert.equal(s.log[0].type, "some_future_event_type");
});

test("reduceAll 对完整真实事件序列端到端跑通，最终 state 内部一致", () => {
  const s = reduceAll(REAL_EVENT_SEQUENCE);
  assert.equal(s.log.length, REAL_EVENT_SEQUENCE.length);
  assert.equal(s.iterations.length, 2);
  assert.equal(s.status, "completed");
  // 交叉校验：finishedResult.bestMetric 应该等于逐 epoch 算出来的 bestMetric
  assert.equal(s.finishedResult.bestMetric, s.bestMetric);
});

test("reduceEvent 不修改传入的原 state（不可变性，防止渲染时状态串扰）", () => {
  const s0 = initialState();
  const s1 = reduceEvent(s0, REAL_EVENT_SEQUENCE[2]);
  assert.deepEqual(s0.labels, []); // 原对象必须还是空的
  assert.notEqual(s1.labels.length, 0);
});

// ── 汇总 ──────────────────────────────────────────────────────────────────

console.log(`\n${passed}/${passed + failed} tests passed`);
if (failed > 0) process.exit(1);
