/**
 * web/app.js — 纯事件归约逻辑（无 DOM 依赖）
 *
 * 设计原则：把"收到一个后端事件 → 状态该怎么变"这件事完全从
 * DOM 渲染中剥离出来，做成纯函数 reduceEvent(state, event) -> newState。
 * 这样可以在 Node 里对着 core/pipeline.py 真实发出的事件字段直接跑
 * 单元测试，不需要浏览器、不需要真的启动后端。
 *
 * 事件契约来源（严格对齐，不是猜的）：
 *   core/pipeline.py 的 _emit(on_event, {...}) 调用
 *   api/routes/tasks.py 的 WebSocket snapshot/finished/error/heartbeat
 *
 * index.html 只做两件事：
 *   1. 把真实收到的 WebSocket 消息传给 reduceEvent()
 *   2. 把返回的新 state 渲染到页面上
 * 渲染逻辑本身不在这个文件里（无法在 Node 里测，会在浏览器里跑）。
 */

export function initialState() {
  return {
    taskId:        null,
    status:        "idle",       // idle | queued | running | completed | failed
    step:          null,         // 当前处于哪个 step_start 阶段
    stepMessage:   "",
    domain:        null,
    labels:        [],
    metric:        "f1",
    dataReport:    null,         // { total_samples, augmented, quality_score, warnings, boundary_count, label_dist }
    modelName:     null,
    nEpochs:       null,
    iterations:    [],           // 每轮的 { iteration, n_data, augment, epochs:[], explanation, hyperparamChanges }
    currentIterationIdx: -1,
    epochHistory:  [],           // 展平的 { iteration, epoch, val_metric, train_loss, val_loss, per_class, confusion }
    bestMetric:    0,
    flywheel:      null,         // 最近一次 flywheel_done
    deploy:        null,         // deploy_done payload
    feedbackBaseline: null,
    error:         null,
    finishedResult: null,
    archInfo:      null,         // 最近一次 arch_designed（模型后端 != sklearn 时才有）
    nnFallbacks:   [],           // 训练后端降级记录 [{stage, attempt, reason}]
    parallelPrep:  null,         // parallel_prep_done：{dataSeconds, designSeconds, designReused}
    log:           [],           // 完整事件流水（用于调试面板）
  };
}

/**
 * 核心归约函数。absorbs 一个事件，返回全新的 state（不修改传入的 state）。
 */
export function reduceEvent(state, event) {
  const s = { ...state, log: [...state.log, event] };

  switch (event.type) {
    case "snapshot": {
      return {
        ...s,
        taskId: event.task_id,
        status: event.status,
      };
    }

    case "step_start": {
      return { ...s, step: event.step, stepMessage: event.message, status: "running" };
    }

    case "task_parsed": {
      return {
        ...s,
        domain: event.domain,
        labels: event.labels,
        metric: event.metric,
      };
    }

    case "data_ready": {
      return {
        ...s,
        dataReport: {
          totalSamples:  event.total_samples,
          augmented:     event.augmented,
          qualityScore:  event.quality_score,
          warnings:      event.warnings || [],
          boundaryCount: event.boundary_count,
          labelDist:     event.label_dist || {},
        },
      };
    }

    case "model_selected": {
      return { ...s, modelName: event.model_name, nEpochs: event.n_epochs };
    }

    case "parallel_prep_done": {
      return {
        ...s,
        parallelPrep: {
          dataSeconds:   event.data_seconds,
          designSeconds: event.design_seconds,
          designReused:  event.design_reused,
        },
      };
    }

    case "arch_designed": {
      return {
        ...s,
        archInfo: {
          mode:      event.mode,
          className: event.class_name || null,
          code:      event.code || null,
          lossFn:    event.loss_fn || null,
          modelId:   event.model_id || null,
          useLora:   event.use_lora ?? null,
          rationale: event.rationale,
        },
      };
    }

    case "nn_codegen_fallback": {
      return {
        ...s,
        nnFallbacks: [...s.nnFallbacks, {
          stage:   event.stage,
          attempt: event.attempt,
          reason:  event.reason,
        }],
      };
    }

    case "iteration_start": {
      const iterations = [...s.iterations, {
        iteration: event.iteration,
        nData:     event.n_data,
        augment:   null,
        epochs:    [],
        explanation: null,
        hyperparamChanges: null,
      }];
      return { ...s, iterations, currentIterationIdx: iterations.length - 1 };
    }

    case "augment_done": {
      const iterations = _updateCurrentIteration(s, (it) => ({
        ...it,
        augment: {
          original:       event.original,
          augmented:      event.augmented,
          balanceBefore:  event.balance_before,
          balanceAfter:   event.balance_after,
        },
      }));
      return { ...s, iterations };
    }

    case "epoch_done": {
      const epochEntry = {
        iteration:  event.iteration,
        epoch:      event.epoch,
        valMetric:  event.val_metric,
        trainLoss:  event.train_loss,
        valLoss:    event.val_loss,
        perClass:   event.per_class || {},
        confusion:  event.confusion || [],
      };
      const iterations = _updateCurrentIteration(s, (it) => ({
        ...it,
        epochs: [...it.epochs, epochEntry],
      }));
      const bestMetric = Math.max(s.bestMetric, event.val_metric);
      return {
        ...s,
        iterations,
        epochHistory: [...s.epochHistory, epochEntry],
        bestMetric,
      };
    }

    case "iteration_done": {
      const explanation = {
        valMetric:      event.val_metric,
        diagnosis:      event.diagnosis,
        rootCause:      event.root_cause,
        recommendation: event.recommendation,
        nextAction:     event.next_action,
        confidence:     event.confidence,
      };
      const iterations = _updateCurrentIteration(s, (it) => ({ ...it, explanation }));
      return { ...s, iterations };
    }

    case "hyperparams_adjusted": {
      const iterations = _updateCurrentIteration(s, (it) => ({
        ...it,
        hyperparamChanges: event.changes || {},
      }));
      return { ...s, iterations };
    }

    case "flywheel_done": {
      return {
        ...s,
        flywheel: {
          removed:       event.removed,
          remaining:     event.remaining,
          avgConfidence: event.avg_confidence,
        },
      };
    }

    case "deploy_done": {
      if (event.error) {
        return { ...s, deploy: { error: event.error } };
      }
      return {
        ...s,
        deploy: {
          format:       event.format,
          modelPath:    event.model_path,
          sizeKb:       event.size_kb,
          labels:       event.labels,
          usageExample: event.usage_example,
        },
      };
    }

    case "feedback_check": {
      return {
        ...s,
        feedbackBaseline: {
          action:          event.action,
          metricName:      event.metric_name,
          baselineMetric:  event.baseline_metric,
          recordedAt:      event.recorded_at,
          nTrainSamples:   event.n_train_samples,
        },
      };
    }

    case "finished": {
      return {
        ...s,
        status: "completed",
        finishedResult: {
          bestMetric:   event.best_metric,
          metricName:   event.metric_name,
          labels:       event.labels,
          domain:       event.domain,
          nSamples:     event.n_samples,
          epochHistory: event.epoch_history || [],
          deployPath:   event.deploy_path,
        },
      };
    }

    case "error": {
      return { ...s, status: "failed", error: event.message || event.error || "未知错误" };
    }

    case "heartbeat": {
      return s; // no-op，只是保活，不改状态（log 里仍会留一条记录用于调试）
    }

    default: {
      // 未知事件类型：不崩溃，原样记录到 log 里，方便发现后端加了新事件类型
      // 而前端还没跟上
      return s;
    }
  }
}

/** 内部辅助：更新 iterations 数组里"当前迭代"那一条，不改变其余条目 */
function _updateCurrentIteration(state, updateFn) {
  if (state.currentIterationIdx < 0) return state.iterations;
  return state.iterations.map((it, idx) =>
    idx === state.currentIterationIdx ? updateFn(it) : it
  );
}

/**
 * 便捷函数：把一串事件（比如 WebSocket 收到的历史消息）依次喂给 reduceEvent，
 * 返回最终 state。主要用于测试，也可以在 index.html 里做"重放调试面板日志"用。
 */
export function reduceAll(events, seedState) {
  return events.reduce(reduceEvent, seedState || initialState());
}
