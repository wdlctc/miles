"use strict";
let selectedTrainingJob = null, trainingData = null, trainingError = "";

function trainingJob() {
  const jobs = state.jobs.filter((j) => j.kind === "train");
  return jobs.find((j) => j.id === selectedTrainingJob) || jobs[0];
}

async function refreshTraining() {
  if (pane !== "jobs") return;
  const job = trainingJob();
  if (!job) return;
  selectedTrainingJob = job.id;
  try {
    const data = await (await api(`/api/jobs/${job.id}/telemetry`)).json();
    if (selectedTrainingJob !== job.id) return;
    trainingData = data;
    trainingError = "";
  } catch (error) {
    if (selectedTrainingJob === job.id) trainingError = error.message;
  }
}

function scalar(value) {
  if (value === null || value === undefined) return "—";
  return value !== 0 && Math.abs(value) < 0.001
    ? value.toExponential(3) : Number(value.toFixed(4)).toString();
}

function trainingChart(samples, field, title) {
  const points = samples.filter((p) => p[field] !== undefined && p[field] !== null);
  if (!points.length) return `<section class="training-chart"><h3>${title}</h3><p>等待首个实测指标…</p></section>`;
  const low = Math.min(...points.map((p) => p[field]));
  const high = Math.max(...points.map((p) => p[field]));
  const lastStep = Math.max(2, ...samples.map((p) => p.step));
  const xy = points.map((p) => [24 + (p.step - 1) / (lastStep - 1) * 392,
    high === low ? 60 : 100 - (p[field] - low) / (high - low) * 80]);
  return `<section class="training-chart"><h3>${title}</h3><svg viewBox="0 0 440 120" role="img" aria-label="${title}，按训练更新步数绘制"><path class="chart-axis" d="M24 12V104H420"/><polyline points="${xy.map((p) => p.join(",")).join(" ")}"/>${xy.map((p, i) => `<circle cx="${p[0]}" cy="${p[1]}" r="3"><title>第 ${points[i].step} 步：${scalar(points[i][field])}</title></circle>`).join("")}</svg><small>第 1–${Math.max(...samples.map((p) => p.step))} 步 · 范围 ${scalar(low)} 至 ${scalar(high)}</small></section>`;
}

function trainingMonitor() {
  const job = trainingJob();
  if (!job) return "";
  const data = trainingData?.job_id === job.id ? trainingData : null;
  const samples = data?.samples || [];
  const trained = samples.filter((p) => p.completed).at(-1);
  const rewarded = samples.filter((p) => p.reward !== undefined).at(-1);
  const total = data?.total_updates ?? job.params.updates;
  const done = data?.completed_updates ?? 0;
  const seconds = Math.floor(data?.elapsed_seconds || 0);
  const selection = `<label class="training-select">查看训练任务<select id="training-job-select">${state.jobs.filter((j) => j.kind === "train").map((j) => `<option value="${j.id}" ${j.id === job.id ? "selected" : ""}>${escape(j.name)}</option>`).join("")}</select></label>`;
  return `<section class="card training-monitor"><div class="card-head"><div><h2>训练实时监控</h2><small>${escape(job.id)} · 每 5 秒刷新</small></div>${pill(job.status)}</div><div class="card-body">${selection}<div class="detail-line"><strong>${escape(job.phase)}</strong><span>${data ? `${Math.floor(seconds / 60)} 分 ${seconds % 60} 秒` : "正在读取…"}</span></div><label>训练更新 ${done} / ${total || "—"}<progress class="gpu-progress training-progress" max="${total || 1}" value="${done}" aria-label="训练更新进度"></progress></label><small>更新完成后继续导出权重与独立评测；总任务状态以右上角为准。</small>${trainingError ? `<p class="error" role="alert">刷新失败：${escape(trainingError)}。当前为上次成功读取的数据，连接恢复后自动重试。</p>` : ""}${data?.scan_truncated || data?.invalid_metrics ? `<p class="error">${data.scan_truncated ? "日志超过扫描上限，指标可能不完整。" : "检测到非有限指标，已标为缺失，请检查日志。"}</p>` : ""}<div class="metrics training-metrics">${metric("原始奖励均值", scalar(rewarded?.reward), rewarded ? `第 ${rewarded.step} 步 · 归一化前的评分` : "等待 rollout 完成")}${metric("训练损失", scalar(trained?.loss), trained ? `第 ${trained.step} 步 · GRPO loss` : "等待优化器更新")}${metric("梯度范数", scalar(trained?.grad_norm), "Grad norm · 来自训练器")}${metric("学习率", scalar(trained?.learning_rate), "实际优化器学习率")}</div><div class="training-charts">${trainingChart(samples, "reward", "奖励曲线")}${trainingChart(samples, "loss", "损失曲线")}</div>${data?.evaluation ? `<p class="notice">独立评测：${data.evaluation.correct} / ${data.evaluation.total} · ${data.evaluation.passed ? "达到工程发布门槛" : "未达到工程发布门槛"}</p>` : ""}<details class="training-samples"><summary>查看每步指标（${samples.length} 条）</summary><div class="table-wrap"><table><thead><tr><th>更新步</th><th>已更新</th><th>原始奖励</th><th>Loss</th><th>梯度范数</th><th>回答 Tokens</th><th>截断比例</th></tr></thead><tbody>${samples.map((p) => `<tr><td>${p.step}</td><td>${p.completed ? "是" : "等待训练"}</td><td>${scalar(p.reward)}</td><td>${scalar(p.loss)}</td><td>${scalar(p.grad_norm)}</td><td>${scalar(p.response_tokens)}</td><td>${scalar(p.truncated_fraction)}</td></tr>`).join("")}</tbody></table></div></details><div class="training-log-head"><h3>实时日志</h3><small>${data ? `最近读取 ${new Date(data.observed_at * 1000).toLocaleTimeString("zh-CN", {hour12: false})}` : "正在连接"} · 最近 64 KB</small></div><pre id="training-log" tabindex="0" aria-label="训练实时日志">${escape(data?.logs || (job.params.imported ? "历史导入任务未采集逐步指标，日志保留在服务器。" : "等待训练进程输出日志…"))}</pre><small>日志在底部时自动跟随；向上滚动可暂停跟随。指标来自 Miles 训练器，缺失值显示为 —。</small></div></section>`;
}

document.addEventListener("change", async (event) => {
  if (event.target.id !== "training-job-select") return;
  selectedTrainingJob = event.target.value;
  trainingData = null;
  trainingError = "";
  await refreshTraining();
  if (pane === "jobs") render();
});
