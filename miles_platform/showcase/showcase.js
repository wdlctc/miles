"use strict";

const repository = "https://github.com/wdlctc/miles/tree/codex/customer-platform";
const guide = "https://github.com/wdlctc/miles/blob/codex/customer-platform/miles_platform/README.md";
const pageNames = {overview: "概览", training: "训练结果", model: "模型与部署", playground: "Playground 记录"};
let record;
let selectedMetric = "reward";
const metricNames = {reward: "原始奖励均值", loss: "训练损失", grad_norm: "梯度范数"};
const content = document.querySelector("#content");
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[char]));
const scalar = (value) => Math.abs(value) > 0 && Math.abs(value) < 0.001 ? value.toExponential(3) : Number(value.toFixed(4)).toString();
const link = (label, url, primary = false) => `<a class="button ${primary ? "primary" : ""}" href="${url}">${label} <span aria-hidden="true">↗</span></a>`;
const metric = (label, value, note) => `<div class="metric"><div class="metric-top">${label}<span class="metric-mark" aria-hidden="true">▦</span></div><div class="metric-value">${value}</div><small>${note}</small></div>`;
const date = (timestamp) => new Date(timestamp * 1000).toISOString().replace("T", " ").slice(0, 19) + " UTC";
const head = (title, subtitle) => `<div class="page-head"><div><div class="eyebrow">MILES PLATFORM / PUBLIC SHOWCASE</div><h1>${title}</h1><p>${subtitle}</p></div>${link("查看源码", repository)}</div>`;
const snapshotNotice = () => `<p class="notice showcase-note">本页展示 ${record.snapshot_date} 已完成的真实运行记录，非实时监控。公开页面提供只读浏览；训练、在线推理与账户管理在自行部署的平台中使用。</p>`;

function chart(field, featured = false) {
  const samples = record.telemetry.samples;
  const low = Math.min(...samples.map((sample) => sample[field]));
  const high = Math.max(...samples.map((sample) => sample[field]));
  const points = samples.map((sample, index) => [40 + index / (samples.length - 1) * 540,
    high === low ? 105 : 180 - (sample[field] - low) / (high - low) * 150]);
  return `<section class="training-chart ${featured ? "featured-chart" : ""}"><h3>${metricNames[field]}</h3><svg viewBox="0 0 620 215" role="img" aria-label="${metricNames[field]}：第 1 至 ${samples.length} 步的实际测量值"><path class="chart-axis" d="M40 15V180H590"/><polyline points="${points.map((point) => point.join(",")).join(" ")}"/>${points.map((point, index) => `<circle cx="${point[0]}" cy="${point[1]}" r="4"><title>第 ${samples[index].step} 步：${scalar(samples[index][field])}</title></circle>`).join("")}<text x="40" y="205" font-size="10" fill="#666963">第 1 步</text><text x="540" y="205" font-size="10" fill="#666963">第 ${samples.length} 步</text></svg><small>范围 ${scalar(low)} 至 ${scalar(high)} · 每个点对应一次训练更新</small></section>`;
}

function overview() {
  const {telemetry, model} = record;
  const minutes = Math.floor(telemetry.elapsed_seconds / 60);
  const seconds = Math.floor(telemetry.elapsed_seconds % 60);
  return `<section class="overview-intro"><div class="intro-copy"><div class="eyebrow">YOUR MODELS. YOUR INFRASTRUCTURE.</div><h1>你的模型。<br>从训练到上线。</h1><p>一份数据，一次迭代，一个可调用的模型。<br>用真实训练记录，探索 Miles 的后训练与推理工作流。</p><div class="actions">${link("查看训练结果", "#training", true)}${link("GitHub 源码", repository)}</div></div><div class="grid-art" aria-hidden="true"><div class="grid-tiles">${"<i></i>".repeat(35)}</div><div class="grid-caption">01 TRAIN　 /　 02 EVALUATE　 /　 03 SERVE</div></div></section>
    <div class="metrics">${metric("完成训练更新", `${telemetry.completed_updates} / ${telemetry.total_updates}`, "Qwen3-0.6B · FSDP GRPO")}${metric("任务用时", `${minutes}m ${seconds}s`, "包含训练、导出与独立评测")}${metric("独立评测", `${telemetry.evaluation.correct} / ${telemetry.evaluation.total}`, "小样本工程验证 · 不代表业务质量")}${metric("实际更新张量", model.metadata.validation.changed_tensors, `共 ${model.metadata.validation.unique_tensors} 个唯一张量`)}</div>
    <div class="grid-main"><section class="card"><div class="card-head"><h2>一轮训练的完整路径</h2><span class="pill">已完成</span></div><ol class="steps"><li><span class="step-number">01</span><div><h3>训练</h3><p>${record.training_gpus} 张 A100 · ${telemetry.total_updates} 次优化器更新</p></div><a class="text-link" href="#training">看曲线 ↗</a></li><li><span class="step-number">02</span><div><h3>评测与模型导出</h3><p>独立评分、权重变更检查与 SHA-256 校验</p></div><a class="text-link" href="#model">看模型 ↗</a></li><li><span class="step-number">03</span><div><h3>部署与调用</h3><p>普通响应、SSE 流式响应和用量计量已验收</p></div><a class="text-link" href="#playground">看记录 ↗</a></li></ol></section>
    <section class="card hero-card"><div class="card-head"><h2>把工作流留在自己的平台</h2><span class="pill">开源代码</span></div><div class="card-body"><h3>训练、评测、上线，统一管理。</h3><p>工作区与角色、模型版本、推理 API、用量记录，以及 Stripe 结账集成。</p><div class="endpoint">Miles + SGLang<br>Qwen3-0.6B / FSDP GRPO</div><small>当前为平台 beta。真实支付和公开客户服务尚未启用。</small>${link("查看部署指南", guide)}</div></section></div>
    <div class="training-charts">${chart("reward")}${chart("loss")}</div>${snapshotNotice()}`;
}

function training() {
  const {job, telemetry} = record;
  const last = telemetry.samples.at(-1);
  return `${head("训练结果", "每次更新都有记录。查看实际奖励、损失、梯度与学习率。")}
    <section class="card training-monitor"><div class="card-head"><div><h2>${escapeHtml(job.name)}</h2><small>${job.id}</small></div><span class="pill">已完成</span></div><div class="card-body"><div class="detail-line"><strong>${record.base_model} · ${record.recipe}</strong><span>${date(job.started)}</span></div><label>训练更新 ${telemetry.completed_updates} / ${telemetry.total_updates}<progress class="gpu-progress training-progress" value="${telemetry.completed_updates}" max="${telemetry.total_updates}" aria-label="训练更新进度"></progress></label>
    <div class="metrics training-metrics">${metric("最终原始奖励", scalar(last.reward), "归一化前的评分")}${metric("最终训练损失", scalar(last.loss), "GRPO loss")}${metric("最终梯度范数", scalar(last.grad_norm), "Grad norm")}${metric("学习率", scalar(last.learning_rate), "实际优化器学习率")}</div>
    <div class="chart-toolbar"><h2>逐步指标</h2><label class="chart-selector"><span class="sr-only">图表指标</span><select id="metric-select" aria-label="图表指标">${Object.entries(metricNames).map(([key, name]) => `<option value="${key}" ${key === selectedMetric ? "selected" : ""}>${name}</option>`).join("")}</select></label></div><div id="selected-chart">${chart(selectedMetric, true)}</div>
    <details class="training-samples"><summary>查看每步指标（${telemetry.samples.length} 条）</summary><div class="table-wrap"><table><thead><tr><th>更新步</th><th>原始奖励</th><th>Loss</th><th>梯度范数</th><th>回答 Tokens</th><th>截断比例</th></tr></thead><tbody>${telemetry.samples.map((sample) => `<tr><td>${sample.step}</td><td>${scalar(sample.reward)}</td><td>${scalar(sample.loss)}</td><td>${scalar(sample.grad_norm)}</td><td>${scalar(sample.response_tokens)}</td><td>${scalar(sample.truncated_fraction)}</td></tr>`).join("")}</tbody></table></div></details>
    <p class="notice quality-note">独立评测 ${telemetry.evaluation.correct} / ${telemetry.evaluation.total}，达到本次工程验证门槛。没有同配置基线对照，不能据此认定训练提升了准确率。</p><div class="actions"><a class="button" href="data.json" download="miles-training-record.json">下载运行记录 JSON ↓</a>${link("查看模型产物", "#model", true)}</div></div></section>${snapshotNotice()}`;
}

function model() {
  const {model, deployment, api_acceptance: api} = record;
  const validation = model.metadata.validation;
  return `${head("模型与部署", "从训练任务追踪到模型产物，再到 API 调用记录。")}
    <div class="showcase-columns"><section class="card"><div class="card-head"><h2>模型版本</h2><span class="pill">权重已验证</span></div><div class="card-body"><h3>${escapeHtml(model.name)}</h3><div class="detail-line"><span>模型 ID</span><strong>${model.id}</strong></div><div class="detail-line"><span>来源任务</span><strong>${model.job_id}</strong></div><div class="detail-line"><span>张量更新</span><strong>${validation.changed_tensors} / ${validation.unique_tensors}</strong></div><div class="detail-line"><span>数值检查</span><strong>${validation.all_finite ? "全部为有限值" : "未通过"}</strong></div><label>权重 SHA-256</label><code class="artifact-hash">${model.sha256}</code><p class="section-note">公开页提供模型产物的校验信息；模型权重保存在运行训练的平台服务器上。</p></div></section>
    <section class="card"><div class="card-head"><h2>部署验收</h2><span class="pill neutral">历史成功记录</span></div><div class="card-body"><div class="detail-line"><span>部署 ID</span><strong>${deployment.id}</strong></div><div class="detail-line"><span>模型别名</span><strong>${api.alias}</strong></div><div class="detail-line"><span>普通 API 响应</span><strong>HTTP ${api.nonstream.status}</strong></div><div class="detail-line"><span>流式 API 响应</span><strong>HTTP ${api.stream.status} · [DONE]</strong></div><div class="detail-line"><span>未认证 / 已撤销密钥</span><strong>${api.unauthenticated_status} / ${api.revoked_key_status}</strong></div><p class="section-note">验收时间：${date(api.verified_at)}。此处不探测服务器的当前状态。</p>${link("查看调用记录", "#playground", true)}</div></section></div>${snapshotNotice()}`;
}

function apiExample() {
  return 'curl "$MILES_BASE_URL/v1/chat/completions" \\\n  -H "Authorization: Bearer $MILES_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d \'{"model":"miles-default","messages":[{"role":"user","content":"17 + 25 = ?"}],"stream":true}\'';
}

function playground() {
  const api = record.api_acceptance;
  const example = api.quality_example;
  return `${head("Playground 记录", "查看真实 API 验收样例。这里展示保存的响应，不执行新的推理请求。")}
    <div class="playground"><section class="chat-panel record-card"><div class="card-head"><h2>${api.alias}</h2><span class="pill neutral">历史调用 · HTTP ${api.nonstream.status}</span></div><div class="chat-messages"><div class="message"><span class="avatar">U</span><div class="message-content"><b>用户输入</b>${escapeHtml(api.prompt)}</div></div><div class="message"><span class="avatar mint">M</span><div class="message-content"><b>模型响应 · 已保存</b>${escapeHtml(api.nonstream.answer)}</div></div></div><div class="chat-entry"><p>输入 ${api.nonstream.usage.prompt_tokens} Tokens · 输出 ${api.nonstream.usage.completion_tokens} Tokens · 单次耗时 ${api.nonstream.elapsed_seconds}s</p><small>仅为这次请求的测量值，不代表延迟基准或服务承诺。</small></div></section><div><section class="card"><div class="card-head"><h2>使用自己的部署</h2></div><div class="card-body"><p>部署平台后，使用工作区 API Key 调用统一网关，也可以在登录后的 Playground 自由提问。</p><div class="actions showcase-note">${link("部署指南", guide, true)}</div></div></section></div></div>
    <section class="card showcase-note"><div class="card-head"><h2>API 调用示例</h2><button id="copy-api" type="button">复制示例</button></div><div class="card-body"><p>先设置自己的 MILES_BASE_URL 和 MILES_API_KEY；此公开展示页不提供推理端点。</p><pre id="api-example">${escapeHtml(apiExample())}</pre><small id="copy-status" role="status"></small></div></section>
    <section class="card"><div class="card-head"><h2>质量边界也保留在记录里</h2><span class="pill pending">待改善</span></div><div class="card-body"><p>工程链路通过验收，并不意味着模型已经适合业务。下面是一条真实未答对的样例：</p><div class="result-block"><p>${escapeHtml(example.prompt)}</p><p><strong>模型答案：${example.observed_answer}</strong>　/　正确答案：${example.expected_answer}</p></div><p class="section-note">当前只完成小样本流程验证。业务发布前，需要独立评测集、基线对比和更多质量验证。</p></div></section>${snapshotNotice()}`;
}

function render() {
  if (!record) return;
  const requested = location.hash.slice(1);
  const page = Object.hasOwn(pageNames, requested) ? requested : "overview";
  document.title = `${pageNames[page]} · Miles Platform`;
  document.querySelector("#page-name").textContent = pageNames[page];
  document.querySelectorAll("#navigation a").forEach((anchor) => {
    const active = anchor.hash === `#${page}`;
    anchor.classList.toggle("active", active);
    if (active) anchor.setAttribute("aria-current", "page");
    else anchor.removeAttribute("aria-current");
  });
  content.innerHTML = ({overview, training, model, playground})[page]();
}

document.addEventListener("change", (event) => {
  if (event.target.id !== "metric-select") return;
  selectedMetric = event.target.value;
  document.querySelector("#selected-chart").innerHTML = chart(selectedMetric, true);
});
document.addEventListener("click", async (event) => {
  if (event.target.closest(".skip-link")) {
    event.preventDefault();
    content.focus();
    return;
  }
  if (!event.target.closest("#copy-api")) return;
  const status = document.querySelector("#copy-status");
  try {
    await navigator.clipboard.writeText(apiExample());
    status.textContent = "已复制。请替换为你自己的平台地址与密钥。";
  } catch {
    status.textContent = "无法访问剪贴板，请选中上方示例手动复制。";
  }
});
window.addEventListener("hashchange", () => { render(); window.scrollTo(0, 0); });
fetch("data.json").then((response) => {
  if (!response.ok) throw new Error("Record unavailable");
  return response.json();
}).then((data) => { record = data; render(); }).catch(() => {
  content.innerHTML = `<h1>记录暂时无法读取</h1><p>请刷新重试，或<a class="text-link" href="${repository}">在 GitHub 查看源码</a>。</p>`;
});
