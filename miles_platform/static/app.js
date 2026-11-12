"use strict";
const $ = (s) => document.querySelector(s);
const escape = (s) =>
  String(s ?? "").replace(
    /[&<>"']/g,
    (c) =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[
        c
      ],
  );
const number = (n) => Number(n || 0).toLocaleString("zh-CN");
const date = (n) =>
  n
    ? new Date(n * 1000).toLocaleString("zh-CN", {
        month: "2-digit",
        day: "2-digit",
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      })
    : "—";
const money = (n) =>
  n === null || n === undefined ? "未配置" : "$" + Number(n).toFixed(4);
const titles = {
  overview: "概览",
  jobs: "训练任务",
  models: "模型仓库",
  deployments: "部署",
  playground: "Playground",
  keys: "API 密钥",
  usage: "用量与费用",
  datasets: "数据集",
  billing: "账单 · Billing",
  team: "团队 · Team",
  account: "账户 · Account",
  customers: "客户 · Customers",
};
const paths = {
  overview: "M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z",
  jobs: "M7 3v5m10-5v5M4 8h16v13H4zM8 13h8m-8 4h5",
  models: "M12 3l9 5-9 5-9-5zM3 12l9 5 9-5M3 16l9 5 9-5",
  deployments: "M3 8h18v12H3zM7 16h1m4-12v9m-4-4 4 4 4-4",
  playground: "M8 5l11 7-11 7z",
  keys: "M14 11a5 5 0 1 0-2 3l3 3h3v3h3v-5l-6-6",
  usage: "M4 20V10m8 10V4m8 16v-7M2 21h20",
  datasets:
    "M4 5c0-4 16-4 16 0s-16 4-16 0v14c0 4 16 4 16 0V5M4 12c0 4 16 4 16 0",
  billing: "M3 5h18v14H3zM3 9h18M7 15h4",
  team: "M9 12a4 4 0 1 0 0-8 4 4 0 0 0 0 8M2 21v-2a7 7 0 0 1 14 0v2M17 5a4 4 0 0 1 0 8m1 3a5 5 0 0 1 4 5",
  account: "M12 13a5 5 0 1 0 0-10 5 5 0 0 0 0 10M3 22a9 9 0 0 1 18 0",
  customers: "M4 21V3h12v18M16 9h4v12M8 7h4M8 11h4M8 15h4M2 21h20",
};
let state,
  runtime = { gpus: [] },
  pane = "overview",
  logJob = null,
  toastTimer,
  busyChat = false;
let conversation = [];

async function api(url, options = {}) {
  const response = await fetch(url, {
    credentials: "same-origin",
    ...options,
    headers: { "Content-Type": "application/json", ...options.headers },
  });
  if (response.status === 401 && url !== "/api/login") {
    showLogin();
    throw new Error("请重新登录");
  }
  if (!response.ok) {
    let detail = "请求失败";
    try {
      const body = await response.json();
      detail =
        typeof body.detail === "string"
          ? body.detail
          : JSON.stringify(body.detail);
    } catch (_) {}
    throw new Error(detail);
  }
  return response;
}
const post = async (url, body = {}, idem = false) =>
  (
    await api(url, {
      method: "POST",
      body: JSON.stringify(body),
      headers: idem ? { "Idempotency-Key": crypto.randomUUID() } : {},
    })
  ).json();
function toast(text) {
  $("#toast").textContent = text;
  $("#toast").hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => ($("#toast").hidden = true), 5500);
}
function showLogin() {
  $("#login").hidden = false;
  $("#shell").hidden = true;
}
function pill(status) {
  const labels = {
    queued: "排队中",
    running: "运行中",
    succeeded: "已完成",
    failed: "失败",
    cancelled: "已取消",
    cancel_requested: "取消中",
  };
  const cls = {
    queued: "pending",
    running: "running",
    succeeded: "",
    failed: "failed",
    cancelled: "neutral",
    cancel_requested: "pending",
  };
  return `<span class="pill ${cls[status] || ""}">${labels[status] || escape(status)}</span>`;
}
function head(title, description, action = "") {
  return `<div class="page-head"><div><span class="eyebrow">MILES PLATFORM / ${pane.toUpperCase()}</span><h1>${title}</h1><p>${description}</p></div><div class="actions">${action}</div></div>`;
}
function button(label, action, id = "", cls = "") {
  if (!allowedAction(action)) return "";
  return `<button class="${cls}" data-action="${action}" data-id="${escape(id)}">${label}</button>`;
}
function empty(text, detail, action = "") {
  return `<div class="empty"><div class="empty-symbol">◇</div><h3>${text}</h3><p>${detail}</p>${action}</div>`;
}
function metric(label, value, note, word = false) {
  return `<div class="metric"><div class="metric-top">${label}<span class="metric-mark" aria-hidden="true">▦</span></div><div class="metric-value ${word ? "word" : ""}">${escape(value)}</div><small>${note}</small></div>`;
}
function modelName(id) {
  return state.models.find((m) => m.id === id)?.name || id || "—";
}
function latestEval(id) {
  return state.evaluations.find((e) => e.model_id === id);
}
function jobTable(jobs, all = false) {
  if (!jobs.length)
    return empty(
      "准备开始一次迭代",
      "选择数据集，启动一次真实的 GPU 训练。",
      button("创建训练任务", "new-job", "", "primary"),
    );
  return `<div class="table-wrap"><table><thead><tr><th>任务</th><th>状态</th><th>${all ? "执行阶段" : "创建时间"}</th><th>操作</th></tr></thead><tbody>${jobs.map((j) => `<tr><td><strong>${escape(j.name)}</strong><small>${escape(j.kind.toUpperCase())} · ${escape(j.id.slice(-8))}${j.params.imported ? " · 历史导入" : ""}</small></td><td>${pill(j.status)}</td><td>${all ? escape(j.phase) : date(j.created)}${j.error ? `<small class="error">${escape(j.error.slice(0, 100))}</small>` : ""}</td><td><div class="actions">${button("日志", "logs", j.id, "tiny")}${["queued", "running"].includes(j.status) ? button("取消", "cancel", j.id, "tiny danger") : ""}</div></td></tr>`).join("")}</tbody></table></div>`;
}
function overview() {
  const t = state.totals,
    active = state.active,
    working = state.job_totals.active || 0;
  return (
    `<section class="overview-intro"><div class="intro-copy"><span class="eyebrow">YOUR MODELS. YOUR INFRASTRUCTURE.</span><h1>你的模型。<br>从训练到上线。</h1><p>在一个工作区，完成训练、评测与部署。<br>让每一次迭代，都有迹可循。</p><div class="actions">${button("创建训练任务 ＋", "new-job", "", "primary")}${button("打开 Playground ↗", "playground")}</div></div><div class="grid-art" aria-hidden="true"><div class="grid-tiles">${"<i></i>".repeat(35)}</div><span class="grid-caption">01 TRAIN &nbsp; / &nbsp; 02 EVALUATE &nbsp; / &nbsp; 03 SERVE</span></div></section>` +
    `<div class="metrics">${metric("推理请求", number(t.requests), "来自 API 与 Playground 的真实请求")}${metric("累计 Tokens", number((t.prompt_tokens || 0) + (t.completion_tokens || 0)), "输入与输出 · 按响应中的 usage 计量")}${metric("训练与操作", number(state.job_totals.jobs), working ? `${working} 个任务正在执行或排队` : "当前队列空闲")}${metric("当前服务版本", active ? modelName(active.model_id) : "尚未部署", "统一端点 miles-default", true)}</div>
    <div class="grid-main"><section class="card"><div class="card-head"><h2>最近任务</h2><a class="link-button" href="#jobs">查看全部 →</a></div>${jobTable(state.jobs.slice(0, 5))}</section>
    <section class="card hero-card"><div class="card-head"><h2>在线推理</h2><span class="pill">${runtime.serving_healthy ? "● 服务健康" : "检查中"}</span></div><div class="card-body"><h3>${escape(active ? modelName(active.model_id) : "等待部署")}</h3><p>独立推理服务 · OpenAI 兼容接口</p><div class="endpoint">${location.origin}/v1</div><div class="detail-line"><span>模型别名</span><strong>miles-default</strong></div><div class="detail-line"><span>请求保护</span><strong>API Key + 请求限额</strong></div>${button("在 Playground 中试用 ↗", "playground")}</div></section></div>
    <section class="card"><div class="card-head"><h2>GPU 资源</h2><small>GPU 0 用于推理 · GPU 1–3 用于训练</small></div><div class="resource-list">${runtime.gpus.length ? runtime.gpus.map((g) => `<div class="resource"><div class="resource-top"><strong>GPU ${g.index} · A100</strong><span class="pill neutral">${g.utilization}%</span></div><progress class="gpu-progress" max="100" value="${(g.used_mb / g.total_mb) * 100}"></progress><small>${(g.used_mb / 1024).toFixed(1)} / ${(g.total_mb / 1024).toFixed(0)} GiB 显存</small></div>`).join("") : "<small>正在读取 GPU 监控…</small>"}</div></section>
    <div class="notice">↗ <span>当前配方为 Qwen3-0.6B 数学任务 GRPO。小样本用于验证流程；上线前请使用与你的业务一致的独立评测集。</span></div>`
  );
}
function jobs() {
  return (
    head(
      "训练任务",
      "从训练到导出与评测，跟踪每一步执行。",
      button("创建训练任务 ＋", "new-job", "", "primary"),
    ) +
    trainingMonitor() +
    `<section class="card"><div class="card-head"><h2>任务记录</h2><small>${state.jobs.length} 个记录 · 自动刷新</small></div>${jobTable(state.jobs, true)}</section>`
  );
}
function models() {
  return (
    head(
      "模型仓库",
      "每个版本保留训练来源、权重摘要和独立评测。",
      button("训练新版本 ＋", "new-job", "", "primary"),
    ) +
    `<div class="notice">✓ 发布门槛：权重校验通过、最新评测正确率 ≥ ${state.quality_threshold * 100}%，且没有输出截断。这是工程门槛，不代表业务效果达标。</div><section class="card"><div class="card-head"><h2>模型版本</h2><small>${state.models.length} 个版本</small></div><div class="table-wrap"><table><thead><tr><th>模型版本</th><th>完整性</th><th>最新评测</th><th>创建时间</th><th>操作</th></tr></thead><tbody>${state.models
      .map((m) => {
        const e = latestEval(m.id);
        return `<tr><td><div class="model-cell"><span class="model-icon">◇</span><div><strong>${escape(m.name)}</strong><small>${escape(m.id)} · ${m.sha256.slice(0, 10)}</small></div></div></td><td><span class="pill ${m.validated ? "" : "pending"}">${m.validated ? "校验通过" : "待校验"}</span></td><td>${e ? `<strong>${e.correct}/${e.total}</strong> <span class="pill ${e.passed ? "" : "pending"}">${e.passed ? "达到门槛" : "未达标"}</span>` : "尚未评测"}</td><td>${date(m.created)}</td><td><div class="actions">${button("详情", "model-detail", m.id, "tiny")}${button("评测", "evaluate", m.id, "tiny")}${state.active?.model_id === m.id ? '<span class="pill">● 在线</span>' : button("部署", "deploy", m.id, "tiny primary")}</div></td></tr>`;
      })
      .join("")}</tbody></table></div></section>`
  );
}
function deployments() {
  const active = state.active;
  return (
    head("部署", "在统一端点后管理模型版本与回滚。") +
    `<div class="split"><section class="card"><div class="card-head"><h2>生产入口</h2><span class="pill ${runtime.serving_healthy ? "" : "pending"}">${runtime.serving_healthy ? "● 健康" : "待就绪"}</span></div><div class="card-body"><h3>${escape(active ? modelName(active.model_id) : "尚未部署")}</h3><div class="detail-line"><span>稳定模型名</span><strong>miles-default</strong></div><div class="detail-line"><span>当前版本</span><strong>${escape(active?.model_id)}</strong></div><div class="detail-line"><span>最近切换</span><strong>${date(active?.created)}</strong></div>${button("测试当前版本 ↗", "playground", "", "primary")}</div></section><section class="card"><div class="card-head"><h2>接入你的应用</h2>${button("复制示例", "copy-code", "", "tiny")}</div><div class="card-body"><pre>${escape(curlExample())}</pre><small>API Key 在密钥页面创建。切换版本不需要修改客户端模型名。</small></div></section></div>
    <section class="card"><div class="card-head"><h2>发布历史</h2><small>切换前始终检查候选服务健康状态</small></div><div class="table-wrap"><table><thead><tr><th>模型版本</th><th>操作</th><th>时间</th><th>状态</th></tr></thead><tbody>${state.deployments.map((d) => `<tr><td><strong>${escape(modelName(d.model_id))}</strong><small>${escape(d.id)}</small></td><td>${{ import: "初始版本", deploy: "部署", rollback: "回滚" }[d.action]}</td><td>${date(d.created)}</td><td>${active?.id === d.id ? '<span class="pill">● 当前服务</span>' : button("回滚到此版本", "rollback", d.id, "tiny")}</td></tr>`).join("")}</tbody></table></div></section>`
  );
}
function keys() {
  return (
    head(
      "API 密钥",
      "创建独立密钥并控制每分钟请求数；密钥原文仅显示一次。",
      button("创建 API Key ＋", "new-key", "", "primary"),
    ) +
    `<section class="card"><div class="card-head"><h2>密钥管理</h2><small>Bearer token 鉴权</small></div>${state.keys.length ? `<div class="table-wrap"><table><thead><tr><th>名称</th><th>密钥</th><th>请求限额</th><th>创建时间</th><th>状态</th></tr></thead><tbody>${state.keys.map((k) => `<tr><td><strong>${escape(k.name)}</strong></td><td><code>${escape(k.prefix)}</code></td><td>${k.rpm} 次 / 分钟</td><td>${date(k.created)}</td><td>${k.revoked ? '<span class="pill neutral">已撤销</span>' : button("撤销", "revoke", k.id, "tiny danger")}</td></tr>`).join("")}</tbody></table></div>` : empty("创建第一个 API Key", "让你的应用连接在线模型。", button("创建密钥", "new-key", "", "primary"))}</section>
    <div class="notice">⌁ 密钥经过散列存储。撤销后，新请求会立即被拒绝；已开始的请求继续完成。</div>`
  );
}
function usage() {
  const seconds = state.job_totals.training_gpu_seconds || 0;
  const trainingCosts = state.job_totals.training_cost || 0;
  const totalCost = (state.totals.cost || 0) + trainingCosts,
    configured = Object.values(state.rates).some((v) => v !== null);
  return (
    head(
      "用量与费用",
      "查看真实请求、Token 与训练资源用量。",
      `<a href="/api/usage.csv"><button>导出 CSV ↓</button></a>${button("设置估算单价", "rates", "", "primary")}`,
    ) +
    `<div class="metrics">${metric("累计请求", number(state.totals.requests), `${number(state.totals.successful)} 次成功`)}${metric("累计 Tokens", number((state.totals.prompt_tokens || 0) + (state.totals.completion_tokens || 0)), `${number(state.totals.unmetered)} 次请求未获得完整用量`)}${metric("训练 GPU 小时", (seconds / 3600).toFixed(3), "按分配的 3 张 GPU × 训练执行时长")}${metric("已计量费用估算", configured ? money(totalCost) : "未配置单价", "仅包含有价格快照与完整计量的记录", true)}</div>
    <div class="notice warn">费用为你配置单价后的估算，不是供应商账单。历史记录保留当时单价；未配置单价、断流或未计量部分不补算，未包含空闲 GPU、存储和网络费用。</div>
    <section class="card"><div class="card-head"><h2>最近推理请求</h2><small>包含 Playground · 最近 50 条</small></div>${state.usage.length ? `<div class="table-wrap"><table><thead><tr><th>请求</th><th>时间</th><th>状态</th><th>输入 / 输出 Tokens</th><th>费用估算</th></tr></thead><tbody>${state.usage.map((r) => `<tr><td><code>${r.id.slice(-12)}</code><small>${r.key_id === "console" ? "Playground" : escape(state.keys.find((k) => k.id === r.key_id)?.name || r.key_id)}</small></td><td>${date(r.created)}</td><td><span class="pill ${r.status === 200 ? "" : r.status === 102 ? "pending" : "failed"}">${r.status === 102 ? "进行中" : r.status}</span></td><td>${number(r.prompt_tokens)} / ${number(r.completion_tokens)}${!r.complete ? "<small>用量不完整</small>" : ""}</td><td>${money(r.cost)}</td></tr>`).join("")}</tbody></table></div>` : empty("还没有推理用量", "在 Playground 发出第一条请求，或使用 API Key 接入应用。", button("打开 Playground", "playground", "", "primary"))}</section>`
  );
}
function datasets() {
  return (
    head(
      "数据集",
      "上传数学任务 JSONL，训练与评测数据分开保存。",
      button("上传数据集 ＋", "new-dataset", "", "primary"),
    ) +
    `<section class="card"><div class="card-head"><h2>版本化数据</h2><small>以内容 SHA-256 标识每个数据集</small></div><div class="table-wrap"><table><thead><tr><th>数据集</th><th>训练样本</th><th>评测样本</th><th>内容摘要</th><th>操作</th></tr></thead><tbody>${state.datasets.map((d) => `<tr><td><strong>${escape(d.name)}</strong><small>${escape(d.id)}</small></td><td>${d.train_rows}</td><td>${d.eval_rows}</td><td><code>${d.sha256.slice(0, 14)}</code></td><td>${button("用于训练", "new-job", d.id, "tiny")}</td></tr>`).join("")}</tbody></table></div></section><div class="notice">训练集支持 4–5,000 条，评测集支持 1–64 条。系统拒绝重复 prompt 以及训练、评测集交叉；奖励按数学答案计算。</div>`
  );
}
function playground() {
  return (
    head("Playground", "直接试用当前在线版本，响应与用量来自真实模型。") +
    `<div class="playground"><section class="chat-panel"><div class="card-head"><h2>miles-default</h2><span class="pill">${escape(modelName(state.active?.model_id))}</span></div><div id="chat-messages" class="chat-messages"><div class="chat-welcome"><span class="brand-mark">M</span><h2>开始与模型对话</h2><p>试试一道数学题，或测试你的业务提示词。</p></div></div><form id="chat-form" class="chat-entry"><textarea id="chat-input" placeholder="输入你的问题…" aria-label="消息" required maxlength="5000"></textarea><div><span>输出以流式返回 · 最大 512 Tokens</span><button type="submit" class="primary">发送 ↑</button></div></form></section><div><section class="card"><div class="card-head"><h2>生成设置</h2>${button("清空", "clear-chat", "", "quiet")}</div><div class="card-body"><label>Temperature<input id="temperature" type="number" min="0" max="2" step="0.1" value="0.7"></label><label>最大输出 Tokens<input id="max-tokens" type="number" min="1" max="512" value="256"></label><label>System prompt<textarea id="system-prompt" placeholder="You are a helpful assistant."></textarea></label><small>上下文窗口为 2,048 Tokens。模型输出可能不准确，请按你的评测标准验证。</small></div></section><div class="notice">请求通过统一网关执行，并计入“用量与费用”。</div></div></div>`
  );
}
function curlExample() {
  return `curl ${location.origin}/v1/chat/completions \\\n  -H "Authorization: Bearer YOUR_API_KEY" \\\n  -H "Content-Type: application/json" \\\n  -d '{"model":"miles-default","messages":[{"role":"user","content":"What is 17 + 25?"}],"stream":true}'`;
}

function render() {
  if (!state) return;
  const logView = $("#training-log");
  const logScroll = logView?.scrollTop || 0;
  const followLog = !logView || logView.scrollHeight - logScroll - logView.clientHeight < 80;
  const expandedSamples = $(".training-samples")?.open;
  const navigation = Object.keys(titles)
    .filter(visiblePane)
    .map(
      (key) =>
        `<a href="#${key}" class="${pane === key ? "active" : ""}"><svg viewBox="0 0 24 24"><path d="${paths[key] || paths.usage}"/></svg>${titles[key]}</a>`,
    )
    .join("");
  if ($("#navigation").dataset.rendered !== navigation) {
    $("#navigation").innerHTML = navigation;
    $("#navigation").dataset.rendered = navigation;
  }
  $("#breadcrumb").textContent = titles[pane];
  $("#content").innerHTML = {
    overview,
    jobs,
    models,
    deployments,
    keys,
    usage,
    datasets,
    playground,
    billing: renderBilling,
    team: renderTeam,
    account: renderAccount,
    customers: renderCustomers,
  }[pane]();
  if (pane === "jobs" && $("#training-log")) {
    $("#training-log").scrollTop = followLog ? $("#training-log").scrollHeight : logScroll;
    $(".training-samples").open = !!expandedSamples;
  }
  if (pane === "playground") {
    conversation = [];
    $("#chat-form").addEventListener("submit", sendChat);
  }
}
async function refresh(redraw = true) {
  try {
    state = await (await api("/api/state")).json();
    await refreshAccount();
    await refreshTraining();
    $("#login").hidden = true;
    $("#shell").hidden = false;
    const healthy =
      state.worker_heartbeat && Date.now() / 1000 - state.worker_heartbeat < 15;
    $("#connection").textContent = healthy ? "执行器在线" : "执行器离线";
    $("#connection").className = "live" + (healthy ? "" : " off");
    if (redraw && pane !== "playground" && !$("#modal").open) render();
    if (logJob && $("#modal").open) await loadLogs(logJob);
  } catch (e) {
    if (!$("#shell").hidden) toast(e.message);
  }
}
function openDialog(title, html, wide = false) {
  logJob = null;
  $("#modal-title").textContent = title;
  $("#modal-body").innerHTML = html;
  $("#modal").className = wide ? "wide" : "";
  if (!$("#modal").open) $("#modal").showModal();
}
function closeDialog() {
  logJob = null;
  $("#modal").close();
}
function form(title, html, submitLabel, callback) {
  openDialog(
    title,
    `<form id="dialog-form">${html}<p id="form-error" class="error" role="alert"></p><div class="form-footer"><button type="button" data-action="close">取消</button><button class="primary" type="submit">${submitLabel}</button></div></form>`,
  );
  $("#dialog-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const submit = e.target.querySelector("[type=submit]");
    submit.disabled = true;
    try {
      await callback(new FormData(e.target));
    } catch (err) {
      if ($("#form-error")) $("#form-error").textContent = err.message;
    } finally {
      submit.disabled = false;
    }
  });
}
async function loadLogs(id) {
  const text = await (await api(`/api/jobs/${id}/logs`)).text();
  const view = $("#log-output");
  if (view) {
    const end = view.scrollHeight - view.scrollTop - view.clientHeight < 80;
    view.textContent = text;
    if (end) view.scrollTop = view.scrollHeight;
  }
}
function confirmAction(title, message, callback) {
  form(title, `<p>${escape(message)}</p>`, "确认", async () => {
    await callback();
    closeDialog();
    await refresh();
    toast("已提交操作");
  });
}
async function actions(action, id) {
  if (await accountAction(action, id)) return;
  if (action === "logout") {
    if (busyChat) return toast("请等待当前回答完成");
    await post("/api/logout");
    showLogin();
    return;
  }
  if (action === "close") return closeDialog();
  if (action === "playground") {
    location.hash = "playground";
    return;
  }
  if (action === "new-job")
    return form(
      "创建训练任务",
      `<div class="notice">使用 Qwen3-0.6B · FSDP GRPO。训练使用 3 张 GPU，在线推理保持运行；单任务最长训练 20 分钟。</div><label>任务名称<input name="name" required maxlength="80" value="Qwen3 · ${new Date().toLocaleDateString("zh-CN")}"></label><label>数据集<select name="dataset_id">${state.datasets.map((d) => `<option value="${d.id}" ${d.id === id ? "selected" : ""}>${escape(d.name)} · ${d.train_rows} / ${d.eval_rows} 条</option>`).join("")}</select></label><div class="form-row"><label>更新轮数<input name="updates" type="number" min="1" max="10" value="2" required><small>每轮 4 个 prompt × 4 个回答</small></label><label>学习率<input name="learning_rate" type="number" min="0.0000001" max="0.00001" step="0.0000001" value="0.000001" required></label></div>`,
      "开始训练",
      async (f) => {
        const created = await post(
          "/api/jobs",
          {
            name: f.get("name"),
            dataset_id: f.get("dataset_id"),
            updates: Number(f.get("updates")),
            learning_rate: Number(f.get("learning_rate")),
          },
          true,
        );
        selectedTrainingJob = created.id;
        trainingData = null;
        closeDialog();
        location.hash = "jobs";
        await refresh();
        toast("训练任务已加入队列");
      },
    );
  if (action === "logs") {
    const j = state.jobs.find((j) => j.id === id);
    openDialog(
      j?.name || "任务日志",
      `<p class="scroll-note">保留最近 64 KB 日志，每 5 秒刷新。</p><pre id="log-output">读取日志…</pre>`,
      true,
    );
    logJob = id;
    return loadLogs(id);
  }
  if (action === "cancel")
    return confirmAction(
      "取消任务",
      "任务取消后会停止执行并清理训练资源。已完成的产物和日志会保留。",
      () => post(`/api/jobs/${id}/cancel`),
    );
  if (action === "evaluate")
    return confirmAction(
      "运行独立评测",
      "将通过模型的聊天 API 顺序评测独立数据集，并记录发布门槛结果。",
      () => post(`/api/models/${id}/evaluate`, {}, true),
    );
  if (action === "deploy")
    return confirmAction(
      "部署模型版本",
      "候选版本通过健康检查后才切换 miles-default。旧版本保留用于回滚。",
      () => post(`/api/models/${id}/deploy`, {}, true),
    );
  if (action === "rollback")
    return confirmAction(
      "回滚模型版本",
      "重新检查目标版本的评测和服务健康状态，然后切换统一入口。",
      () => post(`/api/deployments/${id}/rollback`, {}, true),
    );
  if (action === "model-detail") {
    const m = state.models.find((m) => m.id === id);
    return openDialog(
      m.name,
      `<div class="detail-line"><span>模型版本</span><strong>${m.id}</strong></div><div class="detail-line"><span>数据集</span><strong>${escape(state.datasets.find((d) => d.id === m.dataset_id)?.name)}</strong></div><p class="section-note">权重 SHA-256</p><pre>${m.sha256}</pre><p class="section-note">验证记录</p><pre>${escape(JSON.stringify(m.metadata, null, 2))}</pre><p class="section-note">评测历史</p>${state.evaluations
        .filter((e) => e.model_id === id)
        .map(
          (e) =>
            `<div class="detail-line"><span>${date(e.created)}</span><strong>${e.correct}/${e.total} · ${e.passed ? "达到门槛" : "未达标"}</strong></div>`,
        )
        .join("")}`,
    );
  }
  if (action === "new-key")
    return form(
      "创建 API Key",
      `<label>名称<input name="name" placeholder="例如：开发环境" required maxlength="60"></label><label>每分钟请求上限<input name="rpm" type="number" min="1" max="120" value="30" required></label>`,
      "创建密钥",
      async (f) => {
        const result = await post("/api/keys", {
          name: f.get("name"),
          rpm: Number(f.get("rpm")),
        });
        openDialog(
          "请保存你的 API Key",
          `<p>密钥原文仅在此显示一次。关闭后无法再次查看。</p><code class="key-secret" id="new-secret">${escape(result.secret)}</code>${button("复制密钥", "copy-key", "", "primary")}<div class="form-footer">${button("已保存，关闭", "close")}</div>`,
        );
        await refresh(false);
      },
    );
  if (action === "copy-key") {
    await navigator.clipboard.writeText($("#new-secret").textContent);
    return toast("密钥已复制");
  }
  if (action === "revoke")
    return confirmAction(
      "撤销 API Key",
      "使用这个密钥的新请求将立即被拒绝。此操作无法恢复，可重新创建密钥。",
      () => api(`/api/keys/${id}`, { method: "DELETE" }),
    );
  if (action === "copy-code") {
    await navigator.clipboard.writeText(curlExample());
    return toast("接入示例已复制");
  }
  if (action === "rates")
    return form(
      "设置费用估算单价",
      `<div class="notice warn">以 USD 计价。留空表示未配置；设置仅适用于之后创建的用量记录，不追溯历史费用。</div>${[
        ["gpu_hour", "每 GPU 小时"],
        ["input_million", "每百万输入 Tokens"],
        ["output_million", "每百万输出 Tokens"],
      ]
        .map(
          ([key, label]) =>
            `<label>${label}（USD）<input name="${key}" type="number" min="0" max="${key === "gpu_hour" ? 1000 : 10000}" step="0.0001" value="${state.rates[key] ?? ""}" placeholder="未配置"></label>`,
        )
        .join("")}`,
      "保存单价",
      async (f) => {
        const values = Object.fromEntries(
          ["gpu_hour", "input_million", "output_million"].map((k) => [
            k,
            f.get(k) === "" ? null : Number(f.get(k)),
          ]),
        );
        await api("/api/rates", {
          method: "PUT",
          body: JSON.stringify(values),
        });
        closeDialog();
        await refresh();
        toast("估算单价已保存");
      },
    );
  if (action === "new-dataset")
    return form(
      "上传数学任务数据集",
      `<label>数据集名称<input name="name" required maxlength="80"></label><label>训练集 JSONL<input name="training" type="file" accept=".jsonl,.json" required><small>4–5,000 条，最大 5 MB</small></label><label>独立评测集 JSONL<input name="evaluation" type="file" accept=".jsonl,.json" required><small>1–64 条，最大 1 MB</small></label><p class="section-note">每行格式；prompt 中请要求模型用 \\boxed{} 输出最终答案：</p><pre>{"prompt":[{"role":"user","content":"What is 2+2? Put the answer in \\\\boxed{}."}],"label":"4"}</pre>`,
      "上传并校验",
      async (f) => {
        const a = f.get("training"),
          b = f.get("evaluation");
        if (a.size > 5000000 || b.size > 1000000)
          throw new Error("文件超过大小限制");
        await post("/api/datasets", {
          name: f.get("name"),
          train_jsonl: await a.text(),
          eval_jsonl: await b.text(),
        });
        closeDialog();
        await refresh();
        toast("数据集已保存并校验");
      },
    );
  if (action === "clear-chat") {
    if (busyChat) return toast("请等待当前回答完成");
    conversation = [];
    render();
  }
}
async function sendChat(e) {
  e.preventDefault();
  if (busyChat) return;
  const input = $("#chat-input"),
    text = input.value.trim();
  if (!text) return;
  const messages = $("#chat-messages");
  messages.querySelector(".chat-welcome")?.remove();
  messages.insertAdjacentHTML(
    "beforeend",
    `<div class="message"><span class="avatar">W</span><div class="message-content"><b>你</b>${escape(text)}</div></div><div class="message"><span class="avatar mint">M</span><div class="message-content"><b>Miles</b><span id="latest-answer">正在连接模型…</span></div></div>`,
  );
  input.value = "";
  busyChat = true;
  const submit = e.target.querySelector("button");
  submit.disabled = true;
  const system = $("#system-prompt").value.trim(),
    pending = [...conversation, { role: "user", content: text }];
  let answer = "",
    usage;
  try {
    const response = await api("/api/playground", {
      method: "POST",
      body: JSON.stringify({
        model: "miles-default",
        messages: system
          ? [{ role: "system", content: system }, ...pending]
          : pending,
        max_tokens: Number($("#max-tokens").value),
        temperature: Number($("#temperature").value),
        stream: true,
      }),
    });
    const reader = response.body.getReader(),
      decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data: ") || line === "data: [DONE]") continue;
        const chunk = JSON.parse(line.slice(6));
        if (chunk.error) throw new Error(chunk.error.message);
        answer += chunk.choices?.[0]?.delta?.content || "";
        if (chunk.usage) usage = chunk.usage;
        $("#latest-answer").textContent = answer || "正在生成…";
        messages.scrollTop = messages.scrollHeight;
      }
    }
    conversation = [...pending, { role: "assistant", content: answer }];
    if (usage)
      toast(
        `已完成 · 输入 ${usage.prompt_tokens} / 输出 ${usage.completion_tokens} Tokens`,
      );
  } catch (err) {
    $("#latest-answer").textContent = answer
      ? answer + "\n\n[请求中断] " + err.message
      : err.message;
  } finally {
    $("#latest-answer").removeAttribute("id");
    busyChat = false;
    submit.disabled = false;
    await refresh(false);
  }
}
document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-action]");
  if (b)
    actions(b.dataset.action, b.dataset.id).catch((err) => toast(err.message));
});
$("#close-modal").addEventListener("click", closeDialog);
$("#modal").addEventListener("close", () => {
  logJob = null;
  $("#modal-body").innerHTML = "";
  if (pane !== "playground") render();
});
$("#login-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const b = e.target.querySelector("button");
  b.disabled = true;
  try {
    await post("/api/login", {
      email: new FormData(e.target).get("email"),
      password: new FormData(e.target).get("password"),
    });
    e.target.reset();
    await refresh(false);
    render();
  } catch (err) {
    $("#login-error").textContent = err.message;
  } finally {
    b.disabled = false;
  }
});
$("#logout").addEventListener("click", async () => {
  await post("/api/logout");
  showLogin();
});
window.addEventListener("hashchange", async () => {
  const next = location.hash.slice(1);
  if (titles[next]) {
    if (busyChat) {
      toast("请等待当前回答完成");
      location.hash = pane;
      return;
    }
    pane = next;
    await refreshAccount();
    await refreshTraining();
    render();
  }
});
async function pollRuntime() {
  if ($("#shell").hidden) return;
  try {
    runtime = await (await api("/api/runtime")).json();
    if (["overview", "deployments"].includes(pane) && !$("#modal").open)
      render();
  } catch (_) {}
}
async function start() {
  if (
    location.hash.startsWith("#invite=") ||
    location.hash.startsWith("#reset=")
  ) {
    showLogin();
    return acceptInvitation(
      location.hash.split("=")[1],
      location.hash.startsWith("#reset="),
    );
  }
  pane = titles[location.hash.slice(1)] ? location.hash.slice(1) : "overview";
  await refresh(false);
  if (state) {
    render();
    await pollRuntime();
  }
  setInterval(() => {
    if (!$("#shell").hidden) refresh();
  }, 5000);
  setInterval(pollRuntime, 15000);
}
start();
