"use strict";
let account,
  wallet,
  members = [],
  customers = [];
const roleRank = { viewer: 0, member: 1, admin: 2, owner: 3 };

function visiblePane(key) {
  if (key === "customers") return !!account?.user.operator;
  if (key === "billing") return account?.user.role === "owner";
  if (key === "team") return roleRank[account?.user.role] >= 2;
  return true;
}

function allowedAction(action) {
  if (!account) return false;
  if (action === "rates") return !!account.user.operator;
  if (
    ["new-job", "cancel", "evaluate", "new-dataset", "playground"].includes(
      action,
    )
  )
    return roleRank[account.user.role] >= 1;
  if (["new-key", "revoke", "deploy", "rollback"].includes(action))
    return roleRank[account.user.role] >= 2;
  return true;
}

async function refreshAccount() {
  account = await (await api("/api/account")).json();
  if (!visiblePane(pane)) pane = "overview";
  const workspace = account.workspaces.find(
    (w) => w.id === account.user.tenant,
  );
  $("#workspace-select").innerHTML = account.workspaces
    .map(
      (w) =>
        `<option value="${escape(w.id)}" ${w.id === workspace.id ? "selected" : ""}>${escape(w.name)}</option>`,
    )
    .join("");
  $("#workspace-name").textContent = workspace.name;
  $("#account-profile").innerHTML =
    `${escape(account.user.email)}<small>${escape(account.user.role)}</small>`;
  if (pane === "billing") wallet = await (await api("/api/billing")).json();
  if (pane === "team") members = await (await api("/api/team")).json();
  if (pane === "customers")
    customers = await (await api("/api/operator/tenants")).json();
}

function renderBilling() {
  if (!wallet) return empty("Loading billing", "");
  const enabled = wallet.blockers.length === 0;
  const status =
    wallet.mode === "live"
      ? "LIVE · USD"
      : "TEST MODE · 测试余额不能转为真实余额";
  const balance = `<div class="metrics">${metric("可用余额 / Available", money(wallet.available_microusd / 1e6), status)}${metric("预留费用 / Reserved", money(wallet.held_microusd / 1e6), "任务结束后按用量结算，剩余预留自动释放")}${metric("已入账余额 / Balance", money(wallet.balance_microusd / 1e6), wallet.exempt ? "内部工作区 · 免扣费" : "预付费 · 余额不足时暂停新请求")}</div>`;
  const packs = Object.entries(wallet.packs)
    .map(
      ([key, cents]) =>
        `<article class="card"><div class="card-body"><span class="eyebrow">${escape(key)}</span><h2>$${cents / 100} USD</h2><p>用于训练与 API 推理；税费在结账时单列。</p><button class="primary" data-action="checkout" data-id="${key}" ${enabled ? "" : "disabled"}>${wallet.mode === "live" ? "充值 / Add credits" : "测试结账 / Test checkout"} ↗</button></div></article>`,
    )
    .join("");
  const notices = !enabled
    ? `<div class="notice"><b>支付尚未启用 / Payments unavailable</b><p>${wallet.mode === "live" ? "正在完成上线配置。" : "当前没有连接 Stripe 测试账户。"} 请联系平台运营者。</p>${account.user.operator ? wallet.blockers.map((b) => `<p>${escape(b)}</p>`).join("") : ""}</div>`
    : `<div class="notice">Stripe 托管结账。支付后余额由服务端确认，返回本页不会直接增加余额。${wallet.mode === "test" ? "当前只接受 Stripe 测试卡，不会真实扣款。" : ""}</div>`;
  const orders = wallet.orders
    .map(
      (o) =>
        `<tr><td><code>${escape(o.id)}</code><small>${date(o.created)}</small></td><td>$${(o.cents / 100).toFixed(2)}</td><td>${o.disputed ? "争议审核中" : o.lost ? "争议扣回" : o.refunded ? "已退款 $" + (o.refunded / 100).toFixed(2) : o.paid ? "已到账 / Paid" : "待支付 / Pending"}</td><td>${escape(o.mode)}</td></tr>`,
    )
    .join("");
  const ledger = wallet.ledger
    .map(
      (l) =>
        `<tr><td>${date(l.created)}</td><td>${escape(l.kind)}</td><td>${money(l.amount / 1e6)}</td><td><code>${escape(l.reference)}</code></td></tr>`,
    )
    .join("");
  return (
    head("账单与余额 / Billing", "美元预付余额，训练和推理按已公布单价结算。") +
    balance +
    notices +
    `<div class="billing-packs">${packs}</div><div class="notice">训练预留上限：${money(wallet.training_reservation_microusd == null ? null : wallet.training_reservation_microusd / 1e6)} / 次。按 3 张 GPU、最多 22 分钟预留；实际训练与清理时间计费，未用部分释放。评测与模型托管包含在当前试运营配额内。单价见“用量与费用”。</div><section class="card"><div class="card-head"><h2>支付订单 / Orders</h2></div><div class="table-wrap"><table><thead><tr><th>订单</th><th>Credits</th><th>状态</th><th>Mode</th></tr></thead><tbody>${orders || '<tr><td colspan="4">还没有支付订单</td></tr>'}</tbody></table></div></section><section class="card"><div class="card-head"><h2>余额流水 / Ledger</h2></div><div class="table-wrap"><table><thead><tr><th>时间</th><th>类型</th><th>USD</th><th>关联记录</th></tr></thead><tbody>${ledger || '<tr><td colspan="4">还没有余额变动</td></tr>'}</tbody></table></div></section><p class="section-note">退款请联系 ${escape(wallet.support || "平台运营者")}。${wallet.terms ? `<a href="${escape(wallet.terms)}" target="_blank" rel="noopener">Terms</a>` : ""} ${wallet.privacy ? `<a href="${escape(wallet.privacy)}" target="_blank" rel="noopener">Privacy</a>` : ""}</p>`
  );
}

function renderTeam() {
  return (
    head(
      "团队 / Team",
      "Owner 管理账单与成员；Admin 管理部署和密钥；Member 运行任务；Viewer 只读。",
      button("邀请成员", "invite"),
    ) +
    `<section class="card"><div class="table-wrap"><table><thead><tr><th>账户</th><th>角色</th><th>操作</th></tr></thead><tbody>${members.map((m) => `<tr><td>${escape(m.email)}</td><td>${escape(m.role)}</td><td>${m.role !== "owner" && account.user.role === "owner" ? button("修改角色", "member-role", m.id, "quiet") + button("移除", "member-remove", m.id, "quiet danger") : "—"}</td></tr>`).join("")}</tbody></table></div></section><div class="notice">邀请链接仅展示一次，24 小时内有效。请通过可信渠道交给指定邮箱的使用者；已有账户需输入原密码接受邀请。</div>`
  );
}

function renderAccount() {
  return (
    head(
      "账户 / Account",
      "管理你的登录凭据。",
      button("修改密码", "password"),
    ) +
    `<section class="card"><div class="card-body"><div class="detail-line"><span>账户</span><strong>${escape(account.user.email)}</strong></div><div class="detail-line"><span>当前角色</span><strong>${escape(account.user.role)}</strong></div><p>修改密码会撤销其他登录会话。退出或移除成员后，会话即时失效。</p></div></section>`
  );
}

function renderCustomers() {
  return (
    head(
      "客户工作区 / Customers",
      "每个客户独立保存数据、模型、密钥和账本。客户工作区默认为预付费。",
      button("创建客户工作区", "new-tenant"),
    ) +
    `<div class="notice">上线前需完成：客户 HTTPS 域名、Stripe 商户验证与测试支付、持久存储及异地恢复、服务条款和隐私政策。当前部署保持私有验收访问。</div><section class="card"><div class="table-wrap"><table><thead><tr><th>工作区</th><th>状态</th><th>计费</th><th>操作</th></tr></thead><tbody>${customers.map((t) => `<tr><td>${escape(t.name)}<small>${escape(t.id)}</small></td><td>${escape(t.status)}</td><td>${t.exempt ? "内部免扣费" : "预付费"}</td><td>${t.id === "default" ? "—" : button(t.status === "active" ? "暂停" : "恢复", "tenant-status", t.id, "quiet")}</td></tr>`).join("")}</tbody></table></div></section>`
  );
}

function invitationResult(result) {
  const url = location.origin + "/#invite=" + result.token;
  openDialog(
    "邀请链接 / Invitation",
    `<p>此链接只展示一次，24 小时有效。请交给指定的账户使用者。</p><textarea readonly rows="4" aria-label="Invitation link">${escape(url)}</textarea>`,
  );
}

async function accountAction(action, id) {
  if (action === "checkout") {
    const result = await post("/api/billing/checkout", { pack: id }, true);
    location.assign(result.url);
  } else if (action === "new-tenant" || action === "invite") {
    const create = action === "new-tenant";
    form(
      create ? "创建客户工作区" : "邀请团队成员",
      `${create ? '<label>工作区名称<input name="name" required maxlength="80"></label>' : ""}<label>客户邮箱<input name="email" type="email" required maxlength="254"></label>${!create ? `<label>角色<select name="role"><option value="member">Member · 运行任务</option><option value="viewer">Viewer · 只读</option>${account.user.role === "owner" ? '<option value="admin">Admin · 管理部署与密钥</option>' : ""}</select></label>` : ""}`,
      "生成邀请",
      async (f) => {
        const body = create
          ? { name: f.get("name"), email: f.get("email") }
          : { email: f.get("email"), role: f.get("role") };
        const result = await post(
          create ? "/api/operator/tenants" : "/api/team/invite",
          body,
        );
        await refreshAccount();
        invitationResult(result);
      },
    );
  } else if (action === "password") {
    form(
      "修改密码 / Change password",
      '<label>原密码<input name="current" type="password" autocomplete="current-password" required></label><label>新密码（至少 12 位）<input name="password" type="password" autocomplete="new-password" minlength="12" maxlength="200" required></label>',
      "保存",
      async (f) => {
        await post("/api/account/password", Object.fromEntries(f));
        closeDialog();
        toast("密码已更新，其他会话已撤销");
      },
    );
  } else if (action === "member-role") {
    form(
      "修改成员角色",
      '<label>角色<select name="role"><option value="viewer">Viewer</option><option value="member">Member</option><option value="admin">Admin</option></select></label>',
      "保存",
      async (f) => {
        await api("/api/team/" + id, {
          method: "PATCH",
          body: JSON.stringify({ role: f.get("role") }),
        });
        closeDialog();
        await refresh();
      },
    );
  } else if (action === "member-remove") {
    confirmAction(
      "移除成员",
      "移除后将立即失去此工作区的访问权限。工作区 API 密钥由团队共享，请按需另行撤销。",
      () => api("/api/team/" + id, { method: "DELETE" }),
    );
  } else if (action === "tenant-status") {
    const target = customers.find((t) => t.id === id);
    const status = target.status === "active" ? "suspended" : "active";
    confirmAction(
      "更新工作区状态",
      "暂停会拒绝客户新请求，并取消排队或运行中的任务。已有数据和账本保留。",
      () =>
        api("/api/operator/tenants/" + id, {
          method: "PATCH",
          body: JSON.stringify({ status }),
        }),
    );
  } else return false;
  return true;
}

function acceptInvitation(token, reset = false) {
  const login = $("#login-form");
  login.innerHTML =
    '<span class="eyebrow">WELCOME TO MILES</span><h2>接受邀请 / Accept invitation</h2><p>新账户请设置至少 12 位密码。已有账户请输入原密码。</p><label>Password<input name="password" type="password" minlength="12" maxlength="200" required autocomplete="new-password"></label><button type="submit" class="primary">加入工作区 / Join workspace</button><p id="login-error" class="error" role="alert"></p>';
  if (reset) {
    login.querySelector("h2").textContent = "重设密码 / Reset password";
    login.querySelector("p").textContent =
      "设置至少 12 位新密码。其他会话将被撤销。";
    login.querySelector("button").textContent = "保存新密码 / Save password";
  }
  login.addEventListener(
    "submit",
    async (e) => {
      e.preventDefault();
      e.stopImmediatePropagation();
      const button = login.querySelector("button");
      button.disabled = true;
      try {
        await post(reset ? "/api/account/reset" : "/api/invitations/accept", {
          token,
          password: new FormData(login).get("password"),
        });
        location.replace(location.origin + "/#overview");
        location.reload();
      } catch (err) {
        $("#login-error").textContent = err.message;
      } finally {
        button.disabled = false;
      }
    },
    true,
  );
}

document.addEventListener("DOMContentLoaded", () => {
  $("#workspace-select").addEventListener("change", async (e) => {
    if (busyChat) {
      toast("请等待当前回答完成");
      return refreshAccount();
    }
    try {
      await post("/api/account/switch", { tenant_id: e.target.value });
      location.reload();
    } catch (err) {
      toast(err.message);
      await refreshAccount();
    }
  });
});
