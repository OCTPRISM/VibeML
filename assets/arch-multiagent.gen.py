# -*- coding: utf-8 -*-
"""Multi-Agent 架构图生成器（亮/暗两套）。内容全部对应已核实的源码事实。"""
import io

THEMES = {
 "dark": dict(bg="#0d1117", card="#161b22", card2="#1c2129", stroke="#30363d",
              text="#e6edf3", muted="#9aa5b1", faint="#768390", chipbg="#21262d",
              indigo="#818cf8", violet="#c4b5fd", rose="#fb7185", amber="#e8a33d",
              emerald="#34d399", cyan="#22d3ee", sky="#60a5fa", shield="#17251f"),
 "light": dict(bg="#ffffff", card="#f6f8fa", card2="#ffffff", stroke="#d0d7de",
               text="#1f2328", muted="#59636e", faint="#7d8590", chipbg="#ffffff",
               indigo="#4f46e5", violet="#7c3aed", rose="#be123c", amber="#a16207",
               emerald="#047857", cyan="#0e7490", sky="#1d4ed8", shield="#edfaf4"),
}

STR = {
 "zh": dict(
  title="VibeML · Multi-Agent 技术框架",
  sub="Agent 只通过工具调用驱动既有确定性流水线，不重新实现任何训练逻辑",
  user="用户消息（自然语言）",
  orch_sub="core/agent/agent_orchestrator.py · 主循环",
  badges=["≤ 15 轮","≤ 300 秒","超限友好收尾"],
  steps=["制定计划","选择工具","读取真实结果","判断是否收敛"],
  abs_t="Provider 无关的工具调用抽象",
  abs_s="Agent 循环不认识任何一家的 wire format（core/llm_client.py）",
  xlate="各 provider 自行双向翻译",
  provs=[("Anthropic","tool_use block"),("Ollama","OpenAI 形状 · 需补 id"),
         ("OpenAI 兼容","arguments 为 JSON 串")],
  quota="配额网关", quota_s="每次工具调用都计量入账 —— 挂在 complete_with_tools() 上，而非只挂 complete()",
  tools_t="11 个工具", tools_s="core/agent/tools.py · 包装既有能力，不新增任何实现",
  groups=["计划","数据","建模","训练","协作","收尾"],
  det_t="确定性执行层", det_s="训练、评估、部署的全部实现 —— Agent 只调用，不改写，也绕不开",
  pipes=[("api/worker.py","submit_training_job"),("core/pipeline.py","训练 → 评估 → 门禁"),
         ("core/*_trainer.py","7 种后端 · 子进程隔离"),("core/*_deployer.py","独立可运行部署包")],
  guard_t="Agent 够不到的地方",
  guard=["AST 白名单沙盒 · 子进程墙钟超时 · 噪声门禁判定","绝不让 LLM 自己认定「这样就算成功了」"],
  sub_t="短生命周期子 Agent", sub_s="core/agent/subagent.py",
  sub_badges=["并发 ≤ 3","每个 ≤ 8 轮"], sub_ro="只拿到只读工具",
  sub_no="✕ 不含 submit_training 等有副作用的工具",
  sub_back="结论作为一条 tool_result 回灌主 Agent",
  ev_t="实时事件流 · 10 种", ev_s="AgentEvent → WebSocket → 前端",
  ev_box="resource_usage 的可观测字段",
  fb_t="工具调用不可用时自动降级", fb_s="本地小模型 function calling 稳定性参差不齐",
  fb_box="一轮零工具调用 → 退回确定性状态机",
  fb=["界面如实告知已切换，不假装在思考","降级是粘性的，不每条消息重新试探",
      "功能不受影响，只是少了自主决策过程"]),
 "en": dict(
  title="VibeML · Multi-Agent Architecture",
  sub="The agent drives the existing deterministic pipeline through tool calls — it reimplements no training logic",
  user="User message (natural language)",
  orch_sub="core/agent/agent_orchestrator.py · main loop",
  badges=["≤ 15 turns","≤ 300 s","graceful stop at limit"],
  steps=["Plan","Pick tool","Read result","Converged?"],
  abs_t="Provider-agnostic tool-calling abstraction",
  abs_s="The agent loop knows no vendor's wire format (core/llm_client.py)",
  xlate="Each provider translates both ways",
  provs=[("Anthropic","tool_use block"),("Ollama","OpenAI shape · id synthesized"),
         ("OpenAI-compatible","arguments is a JSON string")],
  quota="Quota gate", quota_s="Every tool call is metered — hooked on complete_with_tools(), not just complete()",
  tools_t="11 tools", tools_s="core/agent/tools.py · wraps existing capabilities, adds no new implementation",
  groups=["Plan","Data","Modeling","Training","Collab","Finish"],
  det_t="Deterministic execution layer",
  det_s="All training, evaluation and deployment logic — the agent only calls it, never rewrites or bypasses it",
  pipes=[("api/worker.py","submit_training_job"),("core/pipeline.py","train → eval → gate"),
         ("core/*_trainer.py","7 backends · subprocess-isolated"),("core/*_deployer.py","self-contained deploy bundle")],
  guard_t="Out of the agent's reach",
  guard=["AST allowlist sandbox · subprocess wall-clock timeout · noise gate",
         "The LLM never gets to declare success on its own"],
  sub_t="Short-lived sub-agents", sub_s="core/agent/subagent.py",
  sub_badges=["≤ 3 concurrent","≤ 8 turns each"], sub_ro="Read-only tools only",
  sub_no="✕ No side-effecting tools such as submit_training",
  sub_back="Findings return to the main agent as one tool_result",
  ev_t="Live event stream · 10 kinds", ev_s="AgentEvent → WebSocket → frontend",
  ev_box="Observable fields of resource_usage",
  fb_t="Automatic fallback when tool calling is unavailable",
  fb_s="Function-calling reliability varies across small local models",
  fb_box="Zero tool calls in a turn → deterministic state machine",
  fb=["The UI says so plainly; it does not fake thinking","Fallback is sticky, not re-probed every message",
      "Capability is unaffected — only autonomy is"]),
}

F = "ui-sans-serif,-apple-system,'PingFang SC','Microsoft YaHei','Segoe UI',Roboto,sans-serif"
M = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,monospace"
W_, H_ = 1280, 1184

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def tw(s, fs):                     # 中日韩按全宽、其余按 0.58 估算
    return sum(fs if ord(ch) > 0x2E80 else fs*0.58 for ch in s)

def gen(t, lang='zh'):
    S = STR[lang]
    c = THEMES[t]; o = []; A = o.append
    A(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_} {H_}" width="{W_}" height="{H_}" font-family="{F}">')
    A(f'<rect width="{W_}" height="{H_}" fill="{c["bg"]}" rx="14"/>')
    A(f'''<defs>
      <marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="{c["muted"]}"/></marker>
      <marker id="arA" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="{c["amber"]}"/></marker>
      <marker id="arE" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="{c["emerald"]}"/></marker>
      <marker id="arC" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M0 0 L10 5 L0 10 z" fill="{c["cyan"]}"/></marker>
    </defs>''')

    def card(x,y,w,h,accent,title,sub=None):
        A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{c["card"]}" stroke="{c["stroke"]}" stroke-width="1.5"/>')
        A(f'<rect x="{x}" y="{y}" width="5" height="{h}" rx="2.5" fill="{accent}"/>')
        tfs = 18 if tw(title,18) <= w-44 else max(13, 18*(w-44)/tw(title,18))
        A(f'<text x="{x+22}" y="{y+31}" fill="{accent}" font-size="{tfs:.1f}" font-weight="700">{esc(title)}</text>')
        if sub: A(f'<text x="{x+22}" y="{y+53}" fill="{c["muted"]}" font-size="12.5">{esc(sub)}</text>')
    def chip(x,y,w,label,col,fs=12,h=26,mono=True):
        A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{c["chipbg"]}" stroke="{col}" stroke-width="1.15"/>')
        A(f'<text x="{x+w/2}" y="{y+h/2+4.3}" fill="{col}" font-size="{fs}" text-anchor="middle" font-family="{M if mono else F}">{esc(label)}</text>')
    def badge(x,y,label,col,fs=12):
        w = tw(label,fs)+22
        A(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="23" rx="11.5" fill="{col}" opacity="0.16"/>')
        A(f'<text x="{x+w/2:.0f}" y="{y+16}" fill="{col}" font-size="{fs}" text-anchor="middle" font-weight="600">{esc(label)}</text>')
        return w

    A(f'<text x="40" y="48" fill="{c["text"]}" font-size="26" font-weight="800">{esc(S["title"])}</text>')
    A(f'<text x="40" y="74" fill="{c["muted"]}" font-size="14">{esc(S["sub"])}</text>')
    A(f'<line x1="40" y1="92" x2="{W_-40}" y2="92" stroke="{c["stroke"]}"/>')

    L, LW = 40, 780
    R, RW = 856, 384

    A(f'<rect x="{L}" y="112" width="{LW}" height="40" rx="20" fill="{c["card2"]}" stroke="{c["stroke"]}" stroke-width="1.5"/>')
    A(f'<text x="{L+LW/2}" y="138" fill="{c["text"]}" font-size="15" text-anchor="middle" font-weight="600">{esc(S["user"])}</text>')
    A(f'<path d="M{L+LW/2} 152 L{L+LW/2} 176" stroke="{c["muted"]}" stroke-width="2" marker-end="url(#ar)"/>')

    # 1 · Orchestrator
    card(L,180,LW,152,c["indigo"],"AgentOrchestrator",S["orch_sub"])
    bx = L+22
    for lb in S["badges"]: bx += badge(bx,66+180,lb,c["indigo"])+9
    sx = L+22                      # 步骤独占一行，避免英文 badge 变长后相撞
    for i,s in enumerate(S["steps"]):
        w = tw(s,12)+22; chip(sx,282,w,s,c["muted"],fs=12,h=26,mono=False); sx += w
        if i < len(S["steps"])-1:
            A(f'<path d="M{sx+1} 295 l8 0" stroke="{c["faint"]}" stroke-width="1.5" marker-end="url(#ar)"/>'); sx += 12
    A('<g transform="translate(0,34)">')      # 以下左列整体下移，给步骤行腾空间
    A(f'<path d="M{L+LW/2} 298 L{L+LW/2} 324" stroke="{c["muted"]}" stroke-width="2" marker-end="url(#ar)"/>')
    A(f'<text x="{L+LW/2+12}" y="317" fill="{c["faint"]}" font-size="12" font-family="{M}">complete_with_tools()</text>')

    # 2 · provider 抽象
    card(L,328,LW,242,c["violet"],S["abs_t"],S["abs_s"])
    tx = L+22
    for n in ["AgentMessage","ToolDef","ToolCall","ToolResult","CompletionResult"]:
        w = tw(n,11.5)+20; chip(tx,392,w,n,c["violet"],fs=11.5,h=26); tx += w+8
    A(f'<text x="{L+22}" y="440" fill="{c["muted"]}" font-size="12.5">{esc(S["xlate"])}</text>')
    px = L+22
    for p,note in S["provs"]:
        A(f'<rect x="{px}" y="452" width="246" height="46" rx="9" fill="{c["card2"]}" stroke="{c["violet"]}" stroke-width="1.15"/>')
        A(f'<text x="{px+123}" y="470" fill="{c["text"]}" font-size="13" text-anchor="middle" font-weight="600">{p}</text>')
        A(f'<text x="{px+123}" y="487" fill="{c["faint"]}" font-size="10.5" text-anchor="middle" font-family="{M}">{esc(note)}</text>')
        px += 252
    A(f'<rect x="{L+22}" y="518" width="{LW-44}" height="36" rx="8" fill="{c["rose"]}" opacity="0.11"/>')
    A(f'<rect x="{L+22}" y="518" width="{LW-44}" height="36" rx="8" fill="none" stroke="{c["rose"]}" stroke-width="1.25" stroke-dasharray="5 4"/>')
    A(f'<text x="{L+40}" y="541" fill="{c["rose"]}" font-size="13" font-weight="700">{esc(S["quota"])}</text>')
    A(f'<text x="{L+150}" y="541" fill="{c["muted"]}" font-size="12">{esc(S["quota_s"])}</text>')
    A(f'<path d="M{L+LW/2} 570 L{L+LW/2} 596" stroke="{c["amber"]}" stroke-width="2.2" marker-end="url(#arA)"/>')

    # 3 · 工具层：3 列 × 2 行
    card(L,600,LW,272,c["amber"],S["tools_t"],S["tools_s"])
    _gt = [["set_plan","update_step_status"],
           ["search_datasets","preview_dataset","request_dataset_confirmation"],
           ["design_architecture","select_backbone"],
           ["submit_training","check_training_progress"],
           ["spawn_subagent"],["finish_run"]]
    groups = list(zip(S["groups"], _gt))
    colw, gap = 236, 14
    for i,(name,tools) in enumerate(groups):
        col, row = i % 3, i // 3
        gx = L+22 + col*(colw+gap)
        gy = 678 + row*126
        A(f'<text x="{gx}" y="{gy}" fill="{c["amber"]}" font-size="12.5" font-weight="700">{name}</text>')
        ty = gy+9
        for tl in tools:
            chip(gx,ty,colw,tl,c["muted"],fs=11.5,h=25); ty += 29
    A(f'<path d="M{L+LW/2} 872 L{L+LW/2} 898" stroke="{c["emerald"]}" stroke-width="2.4" marker-end="url(#arE)"/>')

    # 4 · 确定性流水线
    card(L,902,LW,214,c["emerald"],S["det_t"],S["det_s"])
    fx = L+22
    for nm,sub in S["pipes"]:
        A(f'<rect x="{fx}" y="960" width="178" height="48" rx="9" fill="{c["card2"]}" stroke="{c["emerald"]}" stroke-width="1.15"/>')
        A(f'<text x="{fx+89}" y="979" fill="{c["text"]}" font-size="11.5" text-anchor="middle" font-family="{M}">{esc(nm)}</text>')
        A(f'<text x="{fx+89}" y="996" fill="{c["faint"]}" font-size="10.5" text-anchor="middle">{esc(sub)}</text>')
        fx += 186
    A(f'<rect x="{L+22}" y="1024" width="{LW-44}" height="68" rx="9" fill="{c["shield"]}" stroke="{c["emerald"]}" stroke-width="1.15"/>')
    A(f'<text x="{L+40}" y="1048" fill="{c["emerald"]}" font-size="13.5" font-weight="700">{esc(S["guard_t"])}</text>')
    A(f'<text x="{L+40}" y="1069" fill="{c["muted"]}" font-size="12">{esc(S["guard"][0])}</text>')
    A(f'<text x="{L+40}" y="1086" fill="{c["muted"]}" font-size="12">{esc(S["guard"][1])}</text>')

    A('</g>')

    # 右 · 子 Agent
    card(R,180,RW,196,c["cyan"],S["sub_t"],S["sub_s"])
    bx = R+22
    for lb in S["sub_badges"]: bx += badge(bx,246,lb,c["cyan"])+9
    A(f'<text x="{R+22}" y="296" fill="{c["muted"]}" font-size="12.5">{esc(S["sub_ro"])}</text>')
    chip(R+22,306,166,"search_datasets",c["cyan"],fs=11.5,h=25)
    chip(R+196,306,166,"preview_dataset",c["cyan"],fs=11.5,h=25)
    A(f'<text x="{R+22}" y="352" fill="{c["rose"]}" font-size="12">{esc(S["sub_no"])}</text>')
    A(f'<text x="{R+22}" y="369" fill="{c["faint"]}" font-size="11.5">{esc(S["sub_back"])}</text>')
    A(f'<path d="M{L+LW+10} 240 L{R-10} 240" stroke="{c["cyan"]}" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#arC)"/>')

    # 右 · 事件流
    card(R,406,RW,338,c["sky"],S["ev_t"],S["ev_s"])
    evs = ["plan_created","plan_step_update","tool_result","subagent_spawned","subagent_done",
           "resource_usage","confirmation_required","training_snapshot","fallback_to_workflow","final"]
    ey = 474
    for i in range(0,10,2):
        chip(R+22,ey,168,evs[i],c["sky"],fs=10.5,h=25)
        chip(R+198,ey,168,evs[i+1],c["sky"],fs=10.5,h=25)
        ey += 29
    A(f'<rect x="{R+22}" y="{ey+8}" width="{RW-44}" height="78" rx="9" fill="{c["card2"]}" stroke="{c["sky"]}" stroke-width="1.15"/>')
    A(f'<text x="{R+36}" y="{ey+30}" fill="{c["sky"]}" font-size="12.5" font-weight="700">{esc(S["ev_box"])}</text>')
    A(f'<text x="{R+36}" y="{ey+50}" fill="{c["muted"]}" font-size="10.5" font-family="{M}">turn · duration_ms · prompt_tokens</text>')
    A(f'<text x="{R+36}" y="{ey+68}" fill="{c["muted"]}" font-size="10.5" font-family="{M}">completion_tokens · total_tokens</text>')

    # 右 · 降级
    card(R,774,RW,386,c["rose"],S["fb_t"],S["fb_s"])
    A(f'<rect x="{R+22}" y="844" width="{RW-44}" height="42" rx="8" fill="{c["card2"]}" stroke="{c["rose"]}" stroke-width="1.2" stroke-dasharray="5 4"/>')
    A(f'<text x="{R+RW/2}" y="870" fill="{c["text"]}" font-size="12.5" text-anchor="middle">{esc(S["fb_box"])}</text>')
    for i,line in enumerate(S["fb"]):
        A(f'<circle cx="{R+28}" cy="{912+i*26-4}" r="2.6" fill="{c["rose"]}"/>')
        A(f'<text x="{R+40}" y="{912+i*26}" fill="{c["muted"]}" font-size="12">{esc(line)}</text>')
    A('</svg>')
    return "\n".join(o)

for lg in ("zh","en"):
    for t in ("dark","light"):
        io.open(f"assets/arch-multiagent-{lg}-{t}.svg","w",encoding="utf-8").write(gen(t,lg))
        print(f"  ✅ assets/arch-multiagent-{lg}-{t}.svg")
