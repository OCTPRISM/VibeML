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
F = "ui-sans-serif,-apple-system,'PingFang SC','Microsoft YaHei','Segoe UI',Roboto,sans-serif"
M = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,monospace"
W_, H_ = 1280, 1086

def esc(s): return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
def tw(s, fs):                     # 中日韩按全宽、其余按 0.58 估算
    return sum(fs if ord(ch) > 0x2E80 else fs*0.58 for ch in s)

def gen(t):
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
        A(f'<text x="{x+22}" y="{y+31}" fill="{accent}" font-size="18" font-weight="700">{esc(title)}</text>')
        if sub: A(f'<text x="{x+22}" y="{y+53}" fill="{c["muted"]}" font-size="12.5">{esc(sub)}</text>')
    def chip(x,y,w,label,col,fs=12,h=26,mono=True):
        A(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" fill="{c["chipbg"]}" stroke="{col}" stroke-width="1.15"/>')
        A(f'<text x="{x+w/2}" y="{y+h/2+4.3}" fill="{col}" font-size="{fs}" text-anchor="middle" font-family="{M if mono else F}">{esc(label)}</text>')
    def badge(x,y,label,col,fs=12):
        w = tw(label,fs)+22
        A(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="23" rx="11.5" fill="{col}" opacity="0.16"/>')
        A(f'<text x="{x+w/2:.0f}" y="{y+16}" fill="{col}" font-size="{fs}" text-anchor="middle" font-weight="600">{esc(label)}</text>')
        return w

    A(f'<text x="40" y="48" fill="{c["text"]}" font-size="26" font-weight="800">VibeML · Multi-Agent 技术框架</text>')
    A(f'<text x="40" y="74" fill="{c["muted"]}" font-size="14">Agent 只通过工具调用驱动既有确定性流水线，不重新实现任何训练逻辑</text>')
    A(f'<line x1="40" y1="92" x2="{W_-40}" y2="92" stroke="{c["stroke"]}"/>')

    L, LW = 40, 780
    R, RW = 856, 384

    A(f'<rect x="{L}" y="112" width="{LW}" height="40" rx="20" fill="{c["card2"]}" stroke="{c["stroke"]}" stroke-width="1.5"/>')
    A(f'<text x="{L+LW/2}" y="138" fill="{c["text"]}" font-size="15" text-anchor="middle" font-weight="600">用户消息（自然语言）</text>')
    A(f'<path d="M{L+LW/2} 152 L{L+LW/2} 176" stroke="{c["muted"]}" stroke-width="2" marker-end="url(#ar)"/>')

    # 1 · Orchestrator
    card(L,180,LW,118,c["indigo"],"AgentOrchestrator","core/agent/agent_orchestrator.py · 主循环")
    bx = L+22
    for lb in ["≤ 15 轮","≤ 300 秒","超限友好收尾"]: bx += badge(bx,66+180,lb,c["indigo"])+9
    sx = L+352
    for i,s in enumerate(["制定计划","选择工具","读取真实结果","判断是否收敛"]):
        w = tw(s,12)+22; chip(sx,244,w,s,c["muted"],fs=12,h=26,mono=False); sx += w
        if i < 3: A(f'<path d="M{sx+1} 257 l8 0" stroke="{c["faint"]}" stroke-width="1.5" marker-end="url(#ar)"/>'); sx += 12
    A(f'<path d="M{L+LW/2} 298 L{L+LW/2} 324" stroke="{c["muted"]}" stroke-width="2" marker-end="url(#ar)"/>')
    A(f'<text x="{L+LW/2+12}" y="317" fill="{c["faint"]}" font-size="12" font-family="{M}">complete_with_tools()</text>')

    # 2 · provider 抽象
    card(L,328,LW,212,c["violet"],"Provider 无关的工具调用抽象","Agent 循环不认识任何一家的 wire format（core/llm_client.py）")
    tx = L+22
    for n in ["AgentMessage","ToolDef","ToolCall","ToolResult","CompletionResult"]:
        w = tw(n,11.5)+20; chip(tx,392,w,n,c["violet"],fs=11.5,h=26); tx += w+8
    A(f'<text x="{L+22}" y="446" fill="{c["muted"]}" font-size="12.5">各 provider 自行双向翻译</text>')
    px = L+184
    for p,note in [("Anthropic","tool_use block"),("Ollama","OpenAI 形状 · 需补 id"),("OpenAI 兼容","arguments 为 JSON 串")]:
        A(f'<rect x="{px}" y="428" width="190" height="46" rx="9" fill="{c["card2"]}" stroke="{c["violet"]}" stroke-width="1.15"/>')
        A(f'<text x="{px+95}" y="446" fill="{c["text"]}" font-size="13" text-anchor="middle" font-weight="600">{p}</text>')
        A(f'<text x="{px+95}" y="463" fill="{c["faint"]}" font-size="10.5" text-anchor="middle" font-family="{M}">{esc(note)}</text>')
        px += 196
    A(f'<rect x="{L+22}" y="488" width="{LW-44}" height="36" rx="8" fill="{c["rose"]}" opacity="0.11"/>')
    A(f'<rect x="{L+22}" y="488" width="{LW-44}" height="36" rx="8" fill="none" stroke="{c["rose"]}" stroke-width="1.25" stroke-dasharray="5 4"/>')
    A(f'<text x="{L+40}" y="511" fill="{c["rose"]}" font-size="13" font-weight="700">配额网关</text>')
    A(f'<text x="{L+120}" y="511" fill="{c["muted"]}" font-size="12">每次工具调用都计量入账 —— 挂在 complete_with_tools() 上，而非只挂 complete()</text>')
    A(f'<path d="M{L+LW/2} 540 L{L+LW/2} 566" stroke="{c["amber"]}" stroke-width="2.2" marker-end="url(#arA)"/>')

    # 3 · 工具层：3 列 × 2 行
    card(L,570,LW,272,c["amber"],"11 个工具","core/agent/tools.py · 包装既有能力，不新增任何实现")
    groups = [("计划",["set_plan","update_step_status"]),
              ("数据",["search_datasets","preview_dataset","request_dataset_confirmation"]),
              ("建模",["design_architecture","select_backbone"]),
              ("训练",["submit_training","check_training_progress"]),
              ("协作",["spawn_subagent"]),
              ("收尾",["finish_run"])]
    colw, gap = 236, 14
    for i,(name,tools) in enumerate(groups):
        col, row = i % 3, i // 3
        gx = L+22 + col*(colw+gap)
        gy = 648 + row*126
        A(f'<text x="{gx}" y="{gy}" fill="{c["amber"]}" font-size="12.5" font-weight="700">{name}</text>')
        ty = gy+9
        for tl in tools:
            chip(gx,ty,colw,tl,c["muted"],fs=11.5,h=25); ty += 29
    A(f'<path d="M{L+LW/2} 842 L{L+LW/2} 868" stroke="{c["emerald"]}" stroke-width="2.4" marker-end="url(#arE)"/>')

    # 4 · 确定性流水线
    card(L,872,LW,190,c["emerald"],"既有确定性流水线（一字未改）","Agent 没有任何路径能绕开这里的安全与评估机制")
    fx = L+22
    for nm,sub in [("api/worker.py","submit_training_job"),("core/pipeline.py","训练 → 评估 → 门禁"),
                   ("core/*_trainer.py","7 种后端 · 子进程隔离"),("core/*_deployer.py","独立可运行部署包")]:
        A(f'<rect x="{fx}" y="906" width="178" height="48" rx="9" fill="{c["card2"]}" stroke="{c["emerald"]}" stroke-width="1.15"/>')
        A(f'<text x="{fx+89}" y="925" fill="{c["text"]}" font-size="11.5" text-anchor="middle" font-family="{M}">{esc(nm)}</text>')
        A(f'<text x="{fx+89}" y="942" fill="{c["faint"]}" font-size="10.5" text-anchor="middle">{esc(sub)}</text>')
        fx += 186
    A(f'<rect x="{L+22}" y="970" width="{LW-44}" height="68" rx="9" fill="{c["shield"]}" stroke="{c["emerald"]}" stroke-width="1.15"/>')
    A(f'<text x="{L+40}" y="994" fill="{c["emerald"]}" font-size="13.5" font-weight="700">Agent 够不到的地方</text>')
    A(f'<text x="{L+40}" y="1015" fill="{c["muted"]}" font-size="12">AST 白名单沙盒 · 子进程墙钟超时 · 噪声门禁判定</text>')
    A(f'<text x="{L+40}" y="1032" fill="{c["muted"]}" font-size="12">绝不让 LLM 自己认定「这样就算成功了」</text>')

    # 右 · 子 Agent
    card(R,180,RW,196,c["cyan"],"短生命周期子 Agent","core/agent/subagent.py")
    bx = R+22
    for lb in ["并发 ≤ 3","每个 ≤ 8 轮"]: bx += badge(bx,246,lb,c["cyan"])+9
    A(f'<text x="{R+22}" y="296" fill="{c["muted"]}" font-size="12.5">只拿到只读工具</text>')
    chip(R+22,306,166,"search_datasets",c["cyan"],fs=11.5,h=25)
    chip(R+196,306,166,"preview_dataset",c["cyan"],fs=11.5,h=25)
    A(f'<text x="{R+22}" y="352" fill="{c["rose"]}" font-size="12">✕ 不含 submit_training 等有副作用的工具</text>')
    A(f'<text x="{R+22}" y="369" fill="{c["faint"]}" font-size="11.5">结论作为一条 tool_result 回灌主 Agent</text>')
    A(f'<path d="M{L+LW+10} 240 L{R-10} 240" stroke="{c["cyan"]}" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#arC)"/>')

    # 右 · 事件流
    card(R,406,RW,338,c["sky"],"实时事件流 · 10 种","AgentEvent → WebSocket → 前端")
    evs = ["plan_created","plan_step_update","tool_result","subagent_spawned","subagent_done",
           "resource_usage","confirmation_required","training_snapshot","fallback_to_workflow","final"]
    ey = 474
    for i in range(0,10,2):
        chip(R+22,ey,168,evs[i],c["sky"],fs=10.5,h=25)
        chip(R+198,ey,168,evs[i+1],c["sky"],fs=10.5,h=25)
        ey += 29
    A(f'<rect x="{R+22}" y="{ey+8}" width="{RW-44}" height="78" rx="9" fill="{c["card2"]}" stroke="{c["sky"]}" stroke-width="1.15"/>')
    A(f'<text x="{R+36}" y="{ey+30}" fill="{c["sky"]}" font-size="12.5" font-weight="700">resource_usage 的可观测字段</text>')
    A(f'<text x="{R+36}" y="{ey+50}" fill="{c["muted"]}" font-size="10.5" font-family="{M}">turn · duration_ms · prompt_tokens</text>')
    A(f'<text x="{R+36}" y="{ey+68}" fill="{c["muted"]}" font-size="10.5" font-family="{M}">completion_tokens · total_tokens</text>')

    # 右 · 降级
    card(R,774,RW,288,c["rose"],"工具调用不可用时自动降级","本地小模型 function calling 稳定性参差不齐")
    A(f'<rect x="{R+22}" y="844" width="{RW-44}" height="42" rx="8" fill="{c["card2"]}" stroke="{c["rose"]}" stroke-width="1.2" stroke-dasharray="5 4"/>')
    A(f'<text x="{R+RW/2}" y="870" fill="{c["text"]}" font-size="12.5" text-anchor="middle">一轮零工具调用 → 退回确定性状态机</text>')
    for i,line in enumerate(["界面如实告知已切换，不假装在思考",
                             "降级是粘性的，不每条消息重新试探",
                             "功能不受影响，只是少了自主决策过程"]):
        A(f'<circle cx="{R+28}" cy="{912+i*26-4}" r="2.6" fill="{c["rose"]}"/>')
        A(f'<text x="{R+40}" y="{912+i*26}" fill="{c["muted"]}" font-size="12">{esc(line)}</text>')
    A('</svg>')
    return "\n".join(o)

for t in ("dark","light"):
    io.open(f"assets/arch-multiagent-{t}.svg","w",encoding="utf-8").write(gen(t))
    print(f"  ✅ assets/arch-multiagent-{t}.svg")
