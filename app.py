import html
from typing import Dict, List
import gradio as gr

from debater import run_debate

#-------------------------------------------------------------------------------

TOOL_REQUEST_TOKENS = {"[REQUEST_TOOLS]", "REQUEST_TOOLS", "<REQUEST_TOOLS>", "[CALL_TOOL]", "CALL_TOOL"}


def is_tool_placeholder(text: str) -> bool:
    t = (text or "").strip()
    return t in TOOL_REQUEST_TOKENS or t.upper() in TOOL_REQUEST_TOKENS

def flatten_chat_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    rounds = {"debater1": 0, "debater2": 0}
    out: List[Dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        text = (m.get("content") or "").strip()
        if role not in ("debater1", "debater2"):
            continue
        if is_tool_placeholder(text):
            continue
        rounds[role] += 1
        out.append({
            "side": "left" if role == "debater1" else "right",
            "who": "Debater 1" if role == "debater1" else "Debater 2",
            "round": str(rounds[role]),
            "text": text
        })
    return out

def render_chat_html(entries: List[Dict[str, str]]) -> str:
    items = []
    for e in entries:
        txt = html.escape(e["text"]).replace("\n", "<br>")
        who = e["who"]
        rnd = e["round"]
        side_class = "left" if e["side"] == "left" else "right"
        bubble_class = "bubble-left" if side_class == "left" else "bubble-right"
        items.append(f"""
        <div class="msg-row {side_class}">
          <div class="msg-bubble {bubble_class}">
            <div class="msg-meta">{who} — Round {rnd}</div>
            <div class="msg-text">{txt or "<i>(no message)</i>"} </div>
          </div>
        </div>
        """)
    style = """
    <style>
      .chat-wrap { display:flex; flex-direction:column; gap:10px; }
      .msg-row { display:flex; width:100%; }
      .msg-row.left { justify-content:flex-start; }
      .msg-row.right { justify-content:flex-end; }

      .msg-bubble {
        max-width: 78%;
        padding: 10px 12px;
        border: 1px solid #00000022;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        color: #000 !important;
      }
      .msg-text { color:#000 !important; line-height:1.6; }
      .msg-meta { font-weight:700; font-size: 0.9rem; margin-bottom:6px; color:#000 !important; }

      .bubble-left {
        background: #fff4c2;
        border-radius: 14px 14px 14px 4px;
      }
      .bubble-right {
        background: #c2f2e9;
        border-radius: 14px 14px 4px 14px;
      }
      @media (max-width: 900px) { .msg-bubble { max-width: 92%; } }
    </style>
    """
    return style + f'<div class="chat-wrap">{"".join(items) or "<i>No debater messages found.</i>"}</div>'

def render_judge_md(winner: str, verdict: str) -> str:
    w = (winner or "draw").strip() or "draw"
    v = (verdict or "").strip() or "No verdict provided."
    return f"### Judge's Decision\n\n**Winner:** `{w}`\n\n**Verdict:**\n\n{v}"

def debate_handler(question: str, max_rounds: int):
    question = (question or "").strip()
    if not question:
        return "<i>Please enter a question to start the debate.</i>", "### Judge's Decision\n\nAwaiting input."
    result = run_debate(topic=question, max_rounds=int(max_rounds), print_result=False)
    entries = flatten_chat_messages(result.get("messages", []))
    chat_html = render_chat_html(entries)
    judge_md = render_judge_md(result.get("winner"), result.get("verdict"))
    return chat_html, judge_md


with gr.Blocks(title="LangGraph Debate UI") as demo:
    gr.Markdown("# LangGraph Debate\nEnter a question, then view debaters' chat-style responses.")
    with gr.Row():
        question_in = gr.Textbox(label="Debate Question", placeholder="e.g., Should tabs be preferred over spaces?", lines=3)
    with gr.Row():
        rounds_in = gr.Slider(minimum=1, maximum=6, step=1, value=2, label="Rounds per Debater")
        run_btn = gr.Button("Run Debate", variant="primary")
    with gr.Row():
        debaters_out = gr.HTML(label="Debate Chat")
        judge_out = gr.Markdown(label="Judge")
    run_btn.click(fn=debate_handler, inputs=[question_in, rounds_in], outputs=[debaters_out, judge_out])

if __name__ == "__main__":
    demo.launch()