# Debater-Langgraph

This project runs a structured debate between two LLM agents (**Debater1** and **Debater2**) and then an impartial **Judge** selects a winner. A Gradio UI lets you enter a topic and view the debate as left/right chat bubbles; tool-call placeholders are hidden, and the final JSON verdict is shown.

## How It Works
1. **Debate rounds:** Debater1 and Debater2 alternate turns for `max_rounds`.
2. **Optional facts:** If a debater needs evidence, it calls the `web_search` tool (Google Serper) and incorporates results.
3. **Judging:** The Judge reads the transcript and returns a strict JSON:
   {"winner": "debater1|debater2|draw", "verdict": "short reasoning"}
