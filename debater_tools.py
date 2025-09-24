from dotenv import load_dotenv
from typing import Any, Dict
from langchain_core.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper

#-------------------------------------------------------------------------------

load_dotenv(override=True)
serper = GoogleSerperAPIWrapper()


def _format_serper_results(res: Dict[str, Any]) -> str:
    items = []
    organic = res.get("organic", []) or res.get("results", []) or []

    for r in organic:
        title = r.get("title") or r.get("titleHighlighted") or "Untitled"
        link = r.get("link") or r.get("url") or ""
        snippet = r.get("snippet") or r.get("snippetHighlighted") or ""
        source = r.get("source") or r.get("domain") or ""
        
        if not source and link:
            try:
                from urllib.parse import urlparse
                source = urlparse(link).netloc
            except Exception:
                source = ""

        title = (title[:120] + "…") if len(title) > 120 else title
        snippet = (snippet[:200] + "…") if len(snippet) > 200 else snippet
        items.append(f"- {title} [{source}]\n  {snippet}")

    if not items:
        return "No relevant results."

    return "Search summary:\n" + "\n".join(items)


@tool
def web_search(query: str) -> str:
    """Search the web using Google Serper and return a compact textual summary."""
    if not query or not isinstance(query, str):
        return "No query provided."
    try:
        res = serper.results(query)
        return _format_serper_results(res)
    except Exception as e:
        return f"Search error: {e}"


TOOLS = [web_search]
tools_by_name = {t.name: t for t in TOOLS}