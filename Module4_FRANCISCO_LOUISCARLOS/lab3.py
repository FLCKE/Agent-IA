from typing import Annotated, Literal, TypedDict
from datetime import datetime
from zoneinfo import ZoneInfo

from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode


# -----------------------------
# 1) State schema (IMPORTANT)
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -----------------------------
# 2) Tools
# -----------------------------
@tool
def get_weather(city: str) -> str:
    """Obtenir la météo d'une ville."""
    return f"Il fait 23°C à {city}"

@tool
def get_news(topic: str) -> str:
    """Lire les actualités sur un sujet."""
    return f"Top news sur {topic} : 1) ... 2) ... 3) ..."

@tool
def get_time(city: str) -> str:
    """Donner l'heure locale dans une ville."""
    tz_map = {
        "Paris": "Europe/Paris",
        "Tokyo": "Asia/Tokyo",
        "New York": "America/New_York",
        "London": "Europe/London",
        "Abidjan": "Africa/Abidjan",
    }
    tz = tz_map.get(city, "Europe/Paris")
    now = datetime.now(ZoneInfo(tz))
    return f"Il est {now.strftime('%H:%M')} à {city} ({tz})"


tools = [get_weather, get_news, get_time]
tool_node = ToolNode(tools)


# -----------------------------
# 3) LLM node
# -----------------------------
llm = ChatOllama(model="gemma3:1b", temperature=0.1).bind_tools(tools)

def llm_node(state: AgentState):
    # Ici, messages existe toujours (grâce au state schema)
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}


# -----------------------------
# 4) Router : décider si on exécute un tool
# -----------------------------
def route(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    if getattr(last, "tool_calls", None):
        return "tools"
    return "end"


# -----------------------------
# 5) Build graph
# -----------------------------
builder = StateGraph(AgentState)
builder.add_node("llm", llm_node)
builder.add_node("tools", tool_node)

builder.add_edge(START, "llm")
builder.add_conditional_edges("llm", route, {"tools": "tools", "end": END})
builder.add_edge("tools", "llm")

graph = builder.compile()


# -----------------------------
# 6) Demo (à capturer)
# -----------------------------
if __name__ == "__main__":
    tests = [
        "Quelle est la météo à Paris ?",
        "Donne-moi les news sur l'intelligence artificielle",
        "Quelle heure est-il à Tokyo ?",
    ]

    for q in tests:
        print("\nUtilisateur:", q)
        out = graph.invoke({"messages": [("user", q)]})

        # Réponse finale
        print("Agent:", out["messages"][-1].content)

        # Preuve du choix d'outil (super utile pour le screenshot)
        for m in out["messages"]:
            if getattr(m, "tool_calls", None):
                print("Tool calls détectés:", m.tool_calls)
                break
