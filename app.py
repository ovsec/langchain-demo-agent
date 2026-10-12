"""LangChain demo agent for the agent platform: a tool-calling agent served over AG-UI.

Framework-agnostic contract with the platform:
  POST /run      AG-UI RunAgentInput in, AG-UI SSE events out (via ag-ui-langgraph)
  GET  /health   liveness
  env            OPENAI_BASE_URL / OPENAI_API_KEY (the platform's model gateway and this agent's key), MODEL
  header         x-end-user-id (set by the platform gateway): forwarded on model calls for attribution
"""

import ast
import contextvars
import operator
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from ag_ui_langgraph import LangGraphAgent, add_langgraph_fastapi_endpoint
from fastapi import FastAPI, Request
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

SLUG = "langchain-demo"
end_user: contextvars.ContextVar[str] = contextvars.ContextVar("end_user", default="anonymous")


@tool
def current_time(timezone: str = "Europe/Stockholm") -> str:
    """Current date and time in an IANA timezone, e.g. Europe/Stockholm or Asia/Kolkata."""
    return datetime.now(ZoneInfo(timezone)).strftime("%Y-%m-%d %H:%M:%S %Z")


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.USub: operator.neg}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("only numbers and + - * / ** % are allowed")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as (17 * 23) + 4 / 2."""
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except (ValueError, SyntaxError, ZeroDivisionError) as e:
        return f"error: {e}"


@tool
def word_count(text: str) -> str:
    """Count the words and characters in a text."""
    return f"{len(text.split())} words, {len(text)} characters"


async def _identity_headers(request: httpx.Request) -> None:
    # attribution in the gateway's LLM logs: which agent, on behalf of which user
    request.headers["x-agent-slug"] = SLUG
    request.headers["x-end-user-id"] = end_user.get()


model = ChatOpenAI(
    model=os.getenv("MODEL", "ollama/qwen3:4b"),
    base_url=os.environ["OPENAI_BASE_URL"],
    api_key=os.environ["OPENAI_API_KEY"],
    http_async_client=httpx.AsyncClient(timeout=300, event_hooks={"request": [_identity_headers]}),
)

graph = create_agent(
    model,
    tools=[current_time, calculator, word_count],
    system_prompt="You are a concise test assistant. Use a tool whenever it can answer the question.",
    checkpointer=InMemorySaver(),  # demo only: threads are lost on restart; real agents use a durable checkpointer
)

app = FastAPI(title="langchain-demo")


@app.middleware("http")
async def remember_end_user(request: Request, call_next):
    token = end_user.set(request.headers.get("x-end-user-id", "anonymous"))
    try:
        return await call_next(request)
    finally:
        end_user.reset(token)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


add_langgraph_fastapi_endpoint(
    app, LangGraphAgent(name=SLUG, graph=graph, description="LangChain demo agent"), path="/run"
)
