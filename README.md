# langchain-demo-agent

A minimal LangChain agent for the agent platform, and the reference for code (non-Langflow) agents: a tool-calling
agent (`create_agent`) served over AG-UI by `ag-ui-langgraph`, in its own container. The platform only routes to it.

| | |
|---|---|
| Interface | `POST /run` (AG-UI), `GET /health` |
| Models | `OPENAI_BASE_URL` / `OPENAI_API_KEY` = the platform's model gateway and this agent's key; `MODEL` |
| Identity | `x-end-user-id` from the gateway, forwarded with `x-agent-slug` on model calls |
| Tools | `current_time`, `calculator`, `word_count` |
| State | in memory (demo only: threads are lost on restart) |

## Run it next to a local agent-platform stack
```bash
cp .env.example .env            # GATEWAY_KEY = LANGCHAIN_DEMO_GATEWAY_KEY from agent-platform/.env
docker compose up -d --build    # joins the platform network as langchain-demo:8000
uv run ../agent-platform/scripts/publish.py deploy .     # register agent.yaml: route + portal entry
```
Then ask it in the portal ("What time is it in Kolkata?"), or run the platform e2e with it:
`uv run ../agent-platform/scripts/e2e.py --model-agent langchain-demo --prompt "What is 17 * 23? Use the calculator."`

## License

[Apache-2.0](LICENSE)
