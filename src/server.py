"""
Unified Server for Gemini Enterprise:
- POST / -> A2A Starlette Application (JSON-RPC 2.0 protocol endpoint)
- GET /.well-known/agent-card.json -> Agent Discovery
- GET / -> Interactive HTML Decision Workbench Dashboard
- POST /api/compare -> Real-time multi-model streaming benchmark
"""

import logging
import os
import json
import uvicorn
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.responses import JSONResponse, FileResponse
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore

from src.config import settings
from src.main import app as web_api_app
from src.a2a_agent.agent import DecisionWorkbenchAgent
from src.a2a_agent.executor import DecisionWorkbenchAgentExecutor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("unified-workbench-server")

def clean_dict_nulls(d):
    if isinstance(d, dict):
        return {k: clean_dict_nulls(v) for k, v in d.items() if v is not None}
    elif isinstance(d, list):
        return [clean_dict_nulls(v) for v in d if v is not None]
    return d

def create_unified_app():
    service = os.getenv("K_SERVICE", "gemini-jev-decision-workbench")
    region = os.getenv("CLOUD_RUN_REGION", "us-central1")
    cloud_run_url = os.getenv("CLOUD_RUN_URL", f"https://{service}-36231825761.{region}.run.app")

    agent = DecisionWorkbenchAgent(base_url=cloud_run_url)
    executor = DecisionWorkbenchAgentExecutor(agent)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    a2a_server = A2AStarletteApplication(
        agent_card=agent.agent_card,
        http_handler=request_handler,
    )
    a2a_app = a2a_server.build()

    async def get_index(request):
        return FileResponse("static/index.html")

    async def get_agent_card(request):
        raw_card = json.loads(agent.agent_card.model_dump_json(exclude_none=True))
        return JSONResponse(clean_dict_nulls(raw_card))

    # Master ASGI dispatch:
    # If POST / -> A2A JSON-RPC 2.0 (Gemini Enterprise)
    # If GET / -> Web index
    # If /.well-known/... -> Agent Card
    # If /api/... -> FastAPI web_api_app
    async def master_app(scope, receive, send):
        if scope["type"] == "http":
            path = scope.get("path", "")
            method = scope.get("method", "GET").upper()

            if path in ("/.well-known/agent-card.json", "/.well-known/agent.json"):
                resp = await get_agent_card(scope)
                await resp(scope, receive, send)
                return

            if path == "/" and method == "POST":
                # Route directly to A2A handler
                await a2a_app(scope, receive, send)
                return

            if path == "/" and method == "GET":
                resp = await get_index(scope)
                await resp(scope, receive, send)
                return

        # Fallback to FastAPI for /api/compare, /healthz, /static
        await web_api_app(scope, receive, send)

    return master_app

app = create_unified_app()

if __name__ == "__main__":
    port = int(os.getenv("PORT", settings.port))
    uvicorn.run("src.server:app", host="0.0.0.0", port=port, reload=False)
