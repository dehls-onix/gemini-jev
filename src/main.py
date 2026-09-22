import asyncio
import json
import logging
from typing import Optional
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config import settings
from src.engine.coordinator import StateReducer, TaskCoordinator
from src.models import StateSnapshot, TaskSpec, WorkspaceState
from src.a2a_agent.ui_v08_builder import build_a2ui_v08_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gemini-jev-workbench")

app = FastAPI(
    title="Gemini Enterprise Decision Workbench",
    description="A2UI v0.9 Decision Workbench comparing JEV, LAYA, and BigQuery ML",
    version="1.0.0"
)

# Mount static web UI assets
app.mount("/static", StaticFiles(directory="static"), name="static")

coordinator = TaskCoordinator()

# Authoritative in-memory state store for active runs
_latest_workspace_state: Optional[WorkspaceState] = None

class DocumentRequest(BaseModel):
    doc_id: str
    title: str
    text_content: str
    ground_truth: Optional[str] = None

@app.get("/")
async def root():
    return FileResponse("static/index.html")

@app.get("/healthz")
async def healthz():
    return {
        "status": "healthy",
        "project_id": settings.gcp_project_id,
        "dataset": settings.bigquery_dataset,
        "a2ui_version": settings.a2ui_version,
        "engines": [
            "Gemini_Multimodal", "JEV", "LAYA", "BigQuery_ML",
            "CatBoost", "XGBoost", "GradientXGB", "Scikit_Learn", "Statsmodels"
        ]
    }

@app.post("/api/compare")
async def compare_document(req: DocumentRequest):
    """
    Primary multi-model comparison route.
    Streams Server-Sent Events (SSE) as each specialist agent completes its classification.
    """
    global _latest_workspace_state

    snapshot = StateSnapshot(
        doc_id=req.doc_id,
        title=req.title,
        text_content=req.text_content,
        ground_truth=req.ground_truth
    )

    task = TaskSpec(
        task_id=f"classify_{req.doc_id}",
        task_kind="choice",
        target="Document classification vertical",
        labels=["training_search", "servicenow_article_search", "servicenow_form_search", "servicenow_case_search", "intranet_home_search", "sharepoint_doc_search", "confluence_wiki_search", "jira_issue_search", "github_code_search", "artifactory_package_search", "tableau_report_search"],
        evidence_snapshot_id=req.doc_id,
        ground_truth=req.ground_truth
    )

    import uuid
    active_run_id = f"run_{uuid.uuid4().hex[:8]}"
    current_state = WorkspaceState(
        run_id=active_run_id,
        current_doc=snapshot,
        task_spec=task
    )

    async def event_generator():
        global _latest_workspace_state
        nonlocal current_state
        global _latest_workspace_state

        async for event in coordinator.execute_comparison(snapshot, task, run_id=active_run_id):
            current_state = StateReducer.reduce(current_state, event)
            _latest_workspace_state = current_state

            # Format as SSE
            event_json = event.model_dump_json()
            yield f"data: {event_json}\n\n"
            # Small non-blocking yield for client render breathing room
            await asyncio.sleep(0.01)

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/a2ui/latest")
async def get_latest_a2ui():
    """
    returns the latest compiled a2ui wire specification.
    """
    global _latest_workspace_state
    if not _latest_workspace_state:
        return JSONResponse({"status": "no_active_run"})

    messages = build_a2ui_v08_response(_latest_workspace_state)
    return JSONResponse({
        "run_id": _latest_workspace_state.run_id,
        "revision": _latest_workspace_state.revision,
        "winner_engine": _latest_workspace_state.winner_engine,
        "summary": _latest_workspace_state.synthesis_summary,
        "a2ui_messages": messages
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=settings.debug)
