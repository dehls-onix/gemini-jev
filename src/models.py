import time
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

# ── Shared Semantic Task Contracts ──────────────────────────────────────────

class TaskSpec(BaseModel):
    task_id: str
    task_kind: Literal["choice", "score", "noul", "multiclass", "tool_selection"] = "tool_selection"
    target: str = Field(description="Semantic target description e.g. 'Enterprise RAG Tool Selection & Routing'")
    label_schema_version: str = "3.0"
    labels: List[str] = Field(default_factory=lambda: [
        "training_search",
        "servicenow_article_search",
        "servicenow_form_search",
        "servicenow_case_search",
        "intranet_home_search",
        "sharepoint_doc_search",
        "confluence_wiki_search",
        "jira_issue_search",
        "github_code_search",
        "artifactory_package_search",
        "tableau_report_search"
    ])
    evidence_snapshot_id: str
    state_revision: int = 1
    ground_truth: Optional[str] = None
    prediction_time: str = Field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))

class MultimodalMedia(BaseModel):
    media_kind: Literal["image", "video"]
    mime_type: str
    media_url: Optional[str] = None
    media_name: Optional[str] = None
    media_b64: Optional[str] = None
    extracted_text: Optional[str] = None
    multimodal_summary: Optional[str] = None
    latency_ms: float = 0.0

class StateSnapshot(BaseModel):
    doc_id: str
    title: str
    text_content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    multimodal_media: Optional[MultimodalMedia] = None
    ground_truth: Optional[str] = None
    created_at: float = Field(default_factory=time.time)

# ── Normalized Engine Result ────────────────────────────────────────────────

class EnginePrediction(BaseModel):
    run_id: str
    task_id: str
    engine: str  # e.g. "JEV", "LAYA", "BigQuery_ML", "CatBoost", "XGBoost", "GradientXGB", "Scikit_Learn", "Statsmodels"
    model_version: str
    predicted_label: str  # Primary Selected Tool
    secondary_label: Optional[str] = None  # Secondary / Fallback Tool
    sub_category: Optional[str] = None  # Specific Operational Sub-Category
    intent_nuance: Optional[str] = None  # Boundary disambiguation explanation
    probabilities: Dict[str, float] = Field(default_factory=dict)
    confidence: Optional[float] = None
    confidence_type: str = "calibrated"  # "native_entropy", "softmax", "calibrated", "margin", "gradient_prob"
    latency_ms: float
    cost_per_1k_usd: float
    is_correct: Optional[bool] = None
    evidence_attribution: Optional[str] = None
    status: Literal["completed", "needs_training", "not_configured", "failed", "timed_out"] = "completed"
    error_message: Optional[str] = None
    executed_at: float = Field(default_factory=time.time)

# ── Streaming Event Types ────────────────────────────────────────────────────

EventType = Literal[
    "run.created",
    "plan.ready",
    "task.started",
    "evidence.ready",
    "engine.progress",
    "engine.result",
    "reducer.update",
    "task.failed",
    "run.completed"
]

class WorkspaceEvent(BaseModel):
    event_id: str
    run_id: str
    task_id: str
    event_type: EventType
    timestamp: float = Field(default_factory=time.time)
    payload: Dict[str, Any] = Field(default_factory=dict)

# ── Multi-Agent Reducer State ────────────────────────────────────────────────

class WorkspaceState(BaseModel):
    run_id: str
    current_doc: Optional[StateSnapshot] = None
    task_spec: Optional[TaskSpec] = None
    active_engines: List[str] = Field(default_factory=lambda: [
        "Gemini_Multimodal", "JEV", "LAYA", "BigQuery_ML", "CatBoost", "XGBoost", "GradientXGB", "Scikit_Learn", "Statsmodels"
    ])
    engine_status: Dict[str, str] = Field(default_factory=dict)
    predictions: Dict[str, EnginePrediction] = Field(default_factory=dict)
    disagreements: List[str] = Field(default_factory=list)
    winner_engine: Optional[str] = None
    synthesis_summary: Optional[str] = None
    events_log: List[WorkspaceEvent] = Field(default_factory=list)
    revision: int = 1
