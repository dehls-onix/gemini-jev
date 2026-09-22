"""
Asynchronous Multi-Agent Task Scheduler and State Reducer for Tool Selection.
Coordinates JEV, LAYA, BigQuery ML, CatBoost, XGBoost, GradientXGB, Scikit-Learn, and Statsmodels concurrently.
"""

import asyncio
import time
import uuid
from typing import AsyncGenerator, Dict, List, Optional
from src.adapters.bqml_adapter import BigQueryMLAdapter
from src.adapters.jev_adapter import JevAdapter
from src.adapters.laya_adapter import LayaAdapter
from src.adapters.traditional_ml_adapters import (
    CatBoostAdapter,
    XGBoostAdapter,
    GradientXGBAdapter,
    ScikitLearnAdapter,
    StatsmodelsAdapter,
)
from src.adapters.gemini_multimodal_adapter import GeminiMultimodalAdapter
from src.models import (
    EnginePrediction,
    StateSnapshot,
    TaskSpec,
    WorkspaceEvent,
    WorkspaceState,
)

class TaskCoordinator:
    def __init__(self):
        self.gemini_adapter = GeminiMultimodalAdapter()
        self.bq_adapter = BigQueryMLAdapter()
        self.jev_adapter = JevAdapter()
        self.laya_adapter = LayaAdapter()
        self.catboost_adapter = CatBoostAdapter()
        self.xgboost_adapter = XGBoostAdapter()
        self.gradientxgb_adapter = GradientXGBAdapter()
        self.sklearn_adapter = ScikitLearnAdapter()
        self.statsmodels_adapter = StatsmodelsAdapter()

    async def execute_comparison(
        self,
        snapshot: StateSnapshot,
        task: TaskSpec,
        run_id: Optional[str] = None
    ) -> AsyncGenerator[WorkspaceEvent, None]:
        if not run_id:
            run_id = f"run_{uuid.uuid4().hex[:8]}"

        yield WorkspaceEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            task_id=task.task_id,
            event_type="run.created",
            payload={
                "status": "initialized",
                "active_engines": [
                    "Gemini_Multimodal", "JEV", "LAYA", "BigQuery_ML", "CatBoost", "XGBoost", "GradientXGB", "Scikit_Learn", "Statsmodels"
                ]
            }
        )

        async def run_engine(engine_name: str, coro):
            try:
                res = await coro
                return engine_name, res, None
            except Exception as e:
                return engine_name, None, str(e)

        tasks = [
            run_engine("Gemini_Multimodal", self.gemini_adapter.predict_single(snapshot, task, run_id)),
            run_engine("JEV", self.jev_adapter.predict_single(snapshot, task, run_id)),
            run_engine("LAYA", self.laya_adapter.predict_single(snapshot, task, run_id)),
            run_engine("BigQuery_ML", self.bq_adapter.predict_single(snapshot, task, run_id)),
            run_engine("CatBoost", self.catboost_adapter.predict_single(snapshot, task, run_id)),
            run_engine("XGBoost", self.xgboost_adapter.predict_single(snapshot, task, run_id)),
            run_engine("GradientXGB", self.gradientxgb_adapter.predict_single(snapshot, task, run_id)),
            run_engine("Scikit_Learn", self.sklearn_adapter.predict_single(snapshot, task, run_id)),
            run_engine("Statsmodels", self.statsmodels_adapter.predict_single(snapshot, task, run_id)),
        ]

        for completed_coro in asyncio.as_completed(tasks):
            engine_name, prediction, error = await completed_coro
            if prediction:
                yield WorkspaceEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    run_id=run_id,
                    task_id=task.task_id,
                    event_type="engine.result",
                    payload={"engine": engine_name, "prediction": prediction.model_dump()}
                )
            else:
                yield WorkspaceEvent(
                    event_id=f"evt_{uuid.uuid4().hex[:8]}",
                    run_id=run_id,
                    task_id=task.task_id,
                    event_type="task.failed",
                    payload={"engine": engine_name, "error": error}
                )

        yield WorkspaceEvent(
            event_id=f"evt_{uuid.uuid4().hex[:8]}",
            run_id=run_id,
            task_id=task.task_id,
            event_type="run.completed",
            payload={"status": "completed"}
        )

class StateReducer:
    @staticmethod
    def reduce(state: WorkspaceState, event: WorkspaceEvent) -> WorkspaceState:
        new_state = state.model_copy(deep=True)
        new_state.events_log.append(event)
        new_state.revision += 1

        if event.event_type == "engine.result":
            payload = event.payload
            pred_dict = payload["prediction"]
            pred = EnginePrediction(**pred_dict)
            new_state.predictions[pred.engine] = pred
            new_state.engine_status[pred.engine] = "completed"

            # Check consensus
            labels = [p.predicted_label for p in new_state.predictions.values()]
            if len(set(labels)) > 1:
                new_state.disagreements = list(set(labels))
            else:
                new_state.disagreements = []

            # Compute winner
            if new_state.predictions:
                best_engine = max(
                    new_state.predictions.values(),
                    key=lambda x: (x.confidence or 0.0)
                )
                new_state.winner_engine = best_engine.engine

                tools = [f"{p.engine}: {p.predicted_label} ({round((p.confidence or 0)*100, 1)}%)" for p in new_state.predictions.values()]
                new_state.synthesis_summary = (
                    f"Selected Primary Tool: '{best_engine.predicted_label}'. "
                    f"Consensus across {len(new_state.predictions)} evaluators: {', '.join(tools)}."
                )

        return new_state
