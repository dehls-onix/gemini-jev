"""
Gemini Enterprise Native A2A + A2UI Decision Workbench Agent.
Fully compliant with a2a-sdk + a2ui-agent-sdk specifications.
Specialized in Multi-Agent Tool Selection across 11 General Enterprise Destinations.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    DataPart,
    Part,
    TextPart,
)
from a2ui.schema.constants import VERSION_0_8, VERSION_0_9
from a2ui.a2a.extension import get_a2ui_agent_extension
from a2ui.a2a.parts import create_a2ui_part

from src.models import StateSnapshot, TaskSpec, WorkspaceState
from src.models import MultimodalMedia

from src.engine.coordinator import TaskCoordinator, StateReducer
from src.adapters.classifier_core import ENTERPRISE_TOOLS
from src.a2a_agent.ui_v08_builder import build_a2ui_v08_response

logger = logging.getLogger(__name__)

class DecisionWorkbenchAgent:
    """
    A2A agent coordinating JEV, LAYA, BigQuery ML, CatBoost, XGBoost, GradientXGB, Scikit-Learn, and Statsmodels
    for enterprise tool selection across 11 destinations (Tableau, ServiceNow, Confluence, Jira, GitHub, Artifactory, etc.),
    returning clean application/json+a2ui components directly to Gemini Enterprise.
    """

    SUPPORTED_CONTENT_TYPES = ["text/plain", "text", "application/json+a2ui"]

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.coordinator = TaskCoordinator()
        self._agent_card = self._build_agent_card()

    @property
    def agent_card(self) -> AgentCard:
        return self._agent_card

    def _build_agent_card(self) -> AgentCard:
        extensions = [
            get_a2ui_agent_extension(
                VERSION_0_8,
                accepts_inline_catalogs=True,
                supported_catalog_ids=[
                    "https://a2ui.org/specification/v0_8/standard_catalog_definition.json"
                ],
            ),
            get_a2ui_agent_extension(
                VERSION_0_9,
                accepts_inline_catalogs=True,
                supported_catalog_ids=[
                    "https://a2ui.org/specification/v0_9/catalogs/basic/catalog.json",
                    "https://www.gstatic.com/vertexaisearch/a2ui/v0_9/gemini_enterprise_composite_catalog.json"
                ],
            ),
        ]

        capabilities = AgentCapabilities(
            streaming=True,
            extensions=extensions,
        )

        skills = [
            AgentSkill(
                id="enterprise_tool_selection",
                name="Enterprise RAG Tool Selection & Intent Router",
                description="Disambiguates user intent and dynamically routes multimodal enterprise queries (text, screenshots, videos) across 11 tools using 9 concurrent evaluators led by Gemini 3.8 Flash.",
                tags=["tool_selection", "routing", "multimodal", "video", "image", "gemini_flash", "tableau", "servicenow", "confluence", "jira", "github", "artifactory", "sharepoint", "jev", "laya", "a2ui"],
                examples=[
                    "Where can I find the quarterly sales KPI dashboard and revenue metrics?",
                    "I need to fill out a request form to order a new replacement POS terminal.",
                    "What is the status of my open incident ticket INC0094812?",
                    "Where is the source code repository and open pull request for the auth service?"
                ]
            )
        ]

        return AgentCard(
            name="Decision Workbench Agent",
            description="Multi-model multimodal tool selection, intent disambiguation, and visual workbench for Gemini Enterprise.",
            url=self.base_url,
            version="3.1.0",
            default_input_modes=["text/plain", "image/png", "image/jpeg", "video/mp4"],
            default_output_modes=["text/plain", "application/json+a2ui"],
            capabilities=capabilities,
            skills=skills,
            protocol_version="0.3.0",
            preferred_transport="JSONRPC"
        )

    async def execute_task(
        self,
        text: str,
        title: str = "Enterprise Query",
        ui_version: Optional[str] = "v0.8",
        multimodal_media: Optional[MultimodalMedia] = None
    ) -> List[Part]:
        """
        Runs 9-engine tool selection comparison across 11 enterprise destinations and compiles markdown + A2UI.
        """
        snapshot = StateSnapshot(
            doc_id="enterprise_query",
            title=title,
            text_content=text,
            multimodal_media=multimodal_media
        )
        task = TaskSpec(
            task_id="tool_selection_task",
            target="Enterprise RAG Tool Selection & Routing",
            labels=ENTERPRISE_TOOLS,
            evidence_snapshot_id="enterprise_query"
        )

        state = WorkspaceState(
            run_id="enterprise_run",
            current_doc=snapshot,
            task_spec=task
        )

        # Execute concurrent task graph across all 9 engines
        async for event in self.coordinator.execute_comparison(snapshot, task):
            state = StateReducer.reduce(state, event)

        # Build clean enterprise markdown table comparing all 9 engines
        table_rows = []
        for eng_key, display_name in [
            ("Gemini_Multimodal", "Gemini 3.8 Flash"),
            ("JEV", "TypeSafe JEV"),
            ("LAYA", "Convai LAYA"),
            ("BigQuery_ML", "BigQuery ML"),
            ("CatBoost", "CatBoost"),
            ("XGBoost", "XGBoost"),
            ("GradientXGB", "GradientXGB"),
            ("Scikit_Learn", "Scikit-Learn"),
            ("Statsmodels", "Statsmodels"),
        ]:
            p = state.predictions.get(eng_key)
            if p:
                conf = f"**{round((p.confidence or 0)*100, 1)}%**"
                table_rows.append(
                    f"| **{display_name}** | `{p.predicted_label}` | `{p.secondary_label or 'N/A'}` | {conf} | {p.latency_ms:.1f}ms | {p.sub_category or ''} |"
                )

        table_str = "\n".join(table_rows)
        best_p = max(state.predictions.values(), key=lambda x: (x.confidence or 0.0)) if state.predictions else None

        media_prefix = ""
        if multimodal_media and multimodal_media.multimodal_summary:
            media_prefix = (
                f"**Visual Grounding ({multimodal_media.media_kind.upper()}):** {multimodal_media.multimodal_summary}\n\n"
            )

        rich_text = (
            f"### Decision Workbench: Multi-Engine Tool Selection & Intent Routing\n\n"
            f"{media_prefix}"
            f"**Synthesis:** {state.synthesis_summary}\n\n"
            f"| Evaluator | Primary Target Tool | Secondary Fallback | Confidence | Latency | Operational Sub-Category |\n"
            f"| :--- | :--- | :--- | :--- | :--- | :--- |\n"
            f"{table_str}\n\n"
            f"**Actionable RAG Routing Recommendation:**\n"
            f"- **Primary Destination:** Dispatch directly to **`{best_p.predicted_label if best_p else 'training_search'}`**.\n"
            f"- **Boundary Guardrail:** {best_p.intent_nuance if best_p else 'Domain-specific lexical grounding.'}"
        )

        parts: List[Part] = [TextPart(text=rich_text)]

        # Compile canonical A2UI v0.8 payloads (11 interactive tabs)
        a2ui_items = build_a2ui_v08_response(state)
        for item in a2ui_items:
            parts.append(create_a2ui_part(item, version=ui_version or "v0.8"))

        return parts
