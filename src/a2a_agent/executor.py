"""
A2A Agent Executor for Decision Workbench.
Handles A2A protocol: message routing, task lifecycle, A2UI extension detection.
"""

import logging
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    DataPart,
    Part,
    Task,
    TaskState,
    TextPart,
    UnsupportedOperationError,
)
from a2a.utils import (
    new_agent_parts_message,
    new_agent_text_message,
    new_task,
)
from a2a.utils.errors import ServerError
from a2ui.a2a.extension import try_activate_a2ui_extension
from src.a2a_agent.agent import DecisionWorkbenchAgent
import base64
from a2a.types import FilePart, FileWithBytes, FileWithUri
from src.adapters.gemini_multimodal_adapter import GeminiMultimodalAdapter
from src.models import MultimodalMedia


logger = logging.getLogger(__name__)

class DecisionWorkbenchAgentExecutor(AgentExecutor):
    def __init__(self, agent: DecisionWorkbenchAgent):
        self._agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        query = ""
        action = None

        logger.info(f"--- Requested extensions: {context.requested_extensions} ---")

        # Detect active A2UI version requested by Gemini Enterprise (default to v0.8)
        active_ui_version = try_activate_a2ui_extension(context, self._agent.agent_card) or "v0.8"
        logger.info(f"--- Active A2UI version: {active_ui_version} ---")

        media_part_info = None

        if context.message and context.message.parts:
            for part in context.message.parts:
                # Direct TextPart or wrapped in root
                p_obj = part.root if hasattr(part, "root") else part
                if isinstance(p_obj, TextPart):
                    query += p_obj.text + " "
                elif isinstance(p_obj, FilePart):
                    f = p_obj.file
                    if isinstance(f, FileWithBytes) and f.bytes:
                        try:
                            raw_b = base64.b64decode(f.bytes)
                            media_part_info = (raw_b, f.mime_type or "image/png", f.name)
                        except Exception as ex:
                            logger.error(f"Failed decoding FilePart bytes: {ex}")
                    elif isinstance(f, FileWithUri) and f.uri:
                        # GCS URI reference or remote media
                        media_part_info = (f.uri, f.mime_type or "video/mp4", f.name)

        query = query.strip()

        # Execute multimodal preprocessing if user uploaded/pasted image or video
        multimodal_media: Optional[MultimodalMedia] = None
        if media_part_info:
            media_content, mime_type, filename = media_part_info
            try:
                mm_adapter = GeminiMultimodalAdapter()
                if isinstance(media_content, bytes):
                    multimodal_media, parsed = await mm_adapter.stage_media_and_analyze(
                        media_bytes=media_content,
                        mime_type=mime_type,
                        filename=filename,
                        user_prompt=query
                    )
                    # Enrich query with extracted OCR text & visual context
                    extracted_ocr = parsed.get("extracted_text", "")
                    visual_summary = parsed.get("multimodal_summary", "")
                    query = f"{query} [Visual Summary: {visual_summary}] [OCR: {extracted_ocr}]".strip()
            except Exception as e:
                logger.error(f"Multimodal media processing failed: {e}")

        if not query:
            query = "Invoice #INV-2026-992 Net 30 Total USD 4,520.00 Vendor Acme Industrial."

        # Setup task
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            parts = await self._agent.execute_task(
                text=query,
                title="GE Multimodal Document Analysis" if multimodal_media else "GE Document Analysis",
                ui_version=active_ui_version,
                multimodal_media=multimodal_media
            )
            await updater.update_status(
                TaskState.completed,
                new_agent_parts_message(parts, task.context_id, task.id),
                final=True,
            )
        except Exception as e:
            logger.error(f"Error executing agent task: {e}")
            await updater.fail(error=ServerError(message=str(e)))

    async def cancel(
        self, request: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        raise ServerError(error=UnsupportedOperationError())
