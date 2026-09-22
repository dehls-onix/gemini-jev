import os
from typing import Optional
from pydantic import BaseModel
from dotenv import load_dotenv

# Load from local .env if present
load_dotenv()

class AppSettings(BaseModel):
    # GCP Infrastructure
    gcp_project_id: str = os.getenv("GCP_PROJECT_ID", "search-ahmed")
    gcp_project_number: str = os.getenv("GCP_PROJECT_NUMBER", "36231825761")
    gcp_location: str = os.getenv("GCP_LOCATION", "us-central1")
    bigquery_dataset: str = os.getenv("BIGQUERY_DATASET", "jev_laya_eval")

    # Gemini Enterprise / Discovery Engine
    ge_engine_id: str = os.getenv("GE_ENGINE_ID", "presales-ge-demo_1780625458586")
    ge_collection: str = os.getenv("GE_COLLECTION", "default_collection")
    ge_location: str = os.getenv("GE_LOCATION", "global")
    
    # Gemini Multimodal Models
    gemini_routing_model: str = os.getenv("GEMINI_ROUTING_MODEL", "gemini-3.8-flash")
    gemini_escalation_model: str = os.getenv("GEMINI_ESCALATION_MODEL", "gemini-3.8-flash")
    gemini_multimodal_location: str = os.getenv("GEMINI_MULTIMODAL_LOCATION", "global")
    media_staging_bucket: str = os.getenv("MEDIA_STAGING_BUCKET", "gemini-jev-media-staging")

    # External Provider Keys & Endpoints
    jev_api_key: Optional[str] = os.getenv("JEV_API_KEY", None)
    jev_base_url: str = os.getenv("JEV_BASE_URL", "https://api.typesafe.com/v1")
    
    huggingface_api_key: Optional[str] = os.getenv("HUGGINGFACE_API_KEY") or os.getenv("HUGGINGFACE_TOKEN", None)
    laya_model_id: str = os.getenv("LAYA_MODEL_ID", "NandhaKishorM/laya")
    laya_endpoint_url: Optional[str] = os.getenv("LAYA_ENDPOINT_URL", None)

    laya_repo_id: str = os.getenv("LAYA_REPO_ID", "convaiinnovations/laya")

    # Server Runtime
    port: int = int(os.getenv("PORT", "8080"))
    host: str = os.getenv("HOST", "0.0.0.0")
    debug: bool = os.getenv("DEBUG", "True").lower() in ("true", "1")
    a2ui_version: str = os.getenv("A2UI_VERSION", "v0.9")

settings = AppSettings()
