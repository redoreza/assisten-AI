"""Application configuration loaded from environment / .env file.

Pydantic-settings reads the project-root `.env` (one directory above `backend/`)
so the same file works for backend, scripts, and future frontend tooling.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API keys
    groq_api_key: str = Field(default="", description="Groq API key (required for chat)")
    voyage_api_key: str = Field(default="", description="Voyage embeddings key (Phase 5)")
    gemini_api_key: str = Field(default="", description="Optional Gemini fallback key")
    tavily_api_key: str = Field(
        default="",
        description="Tavily web search key — enables LLM tool calling for current-info questions",
    )
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key — when set, used as a backup LLM alongside Groq",
    )
    openrouter_model: str = "meta-llama/llama-3.3-70b-instruct:free"
    nvidia_api_key: str = Field(
        default="",
        description="NVIDIA NIM API key (build.nvidia.com) — primary LLM when set, first in rotation",
    )
    nvidia_model: str = "meta/llama-3.3-70b-instruct"

    # Azure Speech Services (Cognitive Services). When the key is set, the TTS
    # router uses Azure (SSML, prosody/break tags) instead of free Edge TTS.
    azure_speech_key: str = Field(
        default="",
        description="Azure Speech key (primary) — enables higher-quality SSML TTS when set",
    )
    azure_speech_key_2: str = Field(
        default="",
        description="Azure Speech key (backup) — tried when primary key hits rate limit",
    )
    azure_speech_region: str = "southeastasia"

    # Model selection
    llm_model: str = "llama-3.3-70b-versatile"
    # Dedicated fast model for light chat — uses NVIDIA NIM (llama-3.1-70b ~3s, stabil)
    light_chat_model: str = "meta/llama-3.1-70b-instruct"
    stt_model: str = "whisper-large-v3-turbo"
    embedding_model: str = "nvidia/nv-embedqa-e5-v5"
    tts_voice_default: str = "id-ID-GadisNeural"
    # Hard timeout (seconds) applied to search_task after the filler loop exits
    search_timeout_s: float = 2.0

    # Server
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    # Storage paths (resolved relative to project root)
    sqlite_path: str = "./data/sqlite/app.db"
    chroma_path: str = "./data/chroma"
    personas_dir: str = "./data/personas"
    kb_dir: str = "./data/knowledge_base"
    faces_db_path: str = "./data/sqlite/faces.db"

    # Face recognition (InsightFace)
    # buffalo_l is the accuracy-first model (~300MB). Matches the detection-engine
    # baseline. Smoothness in the UI comes from frontend bbox interpolation, NOT
    # from a smaller model.
    face_model_name: str = "buffalo_l"
    face_match_threshold: float = 0.5
    face_det_size: int = 640
    # Set to false to force CPU-only inference even when CUDA is available.
    face_use_gpu: bool = True
    # Continuous learning — ported from detection-engine. When recognize() finds
    # a match with similarity ≥ adaptive_threshold, the new embedding is
    # appended to that person's profile (up to max_emb_per_person). Builds up
    # pose / lighting / expression variants over time without manual training.
    face_adaptive_enabled: bool = True
    face_adaptive_threshold: float = 0.65
    face_max_emb_per_person: int = 20

    # Defaults
    default_persona: str = "pointer"
    default_mode: str = "companion"

    # RAG (Phase 5)
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    rag_top_k: int = 3
    rag_min_similarity: float = 0.65

    # Memory
    memory_max_turns: int = 10

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def personas_path(self) -> Path:
        return (PROJECT_ROOT / self.personas_dir).resolve()

    @property
    def sqlite_full_path(self) -> Path:
        return (PROJECT_ROOT / self.sqlite_path).resolve()

    @property
    def chroma_full_path(self) -> Path:
        return (PROJECT_ROOT / self.chroma_path).resolve()

    @property
    def faces_db_full_path(self) -> Path:
        return (PROJECT_ROOT / self.faces_db_path).resolve()

    @property
    def kb_full_path(self) -> Path:
        return (PROJECT_ROOT / self.kb_dir).resolve()


settings = Settings()
