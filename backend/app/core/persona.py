"""Persona loader.

Reads JSON files from data/personas/ at startup, validates each via the Persona
model, and exposes a PersonaManager singleton for the rest of the app to query.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from loguru import logger
from pydantic import BaseModel, Field

from app.config import settings


class VoiceConfig(BaseModel):
    provider: Literal["edge", "azure", "openai", "elevenlabs"] = "edge"
    voice_id: str = "id-ID-GadisNeural"
    rate: str = "+0%"
    pitch: str = "+0Hz"


class Persona(BaseModel):
    id: str
    name: str
    avatar_file: str = ""
    language: str = "id"
    personality: str = ""
    background: str = ""
    speaking_style: str = ""
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    expressions: dict[str, dict[str, float]] = Field(default_factory=dict)
    system_prompt_template: str = (
        "Kamu adalah {name}. {personality} Background: {background}. "
        "Cara bicara: {speaking_style}. Jawab dalam bahasa {language}. "
        "JANGAN keluar dari karakter."
    )

    def render_system_prompt(self) -> str:
        return self.system_prompt_template.format(
            name=self.name,
            personality=self.personality,
            background=self.background,
            speaking_style=self.speaking_style,
            language=self.language,
        )


class PersonaManager:
    def __init__(self, personas_dir: Path) -> None:
        self._dir = personas_dir
        self._personas: dict[str, Persona] = {}

    def load_all(self) -> None:
        self._personas.clear()
        if not self._dir.exists():
            logger.warning(f"Personas dir does not exist: {self._dir}")
            return
        for path in sorted(self._dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                persona = Persona.model_validate(data)
                self._personas[persona.id] = persona
                logger.info(f"Loaded persona: {persona.id} ({persona.name})")
            except Exception as exc:
                logger.error(f"Failed to load persona from {path}: {exc}")
        if not self._personas:
            logger.warning("No personas loaded — endpoints will use fallback persona")

    def get(self, persona_id: str) -> Persona:
        if persona_id in self._personas:
            return self._personas[persona_id]
        if self._personas:
            fallback_id = next(iter(self._personas))
            logger.warning(f"Persona '{persona_id}' not found, falling back to '{fallback_id}'")
            return self._personas[fallback_id]
        return Persona(id="default", name="Assistant", language="id")

    def list_ids(self) -> list[str]:
        return list(self._personas.keys())


persona_manager = PersonaManager(settings.personas_path)
