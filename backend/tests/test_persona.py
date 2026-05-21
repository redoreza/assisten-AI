"""Persona loader unit tests — no network, no real data files."""

from __future__ import annotations

import json
from pathlib import Path

from app.core.persona import Persona, PersonaManager


def _write_persona(dir_: Path, data: dict) -> None:
    (dir_ / f"{data['id']}.json").write_text(json.dumps(data), encoding="utf-8")


def test_loads_valid_persona(tmp_path: Path) -> None:
    _write_persona(
        tmp_path,
        {
            "id": "test",
            "name": "Tester",
            "personality": "rapi",
            "background": "lab",
            "speaking_style": "formal",
            "language": "id",
        },
    )
    mgr = PersonaManager(tmp_path)
    mgr.load_all()
    assert mgr.list_ids() == ["test"]
    p = mgr.get("test")
    assert isinstance(p, Persona)
    assert p.name == "Tester"


def test_render_system_prompt_substitutes_placeholders(tmp_path: Path) -> None:
    _write_persona(
        tmp_path,
        {
            "id": "x",
            "name": "Xenia",
            "personality": "tenang",
            "background": "Mars",
            "speaking_style": "santai",
            "language": "id",
            "system_prompt_template": "Saya {name}. Sifat: {personality}. Asal: {background}. "
            "Gaya: {speaking_style}. Bahasa: {language}.",
        },
    )
    mgr = PersonaManager(tmp_path)
    mgr.load_all()
    prompt = mgr.get("x").render_system_prompt()
    assert "Xenia" in prompt
    assert "tenang" in prompt
    assert "Mars" in prompt
    assert "{name}" not in prompt


def test_unknown_persona_falls_back_to_first_loaded(tmp_path: Path) -> None:
    _write_persona(tmp_path, {"id": "alpha", "name": "Alpha"})
    _write_persona(tmp_path, {"id": "beta", "name": "Beta"})
    mgr = PersonaManager(tmp_path)
    mgr.load_all()
    fallback = mgr.get("does_not_exist")
    assert fallback.id in {"alpha", "beta"}


def test_empty_dir_uses_default_persona(tmp_path: Path) -> None:
    mgr = PersonaManager(tmp_path)
    mgr.load_all()
    default = mgr.get("anything")
    assert default.id == "default"
    assert default.name == "Assistant"


def test_invalid_json_is_skipped(tmp_path: Path) -> None:
    (tmp_path / "broken.json").write_text("not valid json {{", encoding="utf-8")
    _write_persona(tmp_path, {"id": "good", "name": "Good"})
    mgr = PersonaManager(tmp_path)
    mgr.load_all()
    assert mgr.list_ids() == ["good"]
