"""Local Ollama strategic-planning adapter and safe plan translation."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import asdict
from typing import Any

from .engine import GameEngine
from .model import Action, Faction, Hex

SYSTEM_PROMPT = """You are a strategic planner for a hex-grid simulation. Return JSON only:
{"intent":"brief rationale","tasks":[{"actor_id":"blue-1","action":"move|attack|ew|wait",
"target_id":null,"target_hex":{"q":0,"r":0}}]}. Protect white units and use only visible information."""


def observation(engine: GameEngine, faction: Faction) -> dict[str, Any]:
    return {
        "time": engine.time,
        "map": {"width": engine.map.width, "height": engine.map.height},
        "terrain": [{"q": h.q, "r": h.r, "type": t.value} for h, t in engine.map.terrain.items()],
        "units": [u.public_dict() for u in engine.visible_units(faction)],
        "recent_events": [asdict(e) for e in list(engine.events)[-20:]],
    }


class OllamaPlanner:
    def __init__(self, model: str = "qwen2.5:7b", host: str = "http://localhost:11434"):
        self.model, self.host = model, host.rstrip("/")

    def plan(self, state: dict[str, Any], timeout: float = 60) -> dict[str, Any]:
        payload = json.dumps({"model": self.model, "stream": False, "format": "json", "prompt": (
            SYSTEM_PROMPT + "\nSTATE:\n" + json.dumps(state)
        )}).encode()
        request = urllib.request.Request(
            f"{self.host}/api/generate", payload, {"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(json.loads(response.read())["response"])


def actions_from_plan(plan: dict[str, Any], engine: GameEngine, faction: Faction) -> list[Action]:
    """Validate untrusted LLM output before exposing it to the engine."""
    actions = []
    for task in plan.get("tasks", []):
        actor = engine.units.get(str(task.get("actor_id", "")))
        kind = task.get("action")
        if not actor or actor.faction != faction or kind not in {"move", "attack", "ew", "wait"}:
            continue
        raw_hex = task.get("target_hex")
        try:
            target_hex = Hex(int(raw_hex["q"]), int(raw_hex["r"])) if isinstance(raw_hex, dict) else None
        except (KeyError, TypeError, ValueError):
            continue
        if target_hex is not None and not engine.map.contains(target_hex):
            continue
        actions.append(Action(actor.id, kind, target_hex, task.get("target_id")))
    return actions
