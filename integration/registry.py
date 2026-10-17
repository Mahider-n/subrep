"""Trusted, explicit action-to-controller registration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Callable

from .contracts import ObjectiveSchema, canonical, digest, identifier, validate_action


OBJECTIVE_REGISTRY = {"lander_safety_fuel_v2": ObjectiveSchema()}


@dataclass(frozen=True)
class PolicyRegistration:
    action_id: str
    behavior_version: str
    schema: ObjectiveSchema
    artifact_digest: str
    artifact_paths: tuple[str, ...]
    configuration_json: str
    provider_version: str
    factory: Callable

    @property
    def skill_id(self):
        return f"{self.action_id}.{self.behavior_version}.{self.identity}"

    @property
    def identity(self):
        return digest(self.manifest())

    def manifest(self):
        return {"action_id": self.action_id, "behavior_version": self.behavior_version,
                "schema_id": self.schema.identity, "artifact_digest": self.artifact_digest,
                "configuration": self.configuration_json, "provider_version": self.provider_version}

    def verify(self):
        if artifact_digest(self.artifact_paths, self.configuration_json) != self.artifact_digest:
            raise ValueError("artifact_digest_changed")

    def new_policy(self, seed: int):
        self.verify()
        policy = self.factory(seed)
        if not callable(policy):
            raise ValueError("unresolved_executable")
        return policy


def artifact_digest(paths, configuration_json: str):
    ordered = sorted(Path(path).resolve() for path in paths)
    if not ordered or len({path.name for path in ordered}) != len(ordered):
        raise ValueError("artifact bundle requires files with unique names")
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in ordered}
    return digest({"files": files, "configuration": configuration_json})


class PolicyRegistry:
    """One current immutable behavior registration per original Action.id."""

    def __init__(self, schema: ObjectiveSchema):
        if OBJECTIVE_REGISTRY.get(schema.version) != schema:
            raise ValueError("unknown_objective_schema")
        self.schema = schema
        self._entries: dict[str, PolicyRegistration] = {}

    def register(self, action, *, behavior_version: str, artifact_paths,
                 factory: Callable, configuration=None, provider_version="lander_python_v1"):
        validate_action(action)
        identifier(behavior_version, "behavior_version")
        identifier(provider_version, "provider_version")
        if action.id in self._entries:
            raise ValueError("action_id_already_registered")
        if not callable(factory):
            raise ValueError("unresolved_executable")
        paths = tuple(str(Path(p).resolve()) for p in artifact_paths)
        config = canonical(configuration or {})
        entry = PolicyRegistration(action.id, behavior_version, self.schema,
                                   artifact_digest(paths, config), paths, config,
                                   provider_version, factory)
        self._entries[action.id] = entry
        return entry

    def resolve(self, action):
        validate_action(action)
        entry = self.get(action.id)
        entry.verify()
        return entry

    def get(self, action_id: str):
        if action_id not in self._entries:
            raise ValueError("unknown_action_id")
        return self._entries[action_id]
