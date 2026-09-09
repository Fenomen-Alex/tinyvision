"""Environment identity management.

An environment is the smallest independent unit for this project: a camera
(sensor + optics) mounted at a fixed viewpoint in a fixed controlled or
semi-controlled lighting/background situation, observing one visible printer
appearance. Two recordings belong to the same environment only if they share
the same physical setup, camera, viewpoint, lighting and printer.

The persistent ID is a random ``env-<uuid4>`` that is never regenerated. The
human-readable ``environment_key`` (``env_<source>_<contributor>_<camera>_<view>``)
is stable and unique *within* a manifest; reusing a key for a different
environment is an error.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from tv.envdata.schema import Environment, environment_key


@dataclass
class EnvironmentRegistry:
    """Registry mapping keys -> IDs, used to keep environment identity stable."""

    _by_key: dict[str, str] = field(default_factory=dict)

    def register(
        self,
        environment_id: str,
        environment_key: str,
        allow_same_id: bool = True,
    ) -> None:
        existing = self._by_key.get(environment_key)
        if existing is not None:
            if existing == environment_id:
                return
            if allow_same_id:
                # Same physical environment listed twice under one key -> same ID.
                if existing != environment_id:
                    raise ValueError(
                        f"environment_key {environment_key!r} is already bound to "
                        f"{existing}, refusing to rebind to {environment_id}"
                    )
            else:
                raise ValueError(
                    f"environment_key {environment_key!r} maps to more than one "
                    f"environment ID ({existing}, {environment_id})"
                )
        self._by_key[environment_key] = environment_id

    def resolve(self, environment_key: str, environment_id: str) -> str:
        """Return the ID for a key: the registered ID, or the one we were given."""
        if environment_key in self._by_key:
            return self._by_key[environment_key]
        self._by_key[environment_key] = environment_id
        return environment_id

    @classmethod
    def from_manifest(cls, environments: list[Environment]) -> "EnvironmentRegistry":
        reg = cls()
        for env in environments:
            reg.register(env.environment_id, env.environment_key)
        return reg


def new_environment_id() -> str:
    return f"env-{uuid.uuid4().hex}"


def make_environment(
    source: str,
    contributor: str | None,
    camera_identity: str | None,
    view: str,
    registry: EnvironmentRegistry | None = None,
    environment_id: str | None = None,
) -> Environment:
    """Create an Environment with a stable key and (optionally forced) ID."""
    key = environment_key(source, contributor, camera_identity, view)
    env_id = environment_id or new_environment_id()
    if registry is not None:
        env_id = registry.resolve(key, env_id)
    return Environment(
        environment_id=env_id,
        environment_key=key,
        source_dataset=source,
        contributor=contributor,
        camera={"identity": camera_identity},
        view=view,
    )