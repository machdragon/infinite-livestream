"""Artifact and model-package handling for MiniMax H3.

Loads the JSON Schema files from the ``schemas/`` directory alongside this
module, provides ``validate_package`` / ``validate_run`` validators, and defines
a ``ModelPackage`` dataclass for representing artifact packages in code.

The no-adapter-stack rule is enforced both in the schema (at most one
``adapter`` component, which must be ``temporary: true``) and in code via
``ModelPackage.adapter_count`` / ``ModelPackage.assert_no_adapter_stack``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import jsonschema
except ImportError:  # pragma: no cover - jsonschema is a test dependency
    jsonschema = None

_SCHEMA_DIR = Path(__file__).resolve().parent / "schemas"

_PACKAGE_SCHEMA_PATH = _SCHEMA_DIR / "model-package-v1.schema.json"
_RUN_SCHEMA_PATH = _SCHEMA_DIR / "training-run-v1.schema.json"


def _load_schema(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def get_package_schema() -> Dict[str, Any]:
    """Return the model-package-v1 schema dict."""
    return _load_schema(_PACKAGE_SCHEMA_PATH)


def get_run_schema() -> Dict[str, Any]:
    """Return the training-run-v1 schema dict."""
    return _load_schema(_RUN_SCHEMA_PATH)


def validate_package(data: Dict[str, Any]) -> None:
    """Validate *data* against the model-package-v1 schema.

    Raises ``jsonschema.ValidationError`` on failure. Also enforces the
    no-adapter-stack rule at the code level as a defence in depth.
    """
    if jsonschema is None:
        raise ImportError("jsonschema is required to validate packages")
    jsonschema.validate(data, get_package_schema())
    # Defence in depth: the schema already limits components.adapter to a
    # single object, but assert explicitly so the rule is testable in code.
    components = data.get("components", {})
    adapter = components.get("adapter")
    if adapter is not None:
        if not isinstance(adapter, dict):
            raise ValueError("adapter must be an object, not a list (adapter stack forbidden)")
        if adapter.get("temporary") is not True:
            raise ValueError("adapter must be temporary")


def validate_run(data: Dict[str, Any]) -> None:
    """Validate *data* against the training-run-v1 schema.

    Raises ``jsonschema.ValidationError`` on failure.
    """
    if jsonschema is None:
        raise ImportError("jsonschema is required to validate runs")
    jsonschema.validate(data, get_run_schema())


@dataclass
class Component:
    """A single typed component within a model package."""

    type: str
    name_or_path: Optional[str] = None
    native: Optional[bool] = None
    temporary: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelPackage:
    """In-code representation of a model-package-v1 artifact.

    Enforces the no-adapter-stack rule: at most one adapter component is
    permitted, and it must be flagged temporary.
    """

    package_id: str
    mode: str
    base: Component
    causal_trunk: Component
    pdd_heads: Component
    control_branch: Component
    adapter: Optional[Component] = None
    revisions: Dict[str, str] = field(default_factory=dict)
    hashes: Dict[str, str] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    dataset_revision: str = ""
    cost: Dict[str, Any] = field(default_factory=dict)

    @property
    def adapter_count(self) -> int:
        """Number of adapter components (0 or 1; never more)."""
        return 1 if self.adapter is not None else 0

    def assert_no_adapter_stack(self) -> None:
        """Raise if more than one adapter or a non-temporary adapter is present."""
        if self.adapter_count > 1:
            raise ValueError("adapter stack forbidden: at most one adapter component allowed")
        if self.adapter is not None and self.adapter.temporary is not True:
            raise ValueError("adapter must be temporary")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelPackage":
        """Build a ``ModelPackage`` from a validated package dict."""
        components = data["components"]

        def _comp(key: str) -> Component:
            d = components[key]
            if not isinstance(d, dict):
                raise ValueError(
                    f"component '{key}' must be an object, not a list "
                    f"(adapter stack forbidden)"
                )
            return Component(
                type=d["type"],
                name_or_path=d.get("name_or_path"),
                native=d.get("native"),
                temporary=d.get("temporary"),
                extra={k: v for k, v in d.items()
                       if k not in ("type", "name_or_path", "native", "temporary")},
            )

        adapter = _comp("adapter") if "adapter" in components else None
        pkg = cls(
            package_id=data["package_id"],
            mode=data["mode"],
            base=_comp("base"),
            causal_trunk=_comp("causal_trunk"),
            pdd_heads=_comp("pdd_heads"),
            control_branch=_comp("control_branch"),
            adapter=adapter,
            revisions=data.get("revisions", {}),
            hashes=data.get("hashes", {}),
            metrics=data.get("metrics", {}),
            dataset_revision=data["dataset_revision"],
            cost=data["cost"],
        )
        pkg.assert_no_adapter_stack()
        return pkg

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to a package dict conforming to model-package-v1."""
        components: Dict[str, Any] = {
            "base": _component_dict(self.base),
            "causal_trunk": _component_dict(self.causal_trunk),
            "pdd_heads": _component_dict(self.pdd_heads),
            "control_branch": _component_dict(self.control_branch),
        }
        if self.adapter is not None:
            components["adapter"] = _component_dict(self.adapter)
        return {
            "schema_version": "model-package-v1",
            "package_id": self.package_id,
            "mode": self.mode,
            "components": components,
            "revisions": self.revisions,
            "hashes": self.hashes,
            "metrics": self.metrics,
            "dataset_revision": self.dataset_revision,
            "cost": self.cost,
        }


def _component_dict(c: Component) -> Dict[str, Any]:
    d: Dict[str, Any] = {"type": c.type}
    if c.name_or_path is not None:
        d["name_or_path"] = c.name_or_path
    if c.native is not None:
        d["native"] = c.native
    if c.temporary is not None:
        d["temporary"] = c.temporary
    d.update(c.extra)
    return d
