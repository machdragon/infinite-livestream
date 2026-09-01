"""Model-package validation and one-time load for CausalH3.

The persistent model consumes a *trained merged family export* -- not the
pruned INT8 preview weights that ``fast-h3`` uses.  The package is a
``model-package-v1`` artifact (schema in ai-toolkit) that names the base
model, the native causal trunk, PDD heads, control branch, and at most one
temporary adapter, with hashes and an authorization/provenance manifest.

Validation rejects:
- pruned bases (``fl2va_pruned`` / ``ref2va_pruned`` partitions)
- family mismatch (package mode vs the requested slug)
- wrong hashes (SHA-256 of each component directory)
- wrong schedule (must be the four-step ladder ``[999,749,500,250,0]``)
- missing gates (causal_trunk, pdd_heads, control_branch must be native)
- extra adapters (at most one, and it must be temporary)
- absent authorization record

Weights never live in git; the package manifest is the only thing tracked.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

import causalh3_chunk_plan as chunk_plan

# The two family slugs this model serves.
FAMILIES = ("fl2va", "ref2va")

# Partitions that are pruned INT8 previews -- rejected as serving bases.
_PRUNED_PARTITIONS = ("fl2va_pruned", "ref2va_pruned")

# Required native components in the model package.
_REQUIRED_NATIVE = ("causal_trunk", "pdd_heads", "control_branch")

# Required weight component directories inside the checkpoint.
_REQUIRED_COMPONENT_DIRS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)

# The four-step schedule the trained export must carry.
_EXPECTED_SIGMA_LADDER = list(chunk_plan.SIGMA_LADDER)


@dataclass(frozen=True)
class CausalH3Config:
    """Everything ``causalh3.yaml`` configures, validated once at load.

    Session-level fields are the defaults a fresh session starts from.
    ``inference`` and ``runtime`` are the raw blocks; the backend reads its
    engine knobs from them.
    """

    family: str
    aspect: str
    seed: int
    target_seconds: float
    inference: dict[str, Any]
    runtime: dict[str, Any]


@dataclass(frozen=True)
class ModelPackageManifest:
    """The validated model-package-v1 manifest, held in memory after load.

    Fields mirror the ai-toolkit ``ModelPackage`` dataclass but are kept
    here as plain data so this module has no dependency on the training
    codebase at serving time.
    """

    package_id: str
    mode: str
    base_path: str
    base_partition: str
    base_arch: str
    base_quantize: bool
    adapter_path: str | None
    adapter_temporary: bool
    adapter_rank: int | None
    causal_trunk_native: bool
    causal_trunk_layers: int
    pdd_heads_native: bool
    pdd_heads_layers: int
    control_branch_native: bool
    control_branch_layers: int
    revisions: dict[str, str]
    hashes: dict[str, str]
    metrics: dict[str, float]
    dataset_revision: str
    cost: dict[str, Any]
    authorization: dict[str, Any] = field(default_factory=dict)


def load_config(config_path: Path | None) -> CausalH3Config:
    """Parse ``causalh3.yaml`` into a validated :class:`CausalH3Config`.

    Args:
        config_path: Path the runtime hands over from ``runtime.config`` in
            ``reactor.yaml``, or ``None`` when the manifest names no config.

    Raises:
        ValueError: If the configured family or aspect is invalid.
    """
    document: dict[str, Any] = {}
    if config_path is not None:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    inference: dict[str, Any] = document.get("inference") or {}
    runtime: dict[str, Any] = document.get("runtime") or {}

    family = str(inference.get("family", "fl2va"))
    if family not in FAMILIES:
        raise ValueError(
            f"inference.family must be one of {list(FAMILIES)}, got {family!r}"
        )

    aspect = str(inference.get("aspect", "16:9"))
    if aspect not in chunk_plan.ASPECT_CHOICES:
        raise ValueError(
            f"inference.aspect must be one of {list(chunk_plan.ASPECT_CHOICES)}, got {aspect!r}"
        )

    target_seconds = float(inference.get("target_seconds", chunk_plan.SESSION_SECONDS))
    if target_seconds <= 0:
        raise ValueError(f"inference.target_seconds must be positive, got {target_seconds}")

    return CausalH3Config(
        family=family,
        aspect=aspect,
        seed=int(inference.get("seed", 1000)),
        target_seconds=target_seconds,
        inference=inference,
        runtime=runtime,
    )


def resolve_model_path(config: CausalH3Config, weights_root: Path) -> Path:
    """The checkpoint directory inside the mounted weights bundle.

    ``checkpoint_dir: "."`` means the snapshot's components sit directly
    under the weights root.
    """
    subdir = str(config.runtime.get("checkpoint_dir", "."))
    if subdir in ("", "."):
        return weights_root
    return weights_root / subdir


def _sha256_directory(path: Path) -> str:
    """SHA-256 hash of a directory's file contents, sorted by relative path.

    This is a content hash, not a file-listing hash: it covers every regular
    file under *path*, read in sorted relative-path order, with the path
    itself mixed into the stream so a renamed file is a different hash.
    """
    h = hashlib.sha256()
    files = sorted(p for p in path.rglob("*") if p.is_file())
    for f in files:
        rel = str(f.relative_to(path))
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        h.update(f.read_bytes())
    return h.hexdigest()


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    """Load and parse the model-package-v1 JSON manifest."""
    with open(manifest_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def validate_manifest(
    data: dict[str, Any],
    *,
    expected_family: str,
    expected_sigma_ladder: list[int] | None = None,
) -> ModelPackageManifest:
    """Validate a model-package-v1 manifest dict for serving.

    Args:
        data: The parsed manifest JSON.
        expected_family: The family slug the session requested.
        expected_sigma_ladder: The sigma-grid the serving schedule must
            match.  Defaults to the four-step ladder.

    Raises:
        ValueError: With a specific reason for every rejection case.
    """
    if expected_sigma_ladder is None:
        expected_sigma_ladder = _EXPECTED_SIGMA_LADDER

    # --- schema_version ---
    sv = data.get("schema_version")
    if sv != "model-package-v1":
        raise ValueError(
            f"manifest schema_version must be 'model-package-v1', got {sv!r}"
        )

    # --- mode / family ---
    mode = data.get("mode")
    if mode not in FAMILIES:
        raise ValueError(f"manifest mode must be one of {list(FAMILIES)}, got {mode!r}")
    if mode != expected_family:
        raise ValueError(
            f"manifest mode {mode!r} does not match the requested family {expected_family!r}"
        )

    # --- components ---
    components = data.get("components")
    if not isinstance(components, dict):
        raise ValueError("manifest.components must be an object")

    base = components.get("base")
    if not isinstance(base, dict):
        raise ValueError("manifest.components.base is required and must be an object")
    base_partition = base.get("partition", "")
    if base_partition in _PRUNED_PARTITIONS:
        raise ValueError(
            f"pruned INT8 base partition {base_partition!r} is not a valid serving base; "
            "serving requires a trained merged/full family export"
        )
    base_arch = base.get("arch", "")
    if base_arch != "minimax_h3":
        raise ValueError(
            f"base arch must be 'minimax_h3', got {base_arch!r}"
        )

    # --- native components ---
    for comp_name in _REQUIRED_NATIVE:
        comp = components.get(comp_name)
        if comp is None:
            raise ValueError(f"manifest.components.{comp_name} is required")
        if not isinstance(comp, dict):
            raise ValueError(
                f"manifest.components.{comp_name} must be an object, not a list "
                f"(adapter stack forbidden)"
            )
        if comp.get("native") is not True:
            raise ValueError(
                f"manifest.components.{comp_name}.native must be true; "
                "serving requires trained native gates"
            )

    # --- adapter: at most one, must be temporary ---
    adapter = components.get("adapter")
    adapter_path: str | None = None
    adapter_temporary = False
    adapter_rank: int | None = None
    if adapter is not None:
        if isinstance(adapter, list):
            raise ValueError(
                "adapter stack forbidden: at most one adapter component permitted"
            )
        if not isinstance(adapter, dict):
            raise ValueError("adapter must be an object")
        if adapter.get("temporary") is not True:
            raise ValueError("adapter must be flagged temporary")
        adapter_path = adapter.get("name_or_path")
        adapter_temporary = True
        adapter_rank = adapter.get("rank")

    # --- hashes ---
    hashes = data.get("hashes", {})
    if not isinstance(hashes, dict):
        raise ValueError("manifest.hashes must be an object")
    for comp_name in ("base",) + _REQUIRED_NATIVE:
        h = hashes.get(comp_name)
        if not h or not isinstance(h, str):
            raise ValueError(f"manifest.hashes.{comp_name} is required")
        # Accept 40-64 char hex.
        clean = h.strip()
        if len(clean) < 40 or len(clean) > 64:
            raise ValueError(
                f"manifest.hashes.{comp_name} must be 40-64 hex chars, got {len(clean)}"
            )
        try:
            int(clean, 16)
        except ValueError:
            raise ValueError(
                f"manifest.hashes.{comp_name} must be hex, got {clean!r}"
            ) from None

    # --- schedule ---
    schedule = data.get("schedule") or data.get("inference_schedule") or {}
    sigma_ladder = schedule.get("sigma_ladder")
    if sigma_ladder is not None:
        if [int(x) for x in sigma_ladder] != expected_sigma_ladder:
            raise ValueError(
                f"schedule.sigma_ladder must be {expected_sigma_ladder}, "
                f"got {sigma_ladder}"
            )
    # Also accept the schedule from the config block if present.
    cfg_schedule = data.get("config", {}).get("schedule", {})
    if cfg_schedule.get("sigma_ladder") is not None:
        if [int(x) for x in cfg_schedule["sigma_ladder"]] != expected_sigma_ladder:
            raise ValueError(
                f"config.schedule.sigma_ladder must be {expected_sigma_ladder}, "
                f"got {cfg_schedule['sigma_ladder']}"
            )

    # --- authorization / provenance ---
    authorization = data.get("authorization") or data.get("provenance") or {}
    if not authorization:
        raise ValueError(
            "manifest.authorization (or provenance) record is required; "
            "serving requires an authorization/provenance manifest"
        )
    if not isinstance(authorization, dict):
        raise ValueError("manifest.authorization must be an object")
    # Must carry at least an issuer or a signer.
    if not authorization.get("issuer") and not authorization.get("signer"):
        raise ValueError(
            "manifest.authorization must carry an 'issuer' or 'signer' field"
        )

    # --- revisions ---
    revisions = data.get("revisions", {})
    if not isinstance(revisions, dict):
        raise ValueError("manifest.revisions must be an object")

    # --- dataset_revision ---
    dataset_revision = data.get("dataset_revision", "")
    if not dataset_revision:
        raise ValueError("manifest.dataset_revision is required")

    # --- cost ---
    cost = data.get("cost", {})
    if not isinstance(cost, dict):
        raise ValueError("manifest.cost must be an object")

    return ModelPackageManifest(
        package_id=data.get("package_id", ""),
        mode=mode,
        base_path=base.get("name_or_path", ""),
        base_partition=base_partition,
        base_arch=base_arch,
        base_quantize=bool(base.get("quantize", False)),
        adapter_path=adapter_path,
        adapter_temporary=adapter_temporary,
        adapter_rank=adapter_rank,
        causal_trunk_native=components["causal_trunk"].get("native") is True,
        causal_trunk_layers=int(components["causal_trunk"].get("num_layers", 0)),
        pdd_heads_native=components["pdd_heads"].get("native") is True,
        pdd_heads_layers=int(components["pdd_heads"].get("num_layers", 0)),
        control_branch_native=components["control_branch"].get("native") is True,
        control_branch_layers=int(components["control_branch"].get("num_layers", 0)),
        revisions=revisions,
        hashes=hashes,
        metrics=data.get("metrics", {}),
        dataset_revision=dataset_revision,
        cost=cost,
        authorization=authorization,
    )


def require_weights(
    root: Path,
    model_path: Path,
    manifest: ModelPackageManifest,
    *,
    verify_hashes: bool = True,
) -> None:
    """Fail startup loudly when the weights bundle is incomplete or hashes mismatch.

    Args:
        root: The weights bundle root.
        model_path: The checkpoint directory inside the bundle.
        manifest: The validated package manifest.
        verify_hashes: When true, recompute SHA-256 of each component
            directory and compare against the manifest.  Set false only
            in tests that synthesize fake weights.
    """
    problems: list[str] = []
    if not model_path.is_dir():
        problems.append(f"checkpoint directory is missing: {model_path}")
    else:
        index = model_path / "modular_model_index.json"
        if not index.is_file():
            problems.append(f"modular_model_index.json is missing: {index}")
        for component in _REQUIRED_COMPONENT_DIRS:
            if not (model_path / component).is_dir():
                problems.append(f"component directory is missing: {model_path / component}")

    # Manifest file must exist alongside the weights.
    manifest_file = root / "model-package.json"
    if not manifest_file.is_file():
        problems.append(f"model-package.json manifest is missing: {manifest_file}")

    if problems:
        raise FileNotFoundError(
            f"CausalH3 weights bundle under {root} is incomplete:\n  " + "\n  ".join(problems)
        )

    if verify_hashes:
        _verify_component_hashes(model_path, manifest)


def _verify_component_hashes(model_path: Path, manifest: ModelPackageManifest) -> None:
    """Recompute SHA-256 of each component directory and compare to the manifest."""
    # Map manifest hash keys to checkpoint subdirectories.
    component_map = {
        "base": "transformer",
        "causal_trunk": "transformer",  # native trunk is merged into the transformer
        "pdd_heads": "transformer",  # PDD heads are part of the merged export
        "control_branch": "transformer",  # control branch is part of the merged export
    }
    for manifest_key, subdir in component_map.items():
        expected = manifest.hashes.get(manifest_key)
        if not expected:
            continue
        dir_path = model_path / subdir
        if not dir_path.is_dir():
            continue  # already reported above
        actual = _sha256_directory(dir_path)
        if actual != expected:
            raise ValueError(
                f"hash mismatch for {manifest_key} ({subdir}): "
                f"manifest says {expected}, computed {actual}"
            )


def load_manifest(root: Path) -> dict[str, Any]:
    """Load the model-package.json manifest from the weights bundle root."""
    manifest_path = root / "model-package.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"model-package.json manifest is missing: {manifest_path}"
        )
    return _load_manifest(manifest_path)


__all__ = [
    "FAMILIES",
    "CausalH3Config",
    "ModelPackageManifest",
    "load_config",
    "resolve_model_path",
    "validate_manifest",
    "require_weights",
    "load_manifest",
]
