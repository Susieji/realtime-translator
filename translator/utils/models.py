"""Helpers for loading FunASR/ModelScope models robustly (incl. offline)."""

import os


def resolve_local_model(model_id: str) -> str:
    """Return the local ModelScope cache dir for ``model_id`` if it's present.

    FunASR's AutoModel, given a bare model ID (optionally with model_revision),
    contacts ModelScope to verify the revision even when the model is already
    cached — which stalls for 60s+ and then fails when offline (no VPN). If the
    model is cached on disk, returning its directory makes AutoModel load
    straight from disk and skip the network check.

    Falls back to the original ID when the cache is absent, so first-time
    downloads still work normally.

    A model is considered cached only if its directory holds a ``model.pt``
    (the weights), guarding against half-populated cache folders.
    """
    home = os.path.expanduser("~")
    cache_dir = os.path.join(home, ".cache", "modelscope", "hub", "models",
                             *model_id.split("/"))
    if os.path.isdir(cache_dir) and os.path.exists(os.path.join(cache_dir, "model.pt")):
        return cache_dir
    return model_id


def is_local_path(model_ref: str) -> bool:
    """True if ``model_ref`` is a filesystem path (vs a bare ModelScope ID)."""
    return os.path.isdir(model_ref)
