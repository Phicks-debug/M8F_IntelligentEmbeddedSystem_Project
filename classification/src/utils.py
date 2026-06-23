"""Utility helpers for runtime paths, checkpoints, and device selection."""

import hashlib
import posixpath
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import torch

REMOTE_SCHEMES = {"s3", "r2", "gs", "gcs", "az", "abfs", "adl"}


def get_device(preferred: str = "auto") -> str:
    """Select best available device."""
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def checkpoint_to_bytes(path: Path) -> bytes:
    """Read a checkpoint file into bytes for Metaflow artifact passing."""
    return path.read_bytes()


def bytes_to_checkpoint(data: bytes, path: Path) -> None:
    """Write checkpoint bytes to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def is_remote_uri(value: str | Path | None) -> bool:
    """Return True for object-store URIs handled through fsspec."""
    if value is None:
        return False
    scheme = urlparse(str(value)).scheme.lower()
    return scheme in REMOTE_SCHEMES


def _storage_options(cfg: dict[str, Any]) -> Mapping[str, Any]:
    paths = cfg.get("paths", {})
    options = paths.get("storage_options", {})
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise TypeError("paths.storage_options must be a mapping when provided")
    return options


def _cache_path(cache_root: Path, kind: str, uri: str) -> Path:
    digest = hashlib.sha1(uri.encode("utf-8")).hexdigest()[:12]
    parsed = urlparse(uri)
    name = Path(parsed.path.rstrip("/")).name or parsed.netloc or digest
    return cache_root / kind / f"{name}-{digest}"


def _url_to_fs(uri: str, storage_options: Mapping[str, Any]):
    try:
        import fsspec
    except ImportError as exc:
        raise RuntimeError(
            "Remote paths require fsspec plus the matching filesystem package "
            "(for S3/R2 install s3fs)."
        ) from exc

    parsed = urlparse(uri)
    if parsed.scheme == "r2":
        remote_path = posixpath.join(parsed.netloc, parsed.path.lstrip("/"))
        return fsspec.filesystem("s3", **dict(storage_options)), remote_path
    return fsspec.core.url_to_fs(uri, **dict(storage_options))


def download_directory(
    remote_uri: str,
    local_dir: Path,
    storage_options: Mapping[str, Any] | None = None,
    *,
    require_non_empty: bool = True,
) -> Path:
    """Download an object-store directory into ``local_dir``."""
    options = storage_options or {}
    fs, remote_path = _url_to_fs(remote_uri, options)
    files = [path for path in fs.find(remote_path) if not fs.isdir(path)]
    if require_non_empty and not files:
        raise FileNotFoundError(f"No files found at remote path: {remote_uri}")

    local_dir.mkdir(parents=True, exist_ok=True)
    for remote_file in files:
        rel = posixpath.relpath(remote_file, remote_path)
        local_file = local_dir / rel
        local_file.parent.mkdir(parents=True, exist_ok=True)
        fs.get(remote_file, str(local_file))
    return local_dir


def sync_directory(
    local_dir: Path,
    remote_uri: str,
    storage_options: Mapping[str, Any] | None = None,
) -> None:
    """Upload the contents of ``local_dir`` to an object-store directory."""
    if not local_dir.exists():
        return

    options = storage_options or {}
    fs, remote_path = _url_to_fs(remote_uri, options)
    fs.mkdirs(remote_path, exist_ok=True)

    for local_file in local_dir.rglob("*"):
        if not local_file.is_file():
            continue
        rel = local_file.relative_to(local_dir).as_posix()
        remote_file = posixpath.join(remote_path, rel)
        fs.mkdirs(posixpath.dirname(remote_file), exist_ok=True)
        fs.put(str(local_file), remote_file)


def prepare_runtime_paths(cfg: dict[str, Any]) -> None:
    """Stage remote data and map remote output dirs to local cache dirs.

    Training code works on normal filesystem paths. This helper mutates the
    runtime config so ``data.dir``, ``paths.checkpoint_dir``, and
    ``paths.export_dir`` point at local cache folders while preserving
    ``remote_*`` values for upload after each stage.
    """
    paths = cfg.setdefault("paths", {})
    cache_root = Path(
        paths.get("local_cache_dir", ".cache/classification")
    ).expanduser()
    options = _storage_options(cfg)

    data_cfg = cfg.get("data", {})
    data_dir = str(data_cfg.get("dir", ""))
    if is_remote_uri(data_dir):
        local_data = _cache_path(cache_root, "data", data_dir)
        print(f"Staging remote data {data_dir} -> {local_data}")
        download_directory(data_dir, local_data, options, require_non_empty=True)
        data_cfg["remote_dir"] = data_dir
        data_cfg["dir"] = str(local_data)

    for key in ("checkpoint_dir", "export_dir"):
        value = str(paths.get(key, ""))
        if not is_remote_uri(value):
            continue

        local_dir = _cache_path(cache_root, key, value)
        paths[f"remote_{key}"] = value
        paths[key] = str(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using local {key} cache {local_dir} for remote {value}")
        download_directory(value, local_dir, options, require_non_empty=False)


def sync_runtime_outputs(cfg: dict[str, Any], *path_keys: str) -> None:
    """Sync configured local output dirs back to their remote locations."""
    paths = cfg.get("paths", {})
    options = _storage_options(cfg)
    for key in path_keys:
        remote = paths.get(f"remote_{key}")
        if not remote:
            continue
        local = Path(paths[key])
        print(f"Syncing {local} -> {remote}")
        sync_directory(local, str(remote), options)
