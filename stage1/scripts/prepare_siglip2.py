#!/usr/bin/env python
"""Download SigLIP2 weights explicitly and extract the vision tower for offline training.

Training never downloads anything. Run this once, review ``provenance.json``, and
point ``model.encoder.weights_path`` at the extracted ``vision_model.safetensors``.

    python scripts/prepare_siglip2.py --repo google/siglip2-large-patch16-256

The full ``model.safetensors`` (vision + text towers, fp32) is kept next to the
extracted file unless ``--remove-full`` is given. The attention-pooling head
(``vision_model.head.*``) is not extracted: reconstruction uses patch tokens only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import requests

FILES = ("config.json", "preprocessor_config.json", "model.safetensors")
VISION_PREFIX = "vision_model."
HEAD_PREFIX = "vision_model.head."


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _headers() -> dict:
    token = os.environ.get("HF_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def download(url: str, target: Path, expected_size: int | None = None) -> dict:
    """Stream to ``target`` with HTTP range resume. Returns size and linked ETag."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = _headers()
    if offset:
        headers["Range"] = f"bytes={offset}-"
    with requests.get(url, headers=headers, stream=True, allow_redirects=True, timeout=60) as response:
        if offset and response.status_code == 200:
            offset = 0
        elif offset and response.status_code != 206:
            response.raise_for_status()
        response.raise_for_status()
        length = response.headers.get("Content-Length")
        total = (int(length) + offset) if length else None
        mode = "ab" if offset else "wb"
        done = offset
        started = time.monotonic()
        with partial.open(mode) as stream:
            for chunk in response.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                stream.write(chunk)
                done += len(chunk)
                if total and (done == total or int(time.monotonic() - started) % 15 == 0):
                    print(f"\r{target.name}: {done / 1e9:.2f} / {total / 1e9:.2f} GB", end="", file=sys.stderr)
            stream.flush()
            os.fsync(stream.fileno())
    print(file=sys.stderr)
    size = partial.stat().st_size
    if total is not None and size != total:
        raise RuntimeError(f"Incomplete download for {target.name}: {size} of {total} bytes")
    if expected_size is not None and size != expected_size:
        raise RuntimeError(f"{target.name}: size {size} differs from the repository listing {expected_size}")
    partial.replace(target)
    return {"size": size}


def repository_listing(repo: str, revision: str) -> tuple[dict, dict]:
    """Commit metadata plus per-file size and LFS SHA256 (the Xet CDN ETag is not a SHA256)."""
    base = f"https://huggingface.co/api/models/{repo}"
    info = requests.get(f"{base}/revision/{revision}", headers=_headers(), timeout=60)
    info.raise_for_status()
    tree = requests.get(f"{base}/tree/{revision}", headers=_headers(), timeout=60)
    tree.raise_for_status()
    entries = {entry["path"]: {"size": entry.get("size"), "sha256": (entry.get("lfs") or {}).get("oid")}
               for entry in tree.json() if entry.get("type") == "file"}
    return info.json(), entries


def extract_vision(full: Path, target: Path) -> dict:
    from safetensors import safe_open
    from safetensors.torch import save_file

    tensors, dropped = {}, []
    with safe_open(str(full), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.startswith(VISION_PREFIX):
                continue
            if key.startswith(HEAD_PREFIX):
                dropped.append(key)
                continue
            tensors[key] = handle.get_tensor(key).contiguous()
    if not tensors:
        raise ValueError(f"No '{VISION_PREFIX}*' tensors found in {full}")
    parameters = sum(int(t.numel()) for t in tensors.values())
    save_file(tensors, str(target), metadata={"format": "pt", "source": full.name,
                                               "contents": "SigLIP2 vision tower without pooling head"})
    return {"tensor_count": len(tensors), "parameter_count": parameters, "dropped_head_tensors": dropped}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", default="google/siglip2-large-patch16-256")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--output", default="weights/siglip2", help="Directory receiving <repo-name>/")
    parser.add_argument("--remove-full", action="store_true", help="Delete model.safetensors after extraction")
    args = parser.parse_args()

    listing, entries = repository_listing(args.repo, args.revision)
    out = Path(args.output) / args.repo.split("/")[-1]
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in FILES:
        if name not in entries:
            raise FileNotFoundError(f"{args.repo} has no {name}")
        expected_size, expected_sha = entries[name]["size"], entries[name]["sha256"]
        target = out / name
        if target.exists() and target.stat().st_size == expected_size:
            print(f"{name}: present, skipping download", file=sys.stderr)
            record = {"size": target.stat().st_size}
        else:
            record = download(f"https://huggingface.co/{args.repo}/resolve/{args.revision}/{name}",
                              target, expected_size=expected_size)
        record["sha256"] = sha256_of(target)
        record["repository_sha256"] = expected_sha
        if expected_sha and expected_sha != record["sha256"]:
            raise RuntimeError(f"{name}: SHA256 {record['sha256']} differs from the repository LFS hash {expected_sha}")
        records[name] = record

    config = json.loads((out / "config.json").read_text())
    vision_config = config.get("vision_config") or config
    extracted = extract_vision(out / "model.safetensors", out / "vision_model.safetensors")
    extracted["sha256"] = sha256_of(out / "vision_model.safetensors")
    extracted["size"] = (out / "vision_model.safetensors").stat().st_size
    provenance = {
        "repository": args.repo, "revision": args.revision, "commit": listing.get("sha"),
        "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "license": "apache-2.0 (see the repository README)",
        "files": records, "vision_model.safetensors": extracted, "vision_config": vision_config,
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2))
    if args.remove_full:
        (out / "model.safetensors").unlink()
    print(json.dumps({"output": str(out), "vision_tensors": extracted["tensor_count"],
                      "vision_parameters": extracted["parameter_count"],
                      "vision_sha256": extracted["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
