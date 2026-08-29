#!/usr/bin/env python3
"""Download locked VBench metric assets with mirror-first fallback and checks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def validate_file(path: Path, item: dict[str, Any]) -> tuple[bool, str | None]:
    if not path.is_file() or path.stat().st_size != item["expected_bytes"]:
        return False, None
    digest = sha256(path)
    expected = item.get("sha256")
    prefix = item.get("sha256_prefix")
    if expected and digest != expected:
        raise ValueError(f"SHA256 mismatch for {path}: {digest} != {expected}")
    if prefix and not digest.startswith(prefix):
        raise ValueError(f"SHA256 prefix mismatch for {path}: {digest}")
    return True, digest


def aria2_download(url: str, part_path: Path, connections: int) -> bool:
    part_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "aria2c",
        "--allow-overwrite=false",
        "--auto-file-renaming=false",
        "--check-certificate=true",
        "--connect-timeout=30",
        "--continue=true",
        "--file-allocation=none",
        f"--max-connection-per-server={connections}",
        "--max-tries=5",
        "--min-split-size=8M",
        f"--split={connections}",
        "--summary-interval=10",
        "--timeout=60",
        "--retry-wait=5",
        f"--dir={part_path.parent}",
        f"--out={part_path.name}",
        url,
    ]
    print(f"DOWNLOAD {url}", flush=True)
    return subprocess.run(command, check=False).returncode == 0


def download_file(
    output_dir: Path, item: dict[str, Any], connections: int
) -> dict[str, Any]:
    target = output_dir / item["target"]
    valid, digest = validate_file(target, item)
    if valid:
        print(f"SKIP verified {item['name']}: {target}", flush=True)
        return {
            "name": item["name"],
            "target": str(target),
            "bytes": target.stat().st_size,
            "sha256": digest,
            "status": "verified_existing",
        }
    if target.exists():
        raise FileExistsError(f"Existing target is invalid; refusing overwrite: {target}")

    part_paths: list[Path] = []
    used_source: dict[str, str] | None = None
    selected_part_path: Path | None = None
    for source_index, source in enumerate(item["urls"], start=1):
        part_path = Path(str(target) + f".source{source_index}.part")
        part_paths.append(part_path)
        valid_part, _ = validate_file(part_path, item)
        if valid_part:
            print(f"RESUME verified {source['kind']} payload: {part_path}", flush=True)
            used_source = source
            selected_part_path = part_path
            break
        if aria2_download(source["url"], part_path, connections):
            if part_path.stat().st_size == item["expected_bytes"]:
                used_source = source
                selected_part_path = part_path
                break
        print(f"SOURCE FAILED {source['kind']}: {source['url']}", flush=True)
    if used_source is None or selected_part_path is None:
        raise RuntimeError(f"All sources failed for {item['name']}")

    valid, digest = validate_file(selected_part_path, item)
    if not valid:
        raise ValueError(f"Downloaded file has an invalid size: {selected_part_path}")
    os.replace(selected_part_path, target)
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)
        Path(str(part_path) + ".aria2").unlink(missing_ok=True)
    print(f"VERIFIED {item['name']} {target.stat().st_size} bytes", flush=True)
    return {
        "name": item["name"],
        "target": str(target),
        "bytes": target.stat().st_size,
        "sha256": digest,
        "status": "downloaded",
        "source_kind": used_source["kind"],
        "source_url": used_source["url"],
    }


def archive_is_complete(output_dir: Path, item: dict[str, Any]) -> bool:
    return all(
        (output_dir / item["extract_root"] / name).is_file()
        and (output_dir / item["extract_root"] / name).stat().st_size == size
        for name, size in item["expected_members"].items()
    )


def download_archive(
    output_dir: Path, item: dict[str, Any], connections: int
) -> dict[str, Any]:
    if archive_is_complete(output_dir, item):
        print(f"SKIP verified extracted archive: {item['name']}", flush=True)
        return {
            "name": item["name"],
            "status": "verified_existing",
            "extracted_bytes": sum(item["expected_members"].values()),
        }

    archive = output_dir / item["archive_target"]
    part_paths: list[Path] = []
    used_source: dict[str, str] | None = None
    selected_part_path: Path | None = None
    for source_index, source in enumerate(item["urls"], start=1):
        part_path = Path(str(archive) + f".source{source_index}.part")
        part_paths.append(part_path)
        if aria2_download(source["url"], part_path, connections):
            if part_path.stat().st_size == item["expected_download_bytes"]:
                used_source = source
                selected_part_path = part_path
                break
    if used_source is None or selected_part_path is None:
        raise RuntimeError(f"All sources failed for {item['name']}")
    digest = sha256(selected_part_path)

    extract_root = output_dir / item["extract_root"]
    extract_root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(selected_part_path) as archive_file:
        names = {entry.filename for entry in archive_file.infolist()}
        if not set(item["expected_members"]).issubset(names):
            raise ValueError(
                f"Archive members do not match the lock: {selected_part_path}"
            )
        for entry in archive_file.infolist():
            destination = (extract_root / entry.filename).resolve()
            if not destination.is_relative_to(extract_root.resolve()):
                raise ValueError(f"Unsafe archive member: {entry.filename}")
        archive_file.extractall(extract_root)
    if not archive_is_complete(output_dir, item):
        raise ValueError(f"Extracted archive validation failed: {item['name']}")
    for part_path in part_paths:
        part_path.unlink(missing_ok=True)
        Path(str(part_path) + ".aria2").unlink(missing_ok=True)
    return {
        "name": item["name"],
        "status": "downloaded_and_extracted",
        "download_bytes": item["expected_download_bytes"],
        "download_sha256": digest,
        "extracted_bytes": sum(item["expected_members"].values()),
        "source_kind": used_source["kind"],
        "source_url": used_source["url"],
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def clone_repository(output_dir: Path, item: dict[str, Any]) -> dict[str, Any]:
    target = output_dir / item["target"]
    if target.is_dir():
        commit = subprocess.check_output(
            ["git", "-C", str(target), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != item["commit"]:
            raise ValueError(f"Unexpected commit in {target}: {commit}")
        return {
            "name": item["name"],
            "status": "verified_existing",
            "target": str(target),
            "commit": commit,
            "bytes": directory_bytes(target),
        }
    target.parent.mkdir(parents=True, exist_ok=True)
    for source_index, source in enumerate(item["urls"], start=1):
        temporary = target.parent / f".{target.name}.clone_{source_index}"
        if temporary.exists():
            shutil.rmtree(temporary)
        command = [
            "git",
            "clone",
            "--depth",
            "1",
            "--filter=blob:none",
            source["url"],
            str(temporary),
        ]
        print(f"CLONE {source['url']}", flush=True)
        if subprocess.run(command, check=False).returncode != 0:
            if temporary.exists():
                shutil.rmtree(temporary)
            continue
        commit = subprocess.check_output(
            ["git", "-C", str(temporary), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != item["commit"]:
            shutil.rmtree(temporary)
            raise ValueError(f"Mirror returned unexpected DINO commit: {commit}")
        os.replace(temporary, target)
        return {
            "name": item["name"],
            "status": "cloned",
            "target": str(target),
            "commit": commit,
            "bytes": directory_bytes(target),
            "source_kind": source["kind"],
            "source_url": source["url"],
        }
    raise RuntimeError(f"All git sources failed for {item['name']}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sources", type=Path, default=script_dir / "download_sources.json"
    )
    parser.add_argument("--connections", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not shutil.which("aria2c"):
        raise RuntimeError("aria2c is required for resumable downloads")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources_path = args.sources.resolve()
    sources = json.loads(sources_path.read_text(encoding="utf-8"))
    manifest_path = output_dir / "download_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir),
        "sources": str(sources_path),
        "sources_sha256": sha256(sources_path),
        "items": [],
    }
    atomic_json(manifest_path, manifest)
    try:
        for item in sources["files"]:
            manifest["items"].append(
                download_file(output_dir, item, args.connections)
            )
            atomic_json(manifest_path, manifest)
        for item in sources["archives"]:
            manifest["items"].append(
                download_archive(output_dir, item, args.connections)
            )
            atomic_json(manifest_path, manifest)
        for item in sources["git_repositories"]:
            manifest["items"].append(clone_repository(output_dir, item))
            atomic_json(manifest_path, manifest)
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        atomic_json(manifest_path, manifest)
        raise
    manifest["status"] = "complete"
    manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["total_file_bytes"] = directory_bytes(output_dir)
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "items": len(manifest["items"]),
        "total_file_bytes": manifest["total_file_bytes"],
        "manifest": str(manifest_path),
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
