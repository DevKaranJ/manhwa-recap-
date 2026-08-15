from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from src.logging_setup import get_logger

log = get_logger("nove.config")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class ConfigError(RuntimeError):
    pass


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolve_api_key(cfg: dict[str, Any]) -> dict[str, Any]:
    env_name = cfg.get("llm", {}).get("api_key_env", "OPENROUTER_API_KEY")
    key = os.environ.get(env_name, "")
    cfg.setdefault("llm", {})["api_key"] = key
    if cfg.get("llm", {}).get("provider") not in ("openrouter", "zenmux"):
        cfg["llm"]["api_key"] = ""
    if cfg.get("llm", {}).get("provider") == "zenmux":
        cfg["llm"]["base_url"] = "https://zenmux.com/v1"
        cfg["llm"]["api_key_env"] = "ZENMUX_API_KEY"
    return cfg


def _ensure_workspace(cfg: dict[str, Any]) -> dict[str, Any]:
    for key in ("scripts_dir", "audio_dir", "images_dir", "output_dir"):
        rel = cfg["paths"].get(key, "")
        if rel:
            path = PROJECT_ROOT / rel
            path.mkdir(parents=True, exist_ok=True)
            cfg["paths"][key] = str(path)
    return cfg


def _check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise ConfigError(
            "ffmpeg binary not found on PATH. Install FFmpeg (https://ffmpeg.org) "
            "and ensure 'ffmpeg' is callable from your shell."
        )
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, text=True, timeout=15
        )
        if proc.returncode != 0:
            raise ConfigError(f"ffmpeg -version failed: {proc.stderr.strip()[:300]}")
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigError(f"ffmpeg could not be executed: {exc}") from exc


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise ConfigError(f"config file not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    _check_ffmpeg()
    cfg = _resolve_api_key(cfg)
    cfg = _ensure_workspace(cfg)
    return cfg
