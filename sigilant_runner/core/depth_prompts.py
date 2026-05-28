from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

_DEFAULT_FILES: Dict[str, str] = {
    "8k": "hard_quality_8k_prompt.txt",
    "14k": "hard_quality_14k_prompt.txt",
    "28k": "hard_quality_28k_prompt.txt",
}


def _packaged_prompt_path(filename: str) -> Path:
    return Path(__file__).resolve().parent.parent / "data" / "depth_prompts" / filename


def resolve_depth_prompt_path(raw_path: str, depth_label: str) -> Tuple[Path, str]:
    """Resolve a depth prompt path with packaged fallback for pip-only usage.

    Returns:
      (path, source), where source is one of:
      - "user_path"
      - "packaged_default"
    """
    user_path = Path(raw_path)
    if user_path.exists():
        return user_path, "user_path"

    fallback_name = _DEFAULT_FILES.get(depth_label)
    if fallback_name:
        pkg_path = _packaged_prompt_path(fallback_name)
        if pkg_path.exists():
            return pkg_path, "packaged_default"

    return user_path, "user_path"
