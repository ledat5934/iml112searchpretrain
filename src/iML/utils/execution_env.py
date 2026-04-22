import os
import sys
from pathlib import Path
from typing import Mapping


def build_child_execution_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """Build an environment that keeps child code pinned to this interpreter."""
    env = dict(base_env or os.environ)

    python_path = Path(sys.executable).resolve()
    python_dir = str(python_path.parent)
    existing_path = env.get("PATH", "")
    env["PATH"] = python_dir if not existing_path else python_dir + os.pathsep + existing_path
    env["PYTHONNOUSERSITE"] = "1"
    env["IML_EXECUTABLE"] = str(python_path)
    env["IML_EXEC_ENV_PREFIX"] = sys.prefix

    if getattr(sys, "base_prefix", sys.prefix) != sys.prefix:
        env["VIRTUAL_ENV"] = sys.prefix
        env["PIP_REQUIRE_VIRTUALENV"] = "1"

    return env
