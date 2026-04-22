import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from iML.agents.debug_agent import DebugAgent
from iML.utils.execution_env import build_child_execution_env


class ExecutionEnvironmentTests(unittest.TestCase):
    def test_child_execution_env_prioritizes_current_interpreter_directory(self):
        env = build_child_execution_env({"PATH": "existing-path"})

        python_dir = str(Path(sys.executable).resolve().parent)

        self.assertEqual(env["PATH"].split(os.pathsep)[0], python_dir)
        self.assertEqual(env["IML_EXECUTABLE"], str(Path(sys.executable).resolve()))
        self.assertEqual(env["IML_EXEC_ENV_PREFIX"], sys.prefix)
        self.assertEqual(env["PYTHONNOUSERSITE"], "1")

    def test_child_execution_env_requires_virtualenv_when_running_inside_one(self):
        with patch.object(sys, "prefix", "/tmp/project/.venv"), patch.object(sys, "base_prefix", "/usr"):
            env = build_child_execution_env({"PATH": "existing-path"})

        self.assertEqual(env["VIRTUAL_ENV"], "/tmp/project/.venv")
        self.assertEqual(env["PIP_REQUIRE_VIRTUALENV"], "1")

    def test_debug_agent_ensure_package_uses_uv_for_current_interpreter(self):
        agent = object.__new__(DebugAgent)

        patched = agent._ensure_package_import("print('ready')", "lightgbm")

        self.assertIn('"uv", "pip", "install", "--python", sys.executable, "lightgbm"', patched)
        self.assertNotIn("sys.executable, '-m', 'pip'", patched)
        self.assertNotIn("pip install", patched)


if __name__ == "__main__":
    unittest.main()
