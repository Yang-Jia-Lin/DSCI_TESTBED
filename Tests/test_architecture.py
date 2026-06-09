from __future__ import annotations

import ast
import importlib
import unittest

from Src.Shared.Config.paths import (
    DEVICE_RESULTS_DIR,
    ModelArtifactPaths,
    PROJECT_ROOT,
    RESNET50_PATHS,
    SOLUTION_CACHE_DIR,
)
from Src.Shared.Config.model_config import RESNET50


SRC_ROOT = PROJECT_ROOT / "Src"


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_phase_import_boundaries(self):
        rules = {
            "Shared": ("Src.Phase1_Offline", "Src.Phase2_Scheduler", "Src.Phase3_Runtime", "Scripts"),
            "Phase1_Offline": ("Src.Phase2_Scheduler", "Src.Phase3_Runtime", "Scripts"),
            "Phase2_Scheduler": ("Src.Phase1_Offline", "Src.Phase3_Runtime", "Scripts"),
            "Phase3_Runtime": ("Src.Phase1_Offline", "Src.Phase2_Scheduler", "Scripts"),
        }
        violations = []
        for area, forbidden_prefixes in rules.items():
            for path in (SRC_ROOT / area).rglob("*.py"):
                tree = ast.parse(
                    path.read_text(encoding="utf-8-sig"), filename=str(path)
                )
                for node in ast.walk(tree):
                    imported = []
                    if isinstance(node, ast.Import):
                        imported = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported = [node.module]
                    for module in imported:
                        if module.startswith(forbidden_prefixes):
                            lineno = getattr(node, "lineno", 0)
                            violations.append(
                                f"{path.relative_to(PROJECT_ROOT)}:{lineno} imports {module}"
                            )
        self.assertEqual(violations, [], "\n".join(violations))

    def test_public_modules_import(self):
        modules = (
            "Src.Phase1_Offline.Training.train_model",
            "Src.Phase1_Offline.Profiling.validate_compute_profile",
            "Src.Phase1_Offline.LookupTables.audit_offline_tables",
            "Src.Phase2_Scheduler.paras",
            "Src.Phase2_Scheduler.Service.api_server",
            "Src.Phase2_Scheduler.Objective.compute_latency",
            "Src.Phase2_Scheduler.Optimizer.DSCI.agent",
            "Src.Phase3_Runtime.Device.run_device",
            "Src.Phase3_Runtime.Edge.run_edge",
            "Src.Phase3_Runtime.Cloud.run_cloud",
            "Src.Shared.Config.model_config",
            "Src.Shared.Models.ModelNet.Resnet50",
            "Src.Shared.Profiles.compute_profile",
            "Src.Shared.Data.dataloader",
        )
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)

    def test_runtime_output_paths(self):
        self.assertEqual(SOLUTION_CACHE_DIR, PROJECT_ROOT / "Data/Runtime/SolutionCache")
        self.assertEqual(DEVICE_RESULTS_DIR, PROJECT_ROOT / "Data/Runtime/DeviceResults")

    def test_model_metadata_does_not_own_paths(self):
        self.assertNotIn("resolve_weight_path", dir(RESNET50))
        self.assertIsInstance(RESNET50_PATHS, ModelArtifactPaths)
        self.assertEqual(RESNET50_PATHS.model, RESNET50)


if __name__ == "__main__":
    unittest.main()
