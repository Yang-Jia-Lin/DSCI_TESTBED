import unittest

from Src.Phase3_Runtime.Cloud.run_cloud import _validate_cloud_request
from Src.Phase3_Runtime.Edge.run_edge import _validate_edge_request
from Src.Phase3_Runtime.Shared import segment_worker
from Src.Phase3_Runtime.Shared.mnn_segment_worker import execute_mnn_range
from Src.Phase3_Runtime.Shared.pytorch_segment_worker import execute_pytorch_range


class FakeManifest:
    final_boundary_id = 3

    def __init__(self):
        self.edge_pairs = []
        self.cloud_ranges = []

    def validate_boundary_pair(self, first, second):
        self.edge_pairs.append((first, second))
        if not 0 <= first < second <= self.final_boundary_id:
            raise ValueError("invalid boundary pair")

    def validate_range(self, start, end):
        self.cloud_ranges.append((start, end))
        if not 0 <= start <= end <= self.final_boundary_id:
            raise ValueError("invalid range")


def runtime_state():
    return {
        "bundle_id": "bundle",
        "manifest_id": "manifest",
        "model_hash": "hash",
    }


class RuntimeValidationTests(unittest.TestCase):
    def test_edge_request_is_validated_before_worker_submit(self):
        manifest = FakeManifest()
        meta, b1, b2 = _validate_edge_request(
            {
                "bundle_id": "bundle",
                "manifest_id": "manifest",
                "model_hash": "hash",
                "boundary_id": 1,
                "meta": {"partition_boundary_2": 2},
            },
            runtime_state(),
            manifest,
        )

        self.assertEqual(meta["partition_boundary_2"], 2)
        self.assertEqual((b1, b2), (1, 2))
        self.assertEqual(manifest.edge_pairs, [(1, 2)])

    def test_edge_request_rejects_invalid_boundary_pair(self):
        with self.assertRaises(ValueError):
            _validate_edge_request(
                {
                    "bundle_id": "bundle",
                    "manifest_id": "manifest",
                    "model_hash": "hash",
                    "boundary_id": 2,
                    "meta": {"partition_boundary_2": 1},
                },
                runtime_state(),
                FakeManifest(),
            )

    def test_cloud_request_is_validated_before_worker_submit(self):
        manifest = FakeManifest()
        boundary_id = _validate_cloud_request(
            {
                "bundle_id": "bundle",
                "manifest_id": "manifest",
                "model_hash": "hash",
                "boundary_id": 2,
            },
            runtime_state(),
            manifest,
        )

        self.assertEqual(boundary_id, 2)
        self.assertEqual(manifest.cloud_ranges, [(2, 3)])

    def test_segment_worker_keeps_backend_entrypoints_separate(self):
        self.assertIs(segment_worker.execute_pytorch_range, execute_pytorch_range)
        self.assertIs(segment_worker.execute_mnn_range, execute_mnn_range)


if __name__ == "__main__":
    unittest.main()
