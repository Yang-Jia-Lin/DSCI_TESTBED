# Data layout

Only the following top-level directories are part of the active data contract:

- `Datasets/`: shared raw datasets. CIFAR10 lives in `Datasets/CIFAR10/`;
  ImageNet100 uses `Datasets/ImageNet100/{train,val}/`.
- `Bundles/<bundle_id>/`: weights, manifest, exit curves, optional simulation
  layer statistics, MNN segments, and bundle-specific analysis.
- `Profiles/{Compute,Segments}/`: measured device profiles.
- `Datasets/<dataset>/TestSets/`: small Git-trackable test packages for
  device-side balanced, easy, and hard sample evaluation.
- `Runtime/`: disposable current-run caches and device output.
- `Archive/`: preserved legacy inputs and historical runtime output. Active code
  must never read assets from this directory.

Every active model artifact belongs under its bundle. Legacy top-level
`Weights`, `OfflineTables`, `PartitionManifests`, `SegmentProfiles`, and
`ComputeProfiles` directories are unsupported.
