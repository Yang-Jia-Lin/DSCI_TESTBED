# Multi-Exit ResNet Training

This project uses the conda environment named `DSCI`.

```powershell
conda activate DSCI
```

## Dataset Paths

Default dataset roots are defined by the bundle:

```text
Data/Datasets/CIFAR10
Data/Datasets/ImageNet100
```

CIFAR-10 can be downloaded by torchvision with `--download`.

ImageNet100 must be prepared manually:

```text
Data/Datasets/ImageNet100/
  train/<class_name>/*.jpg
  val/<class_name>/*.jpg
```

The ImageNet100 loader checks that exactly 100 class folders exist.

To use another location, pass `--data-root <path>` to the training and curve-generation commands.

## Supported Bundles

Current ResNet101 bundles:

```text
resnet101-cifar10-ee-v1
resnet101-imagenet100-ee-v1
```

ResNet101 uses five late exits:

```text
after_layer3_block5   -> layer3.4
after_layer3_block10  -> layer3.9
after_layer3_block15  -> layer3.14
after_layer3_block20  -> layer3.19
after_layer4          -> layer4
```

## 1. Train Backbone and Final Classifier

CIFAR-10:

```powershell
$bundle = "resnet101-cifar10-ee-v1"
python -m Src.Phase1_Offline.Training.train_model --bundle-id $bundle --download --epochs 100 --batch-size 32
```

ImageNet100:

```powershell
$bundle = "resnet101-imagenet100-ee-v1"
python -m Src.Phase1_Offline.Training.train_model --bundle-id $bundle --epochs 100 --batch-size 32
```

This writes:

```text
Data/Bundles/<bundle_id>/weights.pth
Data/Bundles/<bundle_id>/analysis/train_model_log.csv
```

## 2. Fine-Tune Early Exits

CIFAR-10:

```powershell
python -m Src.Phase1_Offline.Training.finetune_exits --bundle-id $bundle --download --epochs-per-exit 50 --batch-size 32
```

ImageNet100:

```powershell
python -m Src.Phase1_Offline.Training.finetune_exits --bundle-id $bundle --epochs-per-exit 50 --batch-size 32
```

This freezes the backbone and trains each exit head. It writes:

```text
Data/Bundles/<bundle_id>/analysis/finetune_exits_log.csv
```

## 3. Generate Manifest and Threshold Test CSV

Generate the partition manifest first, because the combined expectation plot needs exit boundary ids:

```powershell
python -m Src.Phase1_Offline.Profiling.generate_partition_manifest --bundle-id $bundle --overwrite
```

Then scan thresholds on the validation split:

```powershell
python -m Src.Phase1_Offline.LookupTables.generate_exit_curves --bundle-id $bundle --overwrite
```

For CIFAR-10, add `--download` if the dataset has not been downloaded:

```powershell
python -m Src.Phase1_Offline.LookupTables.generate_exit_curves --bundle-id $bundle --download --overwrite
```

This writes:

```text
Data/Bundles/<bundle_id>/exit_curves.csv
Data/Bundles/<bundle_id>/analysis/threshold_curves.csv
```

## 4. Plot the Four Curves

```powershell
python -m Src.Phase1_Offline.Training.plot_exit_analysis --bundle-id $bundle
```

This creates PNG and PDF files under:

```text
Data/Bundles/<bundle_id>/analysis/
```

The four figures are:

```text
<bundle_id>_training_convergence
<bundle_id>_exit_probability
<bundle_id>_accuracy_threshold
<bundle_id>_combined_expectation
```

## Full CIFAR-10 Example

```powershell
conda activate DSCI
$bundle = "resnet101-cifar10-ee-v1"

python -m Src.Phase1_Offline.Training.train_model --bundle-id $bundle --download --epochs 100 --batch-size 32
python -m Src.Phase1_Offline.Training.finetune_exits --bundle-id $bundle --download --epochs-per-exit 50 --batch-size 32
python -m Src.Phase1_Offline.Profiling.generate_partition_manifest --bundle-id $bundle --overwrite
python -m Src.Phase1_Offline.LookupTables.generate_exit_curves --bundle-id $bundle --download --overwrite
python -m Src.Phase1_Offline.Training.plot_exit_analysis --bundle-id $bundle
```

## Adding a New Dataset

Add the dataset spec in `Src/Shared/Config/model_config.py`, then add its loader branch in `Src/Shared/Data/registry.py`.

After that, the same training, threshold-test, and plotting commands work through `--bundle-id`.


## V100 six-bundle workflow (Linux)

Prepare local manifests and the ImageNet2012 symlink without copying images:

```bash
conda run -n DSCI python -m Src.Phase1_Offline.Datasets.prepare_datasets --dataset all
```

Run all stages sequentially on the single V100 (safe to resume after SSH interruption):

```bash
conda run -n DSCI python -m Src.Phase1_Offline.run_six_experiments --resume --overwrite
```

Use `--dry-run`, `--stages`, or `--bundles` to inspect or limit work. New bundles are
`resnet50-{cifar10,imagenet100,neucls64}` and
`vit-base-{cifar10,imagenet100,neucls64}`. ImageNet-100 reads through
`Data/Datasets/ImageNet100/source`; it never downloads or copies ImageNet images.


## Experiment roles and fair baseline protocol

Pretraining source and target dataset are separate concepts. The three ResNet50
bundles start from the same torchvision ImageNet-1K V2 initialization, while the
three ViT bundles start from the same timm ImageNet-pretrained initialization.
Each model is then fine-tuned separately on CIFAR-10, ImageNet-100, or NEU-CLS.

The primary generalization matrix remains the six `2 x 3` bundles. For direct
comparison against scheduling, partitioning, or offloading baselines, use
`resnet50-imagenet100` as the single fixed workload and give every method the
same final `weights.pth`, partition manifest, exit curves, preprocessing, and
test manifest. Baseline methods must not independently fine-tune this checkpoint.

ImageNet-1K is currently a dataset-only compatibility package, not a seventh
training bundle. A future `resnet50-imagenet1000` experiment would require a
1,000-class model and trained early-exit heads; the current 100-class ImageNet-100
weights cannot be evaluated against ImageNet-1K labels.


## Export the four terminal test packages

Export the same model-independent, class-balanced test samples for every terminal:

```bash
conda run --no-capture-output -n DSCI python -m Src.Phase1_Offline.Datasets.export_manifest_test_packages --samples-per-class 10 --seed 42 --overwrite
```

This creates 100 CIFAR-10, 1,000 ImageNet-100, 60 NEU-CLS, and 10,000
ImageNet-1K test images. Each package contains `manifest.csv`, `metadata.json`,
and class-organized `images/`. Balanced package names are dataset-level and are
shared by ResNet50, ViT, and any other model using the same class mapping:

```text
cifar10__test__balanced__10pc__seed42
neucls64__test__balanced__10pc__seed42
imagenet100__test__balanced__10pc__seed42
imagenet1000__test__balanced__10pc__seed42
```

Model-derived `easy` and `hard` packages keep their bundle prefix because their
difficulty labels depend on a specific checkpoint.

ImageNet-1K is a dataset-only test package and does not add a training bundle. Its
source is linked to `/root/commonfiles/Datasets/ImageNet2012`; the labeled official
validation split is treated as test, and 10 images per each of the 1,000 classes are
selected with seed 42. To prepare or export only this package:

```bash
conda run --no-capture-output -n DSCI python -m Src.Phase1_Offline.Datasets.prepare_datasets --dataset imagenet1000
conda run --no-capture-output -n DSCI python -m Src.Phase1_Offline.Datasets.export_manifest_test_packages --datasets imagenet1000 --samples-per-class 10 --seed 42 --copy-workers 16 --overwrite
```
