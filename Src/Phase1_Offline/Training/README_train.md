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
