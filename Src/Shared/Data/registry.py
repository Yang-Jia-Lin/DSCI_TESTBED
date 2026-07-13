"""Dataset construction driven by a model bundle and deterministic manifests."""
from __future__ import annotations
import csv, random, re
from pathlib import Path
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from Src.Shared.Config.model_config import ModelBundleSpec
from Src.Shared.Config.paths import bundle_paths
IMAGE_EXTENSIONS={".bmp",".jpeg",".jpg",".png",".tif",".tiff",".webp"}

def build_transform(bundle: ModelBundleSpec, *, train=False):
 from torchvision import transforms
 from torchvision.transforms import InterpolationMode
 size=bundle.input_shape[1]; mode=InterpolationMode.BICUBIC if bundle.interpolation=="bicubic" else InterpolationMode.BILINEAR
 ops=[]
 if train:
  ops += [transforms.RandomResizedCrop(size,interpolation=mode),transforms.RandomHorizontalFlip()]
  if bundle.dataset_id=="neucls64": ops += [transforms.RandomVerticalFlip(),transforms.RandomRotation(15,interpolation=mode)]
 else: ops += [transforms.Resize(size,interpolation=mode),transforms.CenterCrop(size)]
 return transforms.Compose([*ops,transforms.ToTensor(),transforms.Normalize(bundle.mean,bundle.std)])

class ManifestImageDataset(Dataset):
 def __init__(self,root,manifest,transform):
  self.root,self.transform=Path(root),transform
  with Path(manifest).open(newline="",encoding="utf-8") as f: self.rows=list(csv.DictReader(f))
  if not self.rows: raise ValueError(f"Empty manifest: {manifest}")
  self.targets=[int(x["label"]) for x in self.rows]; self.classes=sorted({x["synset"] for x in self.rows})
 def __len__(self): return len(self.rows)
 def __getitem__(self,index):
  from PIL import Image
  row=self.rows[index]; path=self.root/row["relative_path"]
  with Image.open(path) as im: return self.transform(im.convert("RGB")),int(row["label"])

class FilenameClassImageDataset(Dataset):
 def __init__(self,image_dir,transform=None):
  self.transform=transform; raw=[]
  for p in sorted(Path(image_dir).iterdir()):
   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
    m=re.match(r"(.+)_\d+$",p.stem); raw.append((p,m.group(1) if m else p.stem.rsplit("_",1)[0]))
  self.classes=sorted({x[1] for x in raw}); mapping={x:i for i,x in enumerate(self.classes)}; self.samples=[(p,mapping[y]) for p,y in raw]; self.targets=[y for _,y in self.samples]
 def __len__(self): return len(self.samples)
 def __getitem__(self,index):
  from PIL import Image
  p,y=self.samples[index]
  with Image.open(p) as im: return self.transform(im.convert("RGB")),y

def _manifest_dataset(bundle,split,root,transform):
 manifest=root/"metadata"/f"{split}_manifest.csv"
 if not manifest.is_file(): raise FileNotFoundError(f"Missing {manifest}; run prepare_datasets first")
 if bundle.dataset_id=="cifar10":
  from torchvision.datasets import CIFAR10
  base=CIFAR10(str(root),train=split!="test",transform=transform,download=False)
  with manifest.open(newline="",encoding="utf-8") as f: indices=[int(x["source_index"]) for x in csv.DictReader(f)]
  return Subset(base,indices)
 source=root/"source" if bundle.dataset_id=="imagenet100" else root
 if bundle.dataset_id=="imagenet100" and (not source.is_symlink() or not source.resolve().is_dir()): raise FileNotFoundError(f"Invalid ImageNet symlink: {source}")
 return ManifestImageDataset(source,manifest,transform)

def build_dataset(bundle,split,*,data_root=None,download=False):
 from torchvision import datasets
 if split not in {"train","val","test"}: raise ValueError(split)
 root=Path(data_root or bundle_paths(bundle.bundle_id).dataset_root); transform=build_transform(bundle,train=split=="train")
 if not bundle.bundle_id.endswith("-ee-v1"):
  if download: raise ValueError("Downloads are disabled for manifest-backed bundles")
  return _manifest_dataset(bundle,split,root,transform)
 train=split=="train"
 if bundle.dataset_id=="cifar10": return datasets.CIFAR10(str(root),train=train,transform=transform,download=download)
 if bundle.dataset_id=="imagenet100": return datasets.ImageFolder(str(root/("train" if train else "val")),transform=transform)
 candidates=("train",) if train else ("val","valid","test"); split_dir=next((root/x for x in candidates if (root/x).is_dir()),None)
 if split_dir is None: raise FileNotFoundError(root)
 image_dir=split_dir/"images"; return FilenameClassImageDataset(image_dir,transform) if image_dir.is_dir() else datasets.ImageFolder(str(split_dir),transform=transform)

def _seed_worker(_):
 seed=torch.initial_seed()%(2**32); random.seed(seed)
 try:
  import numpy as np; np.random.seed(seed)
 except ImportError: pass

def build_loader(bundle,split,*,batch_size=64,num_workers=0,data_root=None,download=False,pin_memory=None,persistent_workers=None):
 ds=build_dataset(bundle,split,data_root=data_root,download=download); pin=torch.cuda.is_available() if pin_memory is None else pin_memory; persistent=(num_workers>0) if persistent_workers is None else persistent_workers and num_workers>0
 return DataLoader(ds,batch_size=batch_size,shuffle=split=="train",num_workers=num_workers,pin_memory=pin,persistent_workers=persistent,worker_init_fn=_seed_worker,generator=torch.Generator().manual_seed(42))

class TestPackageDataset(Dataset):
    """Manifest-backed image dataset exported for device-side testing."""

    def __init__(self, bundle: ModelBundleSpec, package_root: str | Path):
        self.bundle = bundle
        self.package_root = Path(package_root)
        self.manifest_path = self.package_root / "manifest.csv"
        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"Test package manifest not found: {self.manifest_path}")
        self.transform = build_transform(bundle, train=False)
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError(f"Test package manifest is empty: {self.manifest_path}")
        required = {"sample_id", "label", "relative_path"}
        missing = required.difference(self.rows[0])
        if missing:
            raise ValueError(
                f"Test package manifest missing columns {sorted(missing)}: {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        from PIL import Image

        row = self.rows[index]
        image_path = self.package_root / row["relative_path"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Test package image not found: {image_path}")
        with Image.open(image_path) as image:
            tensor = self.transform(image.convert("RGB"))
        label = int(row["label"])
        metadata = {
            "sample_id": row.get("sample_id", ""),
            "source_index": row.get("source_index", ""),
            "difficulty": row.get("difficulty", ""),
        }
        return tensor, label, metadata


def build_test_package_dataset(
    bundle: ModelBundleSpec,
    package_root: str | Path,
) -> TestPackageDataset:
    return TestPackageDataset(bundle, package_root)


def build_test_package_loader(
    bundle: ModelBundleSpec,
    package_root: str | Path,
    *,
    batch_size=1,
    num_workers=0,
):
    dataset = build_test_package_dataset(bundle, package_root)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
