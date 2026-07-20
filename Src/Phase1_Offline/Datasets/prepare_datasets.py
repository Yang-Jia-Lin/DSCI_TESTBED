"""Create deterministic local split manifests without copying image data."""
import argparse, csv, json, random
from datetime import datetime, timezone
from pathlib import Path
from Src.Shared.Config.paths import DATASET_DIR

SEED=42
DEFAULT_IMAGENET_SOURCE=Path("/root/commonfiles/Datasets/ImageNet2012")
SYNSETS="""n02869837 n01749939 n02488291 n02107142 n13037406 n02091831 n04517823 n04589890 n03062245 n01773797 n01735189 n07831146 n07753275 n03085013 n04485082 n02105505 n01983481 n02788148 n03530642 n04435653 n02086910 n02859443 n13040303 n03594734 n02085620 n02099849 n01558993 n04493381 n02109047 n04111531 n02877765 n04429376 n02009229 n01978455 n02106550 n01820546 n01692333 n07714571 n02974003 n02114855 n03785016 n03764736 n03775546 n02087046 n07836838 n04099969 n04592741 n03891251 n02701002 n03379051 n02259212 n07715103 n03947888 n04026417 n02326432 n03637318 n01980166 n02113799 n02086240 n03903868 n02483362 n04127249 n02089973 n03017168 n02093428 n02804414 n02396427 n04418357 n02172182 n01729322 n02113978 n03787032 n02089867 n02119022 n03777754 n04238763 n02231487 n03032252 n02138441 n02104029 n03837869 n03494278 n04136333 n03794056 n03492542 n02018207 n04067472 n03930630 n03584829 n02123045 n04229816 n02100583 n03642806 n04336792 n03259280 n02116738 n02108089 n03424325 n01855672 n02090622""".split()
SUFFIX={".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}
FIELDS=("relative_path","synset","label","source_index","split")

def write(path,rows):
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)

def prepare_imagenet(source=DEFAULT_IMAGENET_SOURCE):
 root=DATASET_DIR/"ImageNet100"; source=Path(source).resolve(); root.mkdir(parents=True,exist_ok=True)
 if not (source/"train").is_dir() or not (source/"val").is_dir(): raise FileNotFoundError(source)
 link=root/"source"
 if link.is_symlink() and link.resolve()!=source: link.unlink()
 if link.exists() and not link.is_symlink(): raise FileExistsError(link)
 if not link.exists(): link.symlink_to(source,target_is_directory=True)
 meta=root/"metadata"; meta.mkdir(exist_ok=True); mapping={s:i for i,s in enumerate(SYNSETS)}
 (meta/"imagenet100_synsets.txt").write_text("\n".join(SYNSETS)+"\n")
 (meta/"class_to_idx.json").write_text(json.dumps(mapping,indent=2)+"\n")
 out={x:[] for x in ("train","val","test")}
 for syn,label in mapping.items():
  if not (source/"train"/syn).is_dir() or not (source/"val"/syn).is_dir(): raise ValueError(f"Missing synset {syn}")
  items=list(enumerate(sorted(p for p in (source/"train"/syn).iterdir() if p.suffix.lower() in SUFFIX)))
  random.Random(f"{SEED}:{syn}").shuffle(items); cut=int(len(items)*.9)
  for split,part in (("train",items[:cut]),("val",items[cut:])):
   out[split]+=[dict(relative_path=str(p.relative_to(source)),synset=syn,label=label,source_index=i,split=split) for i,p in part]
  tests=sorted(p for p in (source/"val"/syn).iterdir() if p.suffix.lower() in SUFFIX)
  out["test"] += [dict(relative_path=str(p.relative_to(source)),synset=syn,label=label,source_index=i,split="test") for i,p in enumerate(tests)]
 for split,rows in out.items(): write(meta/f"{split}_manifest.csv",rows)
 counts={k:len(v) for k,v in out.items()}
 if counts["test"]!=5000: raise ValueError(counts)
 info=dict(dataset_id="imagenet100",source=str(source),seed=SEED,classes=100,counts=counts,generated_at=datetime.now(timezone.utc).isoformat())
 (meta/"source_info.json").write_text(json.dumps(info,indent=2)+"\n"); return info

def prepare_imagenet1000(source=DEFAULT_IMAGENET_SOURCE):
 root=DATASET_DIR/"ImageNet1000"; source=Path(source).resolve(); root.mkdir(parents=True,exist_ok=True)
 if not (source/"train").is_dir() or not (source/"val").is_dir(): raise FileNotFoundError(source)
 link=root/"source"
 if link.is_symlink() and link.resolve()!=source: link.unlink()
 if link.exists() and not link.is_symlink(): raise FileExistsError(link)
 if not link.exists(): link.symlink_to(source,target_is_directory=True)
 # Match torchvision.datasets.ImageFolder/ImageNet class-index assignment.
 synsets=sorted(p.name for p in (source/"val").iterdir() if p.is_dir())
 train_synsets=sorted(p.name for p in (source/"train").iterdir() if p.is_dir())
 if len(synsets)!=1000 or train_synsets!=synsets: raise ValueError({"train_classes":len(train_synsets),"val_classes":len(synsets)})
 mapping={syn:i for i,syn in enumerate(synsets)}; rows=[]
 for syn,label in mapping.items():
  images=sorted(p for p in (source/"val"/syn).iterdir() if p.suffix.lower() in SUFFIX)
  if len(images)!=50: raise ValueError(f"Expected 50 validation images for {syn}, found {len(images)}")
  rows += [dict(relative_path=str(p.relative_to(source)),synset=syn,label=label,source_index=i,split="test") for i,p in enumerate(images)]
 meta=root/"metadata"; write(meta/"test_manifest.csv",rows)
 (meta/"imagenet1000_synsets.txt").write_text("\n".join(synsets)+"\n")
 (meta/"class_to_idx.json").write_text(json.dumps(mapping,indent=2)+"\n")
 info=dict(dataset_id="imagenet1000",source=str(source),seed=None,classes=1000,counts={"test":len(rows)},label_mapping="torchvision ImageFolder lexicographic synset order",generated_at=datetime.now(timezone.utc).isoformat())
 (meta/"source_info.json").write_text(json.dumps(info,indent=2)+"\n"); return info

def prepare_neu():
 root=DATASET_DIR/"NEU-CLS-64"; groups={}
 for p in sorted({x.resolve() for x in root.glob("*/images/*") if x.suffix.lower() in SUFFIX}): groups.setdefault(p.stem.rsplit("_",1)[0],[]).append(p)
 if len(groups)!=6 or any(len(x)!=300 for x in groups.values()): raise ValueError({k:len(v) for k,v in groups.items()})
 out={x:[] for x in ("train","val","test")}
 for label,syn in enumerate(sorted(groups)):
  items=list(enumerate(groups[syn])); random.Random(f"{SEED}:{syn}").shuffle(items)
  for split,part in (("train",items[:210]),("val",items[210:255]),("test",items[255:])):
   out[split]+=[dict(relative_path=str(p.relative_to(root)),synset=syn,label=label,source_index=i,split=split) for i,p in part]
 meta=root/"metadata"
 for split,rows in out.items(): write(meta/f"{split}_manifest.csv",rows)
 (meta/"class_to_idx.json").write_text(json.dumps({s:i for i,s in enumerate(sorted(groups))},indent=2)+"\n"); counts={k:len(v) for k,v in out.items()}; (meta/"source_info.json").write_text(json.dumps({"dataset_id":"neucls64","source":str(root.resolve()),"seed":SEED,"counts":counts},indent=2)+"\n"); return counts

def prepare_cifar():
 from torchvision.datasets import CIFAR10
 root=DATASET_DIR/"CIFAR10"; train=CIFAR10(str(root),train=True,download=False); out={x:[] for x in ("train","val","test")}
 for label in range(10):
  items=[i for i,y in enumerate(train.targets) if y==label]; random.Random(f"{SEED}:{label}").shuffle(items)
  for split,part in (("train",items[:4500]),("val",items[4500:])): out[split]+=[dict(relative_path=f"train/{i}",synset=str(label),label=label,source_index=i,split=split) for i in part]
 test=CIFAR10(str(root),train=False,download=False); out["test"]=[dict(relative_path=f"test/{i}",synset=str(y),label=y,source_index=i,split="test") for i,y in enumerate(test.targets)]
 meta=root/"metadata"
 for split,rows in out.items(): write(meta/f"{split}_manifest.csv",rows)
 (meta/"class_to_idx.json").write_text(json.dumps(train.class_to_idx,indent=2)+"\n"); counts={k:len(v) for k,v in out.items()}; (meta/"source_info.json").write_text(json.dumps({"dataset_id":"cifar10","source":str(root.resolve()),"seed":SEED,"counts":counts},indent=2)+"\n"); return counts

def main():
 p=argparse.ArgumentParser(); p.add_argument("--dataset",choices=("all","cifar10","imagenet100","imagenet1000","neucls64"),default="all"); p.add_argument("--imagenet-source",default=str(DEFAULT_IMAGENET_SOURCE)); a=p.parse_args(); result={}
 if a.dataset in ("all","cifar10"): result["cifar10"]=prepare_cifar()
 if a.dataset in ("all","imagenet100"): result["imagenet100"]=prepare_imagenet(a.imagenet_source)
 if a.dataset in ("all","imagenet1000"): result["imagenet1000"]=prepare_imagenet1000(a.imagenet_source)
 if a.dataset in ("all","neucls64"): result["neucls64"]=prepare_neu()
 print(json.dumps(result,indent=2))
if __name__=="__main__": main()
