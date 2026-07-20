"""Shared deterministic AMP training helpers."""
import json, os, platform, random
from pathlib import Path
import numpy as np
import torch

def seed_all(seed=42):
 random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
 if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
 torch.backends.cudnn.benchmark=True

def atomic_save(payload,path):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,tmp); os.replace(tmp,path)

def append_epoch_log(path, row):
 import csv
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
 fieldnames=list(row)
 if path.is_file() and path.stat().st_size:
  with path.open("r",encoding="utf-8",newline="") as handle:
   reader=csv.DictReader(handle); existing_fields=reader.fieldnames or fieldnames
   for current in reader:
    if str(current.get("stage"))==str(row.get("stage")) and int(current.get("epoch",-1))==int(row.get("epoch",-2)):
     return False
  fieldnames=existing_fields
 new_file=not path.is_file() or path.stat().st_size==0
 with path.open("a",encoding="utf-8",newline="") as handle:
  writer=csv.DictWriter(handle,fieldnames=fieldnames,extrasaction="ignore")
  if new_file: writer.writeheader()
  writer.writerow(row); handle.flush(); os.fsync(handle.fileno())
 return True

def cpu_state(module):
 return {key:value.detach().cpu().clone() for key,value in module.state_dict().items()}

def checkpoint_payload(model,optimizer,scheduler,scaler,epoch,best_acc,bad_epochs,extra=None):
 data={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":scaler.state_dict(),"epoch":epoch,"best_acc":best_acc,"bad_epochs":bad_epochs,"python_rng":random.getstate(),"numpy_rng":np.random.get_state(),"torch_rng":torch.get_rng_state()}
 if torch.cuda.is_available(): data["cuda_rng"]=torch.cuda.get_rng_state_all()
 data.update(extra or {}); return data

def restore(path,model,optimizer=None,scheduler=None,scaler=None,device="cpu"):
 data=torch.load(path,map_location="cpu",weights_only=False); model.load_state_dict(data["model"])
 if optimizer: optimizer.load_state_dict(data["optimizer"])
 if scheduler: scheduler.load_state_dict(data["scheduler"])
 if scaler: scaler.load_state_dict(data["scaler"])
 random.setstate(data["python_rng"]); np.random.set_state(data["numpy_rng"]); torch.set_rng_state(data["torch_rng"])
 if torch.cuda.is_available() and "cuda_rng" in data: torch.cuda.set_rng_state_all(data["cuda_rng"])
 return data

def cosine_warmup(optimizer,epochs,warmup=2):
 import math
 def factor(step):
  if step<warmup: return float(step+1)/max(warmup,1)
  return .5*(1+math.cos(math.pi*(step-warmup)/max(epochs-warmup,1)))
 return torch.optim.lr_scheduler.LambdaLR(optimizer,factor)

def run_epoch(model,loader,criterion,device,*,optimizer=None,scaler=None,accumulation=1,exit_id=None,train_head=None):
 training=optimizer is not None
 if exit_id is None: model.train(training)
 else:
  model.eval()
  if train_head is not None: train_head.train(training)
 total_loss=total_correct=total=0
 if training: optimizer.zero_grad(set_to_none=True)
 context=torch.enable_grad if training else torch.no_grad
 with context():
  for step,(images,labels) in enumerate(loader):
   images=images.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True)
   with torch.amp.autocast("cuda",enabled=device.type=="cuda",dtype=torch.float16):
    logits=model(images,exit_id=exit_id) if exit_id else model(images); raw_loss=criterion(logits,labels); loss=raw_loss/accumulation
   if training:
    scaler.scale(loss).backward()
    if (step+1)%accumulation==0 or step+1==len(loader): scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
   total_loss += float(raw_loss.item())*labels.size(0); total_correct += int((logits.argmax(1)==labels).sum()); total += labels.size(0)
 return total_loss/total,100*total_correct/total

def copy_split_metadata(bundle, analysis_root):
 import shutil
 from Src.Shared.Config.paths import bundle_paths
 source=bundle_paths(bundle.bundle_id).dataset_root/"metadata"; target=Path(analysis_root)/"splits"; target.mkdir(parents=True,exist_ok=True)
 for name in ("train_manifest.csv","val_manifest.csv","test_manifest.csv","class_to_idx.json","source_info.json","imagenet100_synsets.txt"):
  if (source/name).is_file(): shutil.copy2(source/name,target/name)

def write_environment(path,bundle,config):
 import importlib.metadata as md
 import hashlib
 from Src.Shared.Config.paths import bundle_paths
 weight_path=bundle_paths(bundle.bundle_id).weight_path; digest=hashlib.sha256(weight_path.read_bytes()).hexdigest() if weight_path.is_file() else None
 payload={"bundle_id":bundle.bundle_id,"weights_sha256":digest,"pretrained_source":bundle.pretrained_source,"seed":42,"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"cudnn":torch.backends.cudnn.version(),"gpu":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,"packages":{x:md.version(x) for x in ("torchvision","timm","huggingface-hub","fvcore")},"config":config}
 Path(path).write_text(json.dumps(payload,indent=2)+"\n")
