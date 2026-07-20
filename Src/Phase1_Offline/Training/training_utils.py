"""Shared deterministic, AMP and checkpoint helpers for offline training."""
import math, os, random
from pathlib import Path
import torch

def seed_all(seed=42):
 random.seed(seed)
 try:
  import numpy as np; np.random.seed(seed)
 except ImportError: pass
 torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def default_batch(bundle,exits=False):
 return (128 if exits else 64) if bundle.architecture=="vit-base" else (256 if exits else 128)

def make_scheduler(optimizer,epochs,warmup=2,start_epoch=-1):
 def factor(epoch):
  if epoch<warmup: return float(epoch+1)/max(warmup,1)
  progress=(epoch-warmup)/max(epochs-warmup-1,1); return .5*(1+math.cos(math.pi*progress))
 return torch.optim.lr_scheduler.LambdaLR(optimizer,factor,last_epoch=start_epoch)

def scaler(enabled): return torch.amp.GradScaler("cuda",enabled=enabled)
def autocast(enabled): return torch.amp.autocast("cuda",dtype=torch.float16,enabled=enabled)
def atomic_save(payload,path):
 path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); temp=path.with_suffix(path.suffix+".tmp"); torch.save(payload,temp); os.replace(temp,path)

def checkpoint_payload(model,optimizer,scheduler,grad_scaler,epoch,best_acc,best_state,extra=None):
 payload={"model":model.state_dict(),"optimizer":optimizer.state_dict(),"scheduler":scheduler.state_dict(),"scaler":grad_scaler.state_dict(),"epoch":epoch,"best_acc":best_acc,"best_state":best_state,"python_rng":random.getstate(),"torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
 payload.update(extra or {}); return payload

def restore_rng(state):
 random.setstate(state["python_rng"]); torch.set_rng_state(state["torch_rng"])
 if torch.cuda.is_available() and state.get("cuda_rng") is not None: torch.cuda.set_rng_state_all(state["cuda_rng"])

def probe_batch(model,loader,criterion,device,amp):
 images,labels=next(iter(loader)); images=images.to(device,non_blocking=True); labels=labels.to(device,non_blocking=True); model.train(); model.zero_grad(set_to_none=True)
 with autocast(amp): loss=criterion(model(images),labels)
 loss.backward(); model.zero_grad(set_to_none=True); del images,labels,loss
 if device.type=="cuda": torch.cuda.empty_cache()
