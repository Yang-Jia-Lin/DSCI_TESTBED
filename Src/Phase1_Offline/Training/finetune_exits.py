"""Train every early-exit head while the backbone remains frozen in eval mode."""
import argparse, copy, hashlib
import torch, torch.nn as nn
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_loader
from Src.Shared.Models.factory import build_model, freeze_for_exit
from Src.Phase1_Offline.Training.runtime import atomic_save, checkpoint_payload, cosine_warmup, restore, run_epoch, seed_all, cpu_state, append_epoch_log

def frozen_digest(model):
 h=hashlib.sha256()
 for key,value in model.state_dict().items():
  if not key.startswith("exit_heads."): h.update(key.encode()); h.update(value.detach().cpu().numpy().tobytes())
 return h.hexdigest()

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--bundle-id",required=True); p.add_argument("--data-root"); p.add_argument("--epochs-per-exit",type=int,default=20); p.add_argument("--batch-size",type=int); p.add_argument("--num-workers",type=int,default=16); p.add_argument("--lr",type=float,default=1e-3); p.add_argument("--warmup-epochs",type=int,default=2); p.add_argument("--patience",type=int,default=5); p.add_argument("--resume",action="store_true"); a=p.parse_args(argv)
 seed_all(); bundle=get_bundle(a.bundle_id); paths=bundle_paths(bundle.bundle_id); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=build_model(bundle).to(device); model.load_state_dict(torch.load(paths.weight_path,map_location=device,weights_only=True)); baseline=frozen_digest(model); batch=a.batch_size or (128 if bundle.architecture=="vit-base" else 256); log_path=paths.analysis_root/"finetune_exits_log.csv"; criterion=nn.CrossEntropyLoss()
 def loaders(bs):
  kw=dict(batch_size=bs,num_workers=a.num_workers,data_root=a.data_root); return build_loader(bundle,"train",**kw),build_loader(bundle,"val",**kw)
 for exit_spec in bundle.exits:
  freeze_for_exit(model,exit_spec.exit_id); head=model.exit_heads[exit_spec.exit_id]; optimizer=torch.optim.AdamW(head.parameters(),lr=a.lr); scheduler=cosine_warmup(optimizer,a.epochs_per_exit,a.warmup_epochs); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda"); checkpoint=paths.analysis_root/f"checkpoint_{exit_spec.exit_id}_latest.pth"; start=0; best=-1.; bad=0; accumulation=1; best_head=cpu_state(head)
  if a.resume and checkpoint.is_file():
   state=restore(checkpoint,model,optimizer,scheduler,scaler,device); start=state["epoch"]+1; best=state["best_acc"]; bad=state["bad_epochs"]; batch=state.get("batch_size",batch); accumulation=state.get("accumulation",1); best_head=state.get("best_head",best_head)
  train_loader,val_loader=loaders(batch); epoch=start
  while epoch<a.epochs_per_exit and bad<a.patience:
   snap=checkpoint_payload(model,optimizer,scheduler,scaler,epoch-1,best,bad,{"best_head":best_head,"batch_size":batch,"accumulation":accumulation}); retry_checkpoint=paths.analysis_root/f"checkpoint_{exit_spec.exit_id}_epoch_start.pth"; atomic_save(snap,retry_checkpoint)
   try: train_loss,train_acc=run_epoch(model,train_loader,criterion,device,optimizer=optimizer,scaler=scaler,accumulation=accumulation,exit_id=exit_spec.exit_id,train_head=head)
   except torch.cuda.OutOfMemoryError:
    if batch<=1: raise
    del train_loader,val_loader; torch.cuda.empty_cache(); restore(retry_checkpoint,model,optimizer,scheduler,scaler,device); batch=max(1,batch//2); accumulation*=2; train_loader,val_loader=loaders(batch); print(f"OOM retry: batch={batch} accumulation={accumulation}"); continue
   val_loss,val_acc=run_epoch(model,val_loader,criterion,device,exit_id=exit_spec.exit_id,train_head=head)
   if val_acc>best: best=val_acc; bad=0; best_head=cpu_state(head)
   else: bad+=1
   scheduler.step(); row=dict(stage=exit_spec.exit_id,epoch=epoch+1,train_loss=train_loss,train_acc=train_acc,val_loss=val_loss,val_acc=val_acc,best_val_acc=best,batch_size=batch,accumulation=accumulation); append_epoch_log(log_path,row); atomic_save(checkpoint_payload(model,optimizer,scheduler,scaler,epoch,best,bad,{"best_head":best_head,"batch_size":batch,"accumulation":accumulation}),checkpoint); print(f"exit={exit_spec.exit_id} epoch={epoch+1} val_acc={val_acc:.2f} best={best:.2f}"); epoch+=1
  head.load_state_dict(best_head); atomic_save(model.state_dict(),paths.weight_path)
  if frozen_digest(model)!=baseline: raise RuntimeError("Frozen backbone or running statistics changed during exit training")
 print(f"Saved exit weights: {paths.weight_path}")
if __name__=="__main__": main()
