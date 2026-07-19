"""Fine-tune a pretrained backbone and final classifier on a V100."""
import argparse, copy
import torch, torch.nn as nn
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_loader
from Src.Shared.Models.factory import build_model
from Src.Phase1_Offline.Training.runtime import atomic_save, checkpoint_payload, cosine_warmup, restore, run_epoch, seed_all, write_environment, copy_split_metadata, cpu_state, append_epoch_log

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--bundle-id",required=True); p.add_argument("--data-root"); p.add_argument("--epochs",type=int,default=30); p.add_argument("--batch-size",type=int); p.add_argument("--num-workers",type=int,default=16); p.add_argument("--backbone-lr",type=float,default=1e-5); p.add_argument("--head-lr",type=float,default=1e-3); p.add_argument("--warmup-epochs",type=int,default=2); p.add_argument("--patience",type=int,default=5); p.add_argument("--resume",action="store_true"); a=p.parse_args(argv)
 seed_all(); bundle=get_bundle(a.bundle_id); paths=bundle_paths(bundle.bundle_id); paths.analysis_root.mkdir(parents=True,exist_ok=True); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
 batch=a.batch_size or (64 if bundle.architecture=="vit-base" else 128); accumulation=1; checkpoint=paths.analysis_root/"checkpoint_final_latest.pth"
 model=build_model(bundle,pretrained=bundle.pretrained_source is not None and not(a.resume and checkpoint.is_file())).to(device); head=model.final_classifier_module(); head_ids={id(x) for x in head.parameters()}; backbone=[x for x in model.parameters() if id(x) not in head_ids]; optimizer=torch.optim.AdamW([{"params":backbone,"lr":a.backbone_lr},{"params":head.parameters(),"lr":a.head_lr}]); scheduler=cosine_warmup(optimizer,a.epochs,a.warmup_epochs); scaler=torch.amp.GradScaler("cuda",enabled=device.type=="cuda")
 start=0; best=-1.; bad=0; best_model=cpu_state(model); log_path=paths.analysis_root/"train_model_log.csv"
 if a.resume and checkpoint.is_file():
  state=restore(checkpoint,model,optimizer,scheduler,scaler,device); start=state["epoch"]+1; best=state["best_acc"]; bad=state["bad_epochs"]; batch=state.get("batch_size",batch); accumulation=state.get("accumulation",1); best_model=state.get("best_model",best_model)
 def loaders(bs):
  kw=dict(batch_size=bs,num_workers=a.num_workers,data_root=a.data_root)
  return build_loader(bundle,"train",**kw),build_loader(bundle,"val",**kw)
 train_loader,val_loader=loaders(batch); criterion=nn.CrossEntropyLoss(); epoch=start
 while epoch<a.epochs and bad<a.patience:
  snapshot=checkpoint_payload(model,optimizer,scheduler,scaler,epoch-1,best,bad,{"best_model":best_model,"batch_size":batch,"accumulation":accumulation}); retry_checkpoint=paths.analysis_root/"checkpoint_final_epoch_start.pth"; atomic_save(snapshot,retry_checkpoint)
  try: train_loss,train_acc=run_epoch(model,train_loader,criterion,device,optimizer=optimizer,scaler=scaler,accumulation=accumulation)
  except torch.cuda.OutOfMemoryError:
   if batch<=1: raise
   del train_loader,val_loader; torch.cuda.empty_cache(); restore(retry_checkpoint,model,optimizer,scheduler,scaler,device); batch=max(1,batch//2); accumulation*=2; train_loader,val_loader=loaders(batch); print(f"OOM retry: batch={batch} accumulation={accumulation}"); continue
  val_loss,val_acc=run_epoch(model,val_loader,criterion,device)
  if val_acc>best: best=val_acc; bad=0; best_model=cpu_state(model)
  else: bad+=1
  scheduler.step(); row=dict(stage="final",epoch=epoch+1,train_loss=train_loss,train_acc=train_acc,val_loss=val_loss,val_acc=val_acc,best_val_acc=best,batch_size=batch,accumulation=accumulation); append_epoch_log(log_path,row)
  atomic_save(checkpoint_payload(model,optimizer,scheduler,scaler,epoch,best,bad,{"best_model":best_model,"batch_size":batch,"accumulation":accumulation}),checkpoint); print(f"epoch={epoch+1} train_acc={train_acc:.2f} val_acc={val_acc:.2f} best={best:.2f}"); epoch+=1
 model.load_state_dict(best_model); atomic_save(model.state_dict(),paths.weight_path)
 copy_split_metadata(bundle,paths.analysis_root); write_environment(paths.analysis_root/"training_metadata.json",bundle,vars(a)|{"resolved_batch_size":batch,"accumulation":accumulation,"device":str(device)})
 print(f"Saved best weights: {paths.weight_path}")
if __name__=="__main__": main()
