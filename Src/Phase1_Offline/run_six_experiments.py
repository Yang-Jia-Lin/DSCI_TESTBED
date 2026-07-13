"""Run the six V100 offline experiments sequentially with resumable stages."""
import argparse, importlib.util, os, shutil, subprocess, sys
import torch
from Src.Shared.Config.paths import DATA_DIR
from Src.Phase1_Offline.Datasets.prepare_datasets import prepare_cifar, prepare_imagenet, prepare_neu
BUNDLES=("resnet50-cifar10","resnet50-neucls64","vit-base-cifar10","vit-base-neucls64","resnet50-imagenet100","vit-base-imagenet100")

def command(module,bundle,*args): return [sys.executable,"-m",module,"--bundle-id",bundle,*map(str,args)]
def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--stages",nargs="+",choices=("prepare","train","exits","manifest","curves","profile","plots"),default=("prepare","train","exits","manifest","curves","profile","plots")); p.add_argument("--bundles",nargs="+",choices=BUNDLES,default=BUNDLES); p.add_argument("--resume",action="store_true"); p.add_argument("--overwrite",action="store_true"); p.add_argument("--dry-run",action="store_true"); a=p.parse_args()
 if not shutil.which("nvidia-smi") or not torch.cuda.is_available(): raise RuntimeError("CUDA V100 is unavailable")
 if shutil.disk_usage(DATA_DIR).free < 15 * 1024**3: raise RuntimeError("At least 15 GiB free disk is required")
 os.environ.setdefault("HF_HOME",str(DATA_DIR/".cache"/"huggingface"))
 for package in ("timm","huggingface_hub","sklearn","fvcore"):
  if importlib.util.find_spec(package) is None: raise RuntimeError(f"Missing dependency: {package}")
 if "prepare" in a.stages and not a.dry_run: print(prepare_cifar()); print(prepare_neu()); print(prepare_imagenet())
 for bundle in a.bundles:
  commands=[]
  if "train" in a.stages: commands.append(command("Src.Phase1_Offline.Training.train_model",bundle,*(('--resume',) if a.resume else ())))
  if "exits" in a.stages: commands.append(command("Src.Phase1_Offline.Training.finetune_exits",bundle,*(('--resume',) if a.resume else ())))
  flag=("--overwrite",) if a.overwrite else ()
  if "manifest" in a.stages: commands.append(command("Src.Phase1_Offline.Profiling.generate_partition_manifest",bundle,*flag))
  if "curves" in a.stages: commands.append(command("Src.Phase1_Offline.LookupTables.generate_exit_curves",bundle,*flag))
  if "profile" in a.stages: commands.append(command("Src.Phase1_Offline.Profiling.generate_offline_profile",bundle,*flag))
  if "plots" in a.stages: commands.append(command("Src.Phase1_Offline.Training.plot_exit_analysis",bundle))
  for cmd in commands:
   print("+"," ".join(cmd),flush=True)
   if not a.dry_run: subprocess.run(cmd,check=True)
if __name__=="__main__": main()
