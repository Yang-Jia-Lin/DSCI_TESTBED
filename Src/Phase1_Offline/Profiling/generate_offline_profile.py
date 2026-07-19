"""Generate boundary FLOPs and the final test-only offline profile."""
import argparse, importlib.metadata as md, json
from pathlib import Path
import pandas as pd
import torch, torch.nn as nn
from fvcore.nn import FlopCountAnalysis
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import bundle_paths
from Src.Shared.Data.registry import build_loader
from Src.Shared.Models.factory import build_model
from Src.Shared.Partitioning.manifest import load_partition_manifest, model_file_hash

class Segment(nn.Module):
 def __init__(self,model,name): super().__init__(); self.model,self.name=model,name
 def forward(self,x): return self.model.execute_partition_segment(self.name,x)

def layer_stats(bundle,model,manifest):
 x=torch.zeros((1,*bundle.input_shape)); rows=[dict(boundary_id=0,name="input",output_shape=str(tuple(x.shape)),num_elements=x.numel(),num_bytes=x.numel()*x.element_size(),serialized_num_bytes=manifest.boundaries[0]["serialized_num_bytes"],approx_flops=0.)]
 model.eval()
 for segment,boundary in zip(manifest.segments,manifest.boundaries[1:]):
  wrapper=Segment(model,segment["name"]).eval(); flops=float(FlopCountAnalysis(wrapper,x).unsupported_ops_warnings(False).uncalled_modules_warnings(False).total())
  with torch.no_grad(): x=wrapper(x)
  rows.append(dict(boundary_id=boundary["boundary_id"],name=segment["name"],output_shape=str(tuple(x.shape)),num_elements=x.numel(),num_bytes=x.numel()*x.element_size(),serialized_num_bytes=boundary["serialized_num_bytes"],approx_flops=flops))
 return pd.DataFrame(rows)

def test_accuracy(bundle,model,device,batch_size,num_workers):
 loader=build_loader(bundle,"test",batch_size=batch_size,num_workers=num_workers); correct={x.exit_id:0 for x in bundle.exits}; correct["final"]=0; total=0; model.eval()
 with torch.no_grad():
  for images,labels in loader:
   images,labels=images.to(device,non_blocking=True),labels.to(device,non_blocking=True); features=model.forward_features(images)
   for item in bundle.exits: correct[item.exit_id]+=int((model.classify_exit(item.exit_id,features[item.attach_point]).argmax(1)==labels).sum())
   correct["final"]+=int((model(images).argmax(1)==labels).sum()); total+=labels.numel()
 return {key:100*value/total for key,value in correct.items()},total

def main(argv=None):
 p=argparse.ArgumentParser(); p.add_argument("--bundle-id",required=True); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--num-workers",type=int,default=16); p.add_argument("--overwrite",action="store_true"); a=p.parse_args(argv)
 bundle=get_bundle(a.bundle_id); paths=bundle_paths(bundle.bundle_id); manifest=load_partition_manifest(bundle.bundle_id); model=build_model(bundle); model.load_state_dict(torch.load(paths.weight_path,map_location="cpu",weights_only=True)); stats=layer_stats(bundle,model,manifest)
 if paths.layer_stats_path.exists() and not a.overwrite: raise FileExistsError(paths.layer_stats_path)
 stats.to_csv(paths.layer_stats_path,index=False); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=model.to(device); accuracies,count=test_accuracy(bundle,model,device,a.batch_size,a.num_workers)
 metadata_path=paths.analysis_root/"training_metadata.json"; training=json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
 dataset_meta=paths.dataset_root/"metadata"; source_info=json.loads((dataset_meta/"source_info.json").read_text()) if (dataset_meta/"source_info.json").is_file() else {"seed":42}; class_map=json.loads((dataset_meta/"class_to_idx.json").read_text())
 payload={"bundle_id":bundle.bundle_id,"architecture":bundle.architecture,"dataset_id":bundle.dataset_id,"pretrained_source":bundle.pretrained_source,"model_hash":model_file_hash(paths.weight_path),"manifest_id":manifest.manifest_id,"input_shape":list(bundle.input_shape),"interpolation":bundle.interpolation,"normalization":{"mean":bundle.mean,"std":bundle.std},"exits":[{"exit_id":x.exit_id,"attach_point":x.attach_point,"test_accuracy":accuracies[x.exit_id]} for x in bundle.exits]+[{"exit_id":"final","test_accuracy":accuracies["final"]}],"test_samples":count,"segment_flops":stats["approx_flops"].tolist(),"cumulative_flops":stats["approx_flops"].cumsum().tolist(),"activation_bytes":stats["num_bytes"].astype(int).tolist(),"serialized_bytes":stats["serialized_num_bytes"].astype(int).tolist(),"dataset":{"source_info":source_info,"class_to_idx":class_map},"training":training,"versions":{x:md.version(x) for x in ("torch","torchvision","timm","fvcore")}}
 (paths.root/"offline_profile.json").write_text(json.dumps(payload,indent=2)+"\n"); print(json.dumps({"profile":str(paths.root/"offline_profile.json"),"test_accuracy":accuracies},indent=2))
if __name__=="__main__": main()
