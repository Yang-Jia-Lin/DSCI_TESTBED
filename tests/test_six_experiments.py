import csv, unittest
from pathlib import Path
import torch
from Src.Shared.Config.model_config import get_bundle
from Src.Shared.Config.paths import DATASET_DIR
from Src.Shared.Data.registry import build_dataset
from Src.Shared.Models.factory import build_model

IDS=("resnet50-cifar10","resnet50-imagenet100","resnet50-neucls64","vit-base-cifar10","vit-base-imagenet100","vit-base-neucls64")
class SixExperimentTests(unittest.TestCase):
 def test_registry_and_exits(self):
  for bundle_id in IDS:
   bundle=get_bundle(bundle_id); self.assertEqual(bundle.input_shape,(3,224,224)); self.assertEqual(len(bundle.exits),3)
 def test_split_counts_and_disjointness(self):
  expected={"CIFAR10":(45000,5000,10000),"ImageNet100":(114015,12674,5000),"NEU-CLS-64":(1260,270,270)}
  for directory,counts in expected.items():
   sets=[]
   for split,count in zip(("train","val","test"),counts):
    with (DATASET_DIR/directory/"metadata"/f"{split}_manifest.csv").open() as f: rows=list(csv.DictReader(f))
    self.assertEqual(len(rows),count); sets.append({(x["relative_path"],x["source_index"]) for x in rows})
   self.assertFalse(sets[0]&sets[1]); self.assertFalse(sets[0]&sets[2]); self.assertFalse(sets[1]&sets[2])
 def test_dataset_samples(self):
  for bundle_id in ("resnet50-cifar10","resnet50-imagenet100","resnet50-neucls64"):
   sample,label=build_dataset(get_bundle(bundle_id),"val")[0]; self.assertEqual(tuple(sample.shape),(3,224,224)); self.assertIsInstance(label,int)
 def test_segment_equivalence_cpu_resnet(self):
  bundle=get_bundle("resnet50-cifar10"); model=build_model(bundle).eval(); x=torch.randn(1,*bundle.input_shape)
  with torch.no_grad():
   expected=model(x); value=x
   for name in model.partition_segment_names(): value=model.execute_partition_segment(name,value)
  torch.testing.assert_close(value,expected)
if __name__=="__main__": unittest.main()
