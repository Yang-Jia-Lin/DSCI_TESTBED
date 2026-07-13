"""Multi-exit ViT-Base backed by timm."""
import torch.nn as nn
class MultiExitViT(nn.Module):
 model_name="vit_base_patch16_224.orig_in21k_ft_in1k"
 def __init__(self,bundle,*,pretrained=False):
  super().__init__()
  try: import timm
  except ImportError as e: raise ImportError("ViT-Base requires timm in DSCI") from e
  self.bundle=bundle; self.backbone=timm.create_model(self.model_name,pretrained=pretrained,num_classes=bundle.num_classes); dim=int(self.backbone.embed_dim)
  self.exit_heads=nn.ModuleDict({x.exit_id:nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,bundle.num_classes)) for x in bundle.exits}); self.exit_attach_points={x.exit_id:x.attach_point for x in bundle.exits}
 def _embed(self,x):
  x=self.backbone.patch_embed(x); x=self.backbone._pos_embed(x); x=self.backbone.patch_drop(x); return self.backbone.norm_pre(x)
 def partition_segment_names(self): return ["patch_embed",*[f"blocks.{i}" for i in range(len(self.backbone.blocks))],"norm","final_classifier"]
 def execute_partition_segment(self,name,x):
  if name=="patch_embed": return self._embed(x)
  if name.startswith("blocks."): return self.backbone.blocks[int(name.split(".")[1])](x)
  if name=="norm": return self.backbone.norm(x)[:,0]
  if name=="final_classifier": return self.backbone.head(x)
  raise ValueError(name)
 def resolve_partition_segment(self,name): return lambda x:self.execute_partition_segment(name,x)
 def classify_exit(self,exit_id,features): return self.exit_heads[exit_id](features[:,0] if features.dim()==3 else features)
 def forward_features(self,x):
  out={}; x=self._embed(x)
  for i,block in enumerate(self.backbone.blocks): x=block(x); out[f"blocks.{i}"]=x
  out["norm"]=self.backbone.norm(x)[:,0]; return out
 def forward(self,x,exit_id=None):
  f=self.forward_features(x); return self.classify_exit(exit_id,f[self.exit_attach_points[exit_id]]) if exit_id else self.backbone.head(f["norm"])
 def final_classifier_module(self): return self.backbone.head
 def final_classifier_name(self): return "backbone.head"
 def freeze_for_exit(self,exit_id=None):
  for p in self.parameters(): p.requires_grad=False
  head=self.final_classifier_module() if exit_id is None else self.exit_heads[exit_id]
  for p in head.parameters(): p.requires_grad=True
