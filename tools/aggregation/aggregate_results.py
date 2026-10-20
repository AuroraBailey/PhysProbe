#!/usr/bin/env python3
"""Aggregate evaluator CSVs using the manuscript's target-specific estimators.

No image evaluation or model inference is performed. Diagnostic directories use
the paper setting IDs directly; no result-directory remapping is applied.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np

REPO = Path(__file__).resolve().parents[2]
METRICS = {
 'albedo': ('mae','psnr','ssim','lpips'),
 'depth': ('absrel_ai','rmse_ai','mae_ai','delta1_ai','delta2_ai','delta05_ai','spearman_rho','kendall_tau','edge_detail_f1'),
 'normal': ('mean_angular','median_angular','acc_11_25','acc_22_5','acc_30'),
 'roughness': ('rmse','bin_bmae','region_macro_mae','ssim'),
 'metallic': ('mae','rb_mae','iou_at_05','ap'),
}
BINS = ('bin_0_02','bin_02_04','bin_04_06','bin_06_08','bin_08_10')
PAPER_API = {
 'albedo': {'Doubao_Seedream_5.0','GPT-Image-1.5','GPT-Image-2.0','Gemini_3_Pro_Image','Qwen-Image-2.0','WAN2.7-Image'},
 'depth': {'Doubao_Seedream_5.0','GPT-Image-1.5','GPT-Image-2.0','Gemini_3_Pro_Image','Qwen-Image-2.0','WAN2.7-Image'},
 'normal': {'Doubao_Seedream_5.0','GPT-Image-1.5','FireRed-Image-Edit'},
 'roughness': {'Doubao_Seedream_5.0','GPT-Image-1.5','GPT-Image-2.0','Gemini_3_Pro_Image'},
}
PAPER_SETTINGS = {
 'albedo': {'A0','A1','A2','A3','B0'},
 'depth': {'A0','A1','A2'},
 'normal': {'A0','A1','A2','A3'},
 'roughness': {'A0','A1','A2','A3'},
}
SLICES = {'normal_light','low_light','hdr','dark_heavy','highlight_heavy','mixed_temperature'}
MODELS = {
 'all_black':['All-zero prediction'],
 'GPT-Image-1.5':['GPT-Image-1.5'], 'GPT-Image-2.0':['GPT-Image-2.0'],
 'WAN2.7-Image':['Wan2.7-Image','WAN2.7-Image'],
 'Qwen-Image-2.0':['Qwen-Image-2.0'], 'Doubao_Seedream_5.0':['Doubao Seedream 5.0'],
 'Gemini_3_Pro_Image':['Gemini 3 Pro Image'], 'FLUX.1-Kontext-dev':['FLUX.1-Kontext-dev'],
 'FireRed-Image-Edit':['FireRed-Image-Edit'],'LongCat-Image-Edit':['LongCat-Image-Edit'],
 'Lotus':['Lotus'],'StableNormal':['StableNormal'],'Depth_anything3':['Depth Anything 3'],
 'DNF-Intrinsic':['DNF-Intrinsic'],'MVInverse':['MVInverse'],
 'Colorful_Diffusion_Intrinsic':['Colorful Diffuse Intrinsic','Colorful Diffusion Intrinsic'],
 'Intrinsic_Image_Diffusion':['Intrinsic Image Diffusion'],
}

def number(value):
 try: return float(value)
 except (ValueError, TypeError): return float('nan')

def canonical_scene(name):
 match=re.search(r'scene\d+_\d+',name)
 if match:return match.group()
 match=re.search(r'L3D[^/\\]+',name,re.I)
 if match:return match.group()
 parts=[p for p in Path(name).parts if p.lower() not in {'normal','albedo','roughness','relative_depth'}]
 return parts[0] if parts else name

def source_of(path):
 text=str(path).lower()
 if 'interiorverse' in text:return 'interiorverse'
 if 'openroom' in text:return 'openroomff'
 return 'pbr_rooms'

def setting_of(target, directory):
 """Read the target's paper setting ID without translating directory names."""
 name=directory.upper()
 return name if name in PAPER_SETTINGS.get(target,set()) else None

def exemplar_of(parts):
 for part in parts:
  name=part.upper()
  if name in {'E1','E2','E3'}:return name
 return 'none'

def read_rows(path,target):
 with path.open(encoding='utf-8-sig',newline='') as handle: rows=list(csv.DictReader(handle))
 if target=='normal':rows=[r for r in rows if r.get('strategy','fixed')=='fixed']
 return rows

def discover(root, audit):
 records=[]
 for split in ('1_mainaxis','2_stresstest','3_comparison'):
  for target in METRICS:
   folder=root/split/target
   if not folder.is_dir():continue
   filename=f'{target}_scene_metrics.csv' if target=='metallic' else f'{target}_metrics.csv'
   for path in sorted(folder.rglob(filename)):
    relative=path.relative_to(folder);parts=relative.parts
    if len(parts)<2:continue
    model=parts[0]
    if model not in MODELS:
     continue # review folders, baseline probes and bootstrap exports are not methods
    setting='official';exemplar=exemplar_of(parts)
    if split=='3_comparison':
     if target not in PAPER_API or model not in PAPER_API[target]:continue
     setting=setting_of(target,parts[1])
     if not setting:
      audit.append({'file':str(path),'status':'unsupported_setting','variant':parts[1]});continue
    if target=='normal' and model in PAPER_API['normal']:
     needs_example=split!='3_comparison' or setting in {'A0','A2'}
     if needs_example == (exemplar=='none'):
      audit.append({'file':str(path),'status':'unsupported_exemplar','setting':setting});continue
    source=source_of(path)
    slice_name=next((p for p in parts if p in SLICES),'all')
    rows=read_rows(path,target)
    audit.append({'file':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'rows':len(rows),'status':'loaded'})
    for row in rows:
     scene=row.get('scene_id') or row.get('scene') or canonical_scene(row.get('name',''))
     # OpenRooms stress rows sometimes encode render variant before scene ID.
     if source=='openroomff':scene=canonical_scene(scene)
     partition=next((p.rsplit('_',1)[-1] for p in parts if p.endswith(('_mainaxis','_stresstest'))),row.get('split','mainaxis'))
     records.append(dict(row, _target=target,_model=model,_setting=setting,
       _exemplar=exemplar,_source=source,_slice=slice_name,_partition=partition,
       _scene=scene,_split=split,_file=str(path)))
 return records

def pooled(rows,key,weight=None,power=1):
 valid=[(number(r.get(key)),number(r.get(weight,1)) if weight else 1.) for r in rows]
 valid=[(v,w) for v,w in valid if math.isfinite(v) and math.isfinite(w) and w>0]
 if not valid:return float('nan')
 return (math.fsum((v**power)*w for v,w in valid)/math.fsum(w for v,w in valid))**(1/power)

def estimate(rows,target,metric,diagnostic=False):
 if target=='roughness':
  if metric=='bin_bmae':
   values=[pooled(rows,b+'_mae',b+'_pixels') for b in BINS]
   values=[v for v in values if math.isfinite(v)]
   return math.fsum(values)/len(values) if values else float('nan')
  if metric=='region_macro_mae':return pooled(rows,metric,'region_count')
  if metric=='rmse' and diagnostic:
   partitions=defaultdict(list)
   for row in rows:partitions[row['_partition']].append(row)
   if set(partitions)!={'mainaxis','stresstest'}:return float('nan')
   return math.fsum(pooled(partitions[p],'rmse','valid_pixels',2) for p in partitions)/2
 # Albedo/Depth source balance; normal additionally averages independent exemplars.
 if target in ('albedo','depth','normal'):
  buckets=defaultdict(list)
  for row in rows:
   key=(row['_source'],row['_partition'] if diagnostic and target=='albedo' else '',row['_exemplar'] if target=='normal' else '')
   buckets[key].append(row)
  values=[pooled(bucket,metric) for bucket in buckets.values()]
  return math.fsum(values)/len(values) if values and all(map(math.isfinite,values)) else float('nan')
 return pooled(rows,metric)

def merge_records(records, audit):
 """Reject overlapping exports rather than double-counting images/slices."""
 seen={};result=[]
 for row in records:
  key=tuple(row[k] for k in ('_split','_target','_model','_setting','_exemplar','_source','_slice','_partition'))+(row.get('name') or row['_scene'],)
  if key in seen:
   raise ValueError(f'Duplicate scoring entry: {key}; {seen[key]} and {row["_file"]}. Select variants explicitly.')
  seen[key]=row['_file'];result.append(row)
 return result

def image_identity(row):
 name=row.get('name','')
 scene=canonical_scene(name)
 variant=re.search(r'main(?:DiffLight|DiffMat)?_xml1?',name)
 view=re.match(r'\d+',Path(name).name)
 return (row['_source'],scene,variant.group() if variant else '',view.group() if view else Path(name).stem)

def recorded_stress_labels(records):
 """Read per-image labels from explicit slice exports."""
 labels={}
 for row in records:
  if row['_split']!='2_stresstest' or row['_slice']=='all' or row['_source']=='pbr_rooms':continue
  identity=image_identity(row)
  if identity in labels and labels[identity]!=row['_slice']:
   raise ValueError(f'Conflicting image-level stress assignments: {identity}')
  labels[identity]=row['_slice']
 return labels

def groups(records, stress_labels, recorded_labels=None):
 result=defaultdict(list)
 for row in records:
  if row['_split']=='2_stresstest' and row['_slice']=='all' and row['_source']!='pbr_rooms':
   label=(recorded_labels or {}).get(image_identity(row)) or stress_labels.get((row['_source'],row['_scene']))
   row=dict(row,_slice=label or 'unassigned')
  base=(row['_split'],row['_target'],row['_model'],row['_setting'])
  exemplar=row['_exemplar']
  result[base+(exemplar,row['_slice'],row['_source'])].append(row)
  if row['_source']!='pbr_rooms':result[base+(exemplar,row['_slice'],'source_balanced')].append(row)
 # Three independent exemplars are averaged as scores, never fused as predictions.
 for key,rows in list(result.items()):
  if key[1]=='normal' and key[4]!='none':result[key[:4]+('Avg',)+key[5:]].extend(rows)
 return result

def cluster_ci(rows,target,metric,resamples,seed,draw_store=None):
 """Shared deterministic scene draws; retain all images and all exemplars per scene."""
 sources=sorted({r['_source'] for r in rows},key=lambda s: ('openroomff','interiorverse','pbr_rooms').index(s));replicates=[]
 # Recreate the same indices for every method/metric sharing a scene support.
 rng=np.random.default_rng(seed)
 draws={}
 for source in sources:
  scenes=sorted({r['_scene'] for r in rows if r['_source']==source})
  draws[source]=(scenes,rng.integers(0,len(scenes),size=(resamples,len(scenes))))
  if draw_store is not None:
   name=target+'_'+source+'_'+hashlib.sha256(json.dumps(scenes).encode()).hexdigest()[:12]
   draw_store[name]=(scenes,draws[source][1])
 # Vectorized scene sufficient statistics for each source/exemplar bucket.
 values=[]
 for source in sources:
  scenes,indices=draws[source]
  exemplars=sorted({r['_exemplar'] for r in rows if r['_source']==source})
  for exemplar in exemplars:
   subset=[r for r in rows if r['_source']==source and r['_exemplar']==exemplar]
   nums=np.array([math.fsum(number(r[metric]) for r in subset if r['_scene']==scene and math.isfinite(number(r.get(metric)))) for scene in scenes])
   counts=np.array([sum(r['_scene']==scene and math.isfinite(number(r.get(metric))) for r in subset) for scene in scenes])
   den=counts[indices].sum(axis=1)
   if np.any(den==0):return None
   values.append(nums[indices].sum(axis=1)/den)
 boot=np.mean(values,axis=0)
 return [float(v) for v in np.quantile(boot,[.025,.975],method='linear')]

def expected_counts(labels):
 counts=defaultdict(int)
 for split in ('mainaxis','stresstest','ablation'):
  for target in METRICS:
   path=REPO/f'data/manifests/{split}/{target}_manifest_{split}.csv'
   if not path.is_file():continue
   for row in csv.DictReader(path.open(encoding='utf-8-sig')):
    source=row['datasource'] if row['datasource'] in {'openroomff','interiorverse'} else 'pbr_rooms'
    slice_name=labels.get((source,row['scene_id']),'all') if split=='stresstest' else 'all'
    if source=='pbr_rooms' and split=='stresstest':
     light=re.search(r'_L([0-5])',row['rgb_path'])
     slice_name=('normal_light','low_light','hdr','dark_heavy','highlight_heavy','mixed_temperature')[int(light.group(1))] if light else 'all'
    counts[(split,target,source,slice_name)]+=1
 return counts

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--evaluation-root',type=Path,required=True,help='Parent of 1_mainaxis, 2_stresstest and 3_comparison.')
 p.add_argument('--output-dir',type=Path,required=True)
 p.add_argument('--stress-assignment-mode',choices=('recorded','current'),default='recorded',
                help='Stress labels: recorded per-image slice exports (default), or current release scene metadata.')
 p.add_argument('--resamples',type=int,default=1000);p.add_argument('--seed',type=int,default=42)
 a=p.parse_args(argv)
 a.evaluation_root=a.evaluation_root.resolve()
 if a.resamples<0:p.error('--resamples must be nonnegative')
 if not a.evaluation_root.is_dir():p.error('Evaluation root does not exist')
 a.output_dir.mkdir(parents=True,exist_ok=True)
 config={'stress_assignment_mode':a.stress_assignment_mode};audit=[]
 rows=merge_records(discover(a.evaluation_root,audit),audit)
 labels={}
 with (REPO/'data/metadata/stress_assignments/scene_stress_slice.csv').open(encoding='utf-8-sig') as f:
  for r in csv.DictReader(f):labels[(r['datasource'],r['scene_id'])]=r['stress_slice']
 recorded_labels=recorded_stress_labels(rows) if a.stress_assignment_mode=='recorded' else {}
 counts=expected_counts(labels)
 output=[];draw_store={}
 for key,subset in sorted(groups(rows,labels,recorded_labels).items()):
  split,target,model,setting,exemplar,slice_name,source=key
  if source=='source_balanced' and {r['_source'] for r in subset}!={'openroomff','interiorverse'}:continue
  if exemplar=='Avg' and {r['_exemplar'] for r in subset}!={'E1','E2','E3'}:
   audit.append({'status':'incomplete_exemplar_set','group':key});continue
  item=dict(split=split,target=target,model=model,setting=setting,exemplar=exemplar,slice=slice_name,source=source,entries=len(subset),scenes=len({(r['_source'],r['_scene']) for r in subset}))
  support=defaultdict(int)
  for row in subset:support[(row['_source'],row['_exemplar'])]+=int(number(row.get('num_images',1)))
  partition={'1_mainaxis':'mainaxis','2_stresstest':'stresstest','3_comparison':'ablation'}[split]
  complete=all(counts.get((partition,target,src,slice_name),0)==n for (src,e),n in support.items())
  item['support_complete_by_manifest_counts']=complete
  if split=='2_stresstest' and target!='metallic' and recorded_labels:
   recorded_counts=defaultdict(int)
   for identity,label in recorded_labels.items():
    if label==slice_name:recorded_counts[identity[0]]+=1
   item['support_complete_by_recorded_counts']=all(recorded_counts[src]==n for (src,e),n in support.items())

  if not complete:audit.append({'status':'incomplete_or_different_manifest_counts','group':key,'observed':{str(k):v for k,v in support.items()}})

  diagnostic=split=='3_comparison'
  for metric in METRICS[target]:
   value=estimate(subset,target,metric,diagnostic)
   item[metric]=value if math.isfinite(value) else None
  output.append(item)
  # Main-table scalar CIs use the estimators and resampling protocol in the paper.
  primary={'depth':'absrel_ai','normal':'acc_22_5','albedo':'mae','roughness':'rmse','metallic':'rb_mae'}[target]
  if split=='1_mainaxis' and a.resamples and (source=='source_balanced' or target in {'roughness','metallic'}):
   for metric in (METRICS[target] if target=='metallic' else (primary,)):
    ci=cluster_ci(subset,target,metric,a.resamples,a.seed,draw_store)
    if ci:item[metric+'_ci95_lower'],item[metric+'_ci95_upper']=ci
 def save_csv(name,items):
  if not items:
   (a.output_dir/name).unlink(missing_ok=True)
   return
  fields=list(dict.fromkeys(k for item in items for k in item))
  with (a.output_dir/name).open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(items)
 if draw_store:
  np.savez_compressed(a.output_dir/'scene_resampling_indices.npz',**{name:entry[1] for name,entry in draw_store.items()})
  (a.output_dir/'bootstrap_scene_ids.json').write_text(json.dumps({name:entry[0] for name,entry in draw_store.items()},indent=2)+'\n')


 recorded_export=[dict(datasource=key[0],scene_id=key[1],render_variant=key[2],view_id=key[3],stress_slice=label) for key,label in sorted(recorded_labels.items())]
 save_csv('recorded_stress_assignments.csv',recorded_export)
 save_csv('tables.csv',output)

 (a.output_dir/'tables.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
 def public_value(value):
  if isinstance(value, dict):return {key:public_value(item) for key,item in value.items()}
  if isinstance(value, (list,tuple)):return [public_value(item) for item in value]
  if isinstance(value, str) and Path(value).is_absolute():
   path=Path(value)
   try:return path.relative_to(a.evaluation_root.resolve()).as_posix()
   except ValueError:
    try:return path.relative_to(REPO).as_posix()
    except ValueError:return path.name
  return value
 (a.output_dir/'audit.json').write_text(json.dumps(dict(files=public_value(audit),config=public_value(config),resamples=a.resamples,seed=a.seed,numpy_version=np.__version__,quantile_method='linear',notes=['Aggregation reads evaluator CSVs without re-evaluating image pixels.','Threshold or generation changes require new evaluator outputs.','Undefined/nonfinite metrics remain null; never zero-filled.']),indent=2)+'\n')
 print(json.dumps({'table_rows':len(output),'unsupported_settings':sum(r['status']=='unsupported_setting' for r in audit)},indent=2))

if __name__=='__main__':main()
