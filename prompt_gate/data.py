"""Read masked manifest inputs separately from complete teacher targets."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

LANES=('visible','infrared')


def videos(manifest):
    groups={}
    for line in Path(manifest).read_text().splitlines():
        row=json.loads(line)
        groups.setdefault(row['sequence'],[]).append(row)
    return groups


def load_images(paths,root,size,device):
    result={}
    for lane,path in paths.items():
        if path is None:continue
        with Image.open(Path(root)/path) as image:
            array=np.array(image.convert('RGB' if lane=='visible' else 'L'),copy=True)
        if array.ndim==2:array=array[:,:,None]
        value=torch.from_numpy(array).permute(2,0,1)[None].to(device=device,dtype=torch.float32)/255.
        result[lane]=F.interpolate(value,size,mode='bilinear',align_corners=False)
    return result
