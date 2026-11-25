"""Fixed backward feature sampling; no learned correspondence or confidence."""
import torch
from torch.nn import functional as F


def warp(source,flow):
    height,width=source.shape[-2:]
    y,x=torch.meshgrid(torch.arange(height,device=source.device,dtype=source.dtype),
                       torch.arange(width,device=source.device,dtype=source.dtype),indexing='ij')
    sx,sy=x[None]+flow[:,0],y[None]+flow[:,1]
    grid=torch.stack((2*sx/(width-1)-1,2*sy/(height-1)-1),-1)
    support=((sx>=0)&(sx<=width-1)&(sy>=0)&(sy<=height-1))[:,None]
    return F.grid_sample(source,grid,padding_mode='border',align_corners=True),support


def compose_flow(first,second):
    return first+warp(second,first)[0]


def resize_flow(flow,size):
    height,width=flow.shape[-2:]
    value=F.interpolate(flow,size,mode='bilinear',align_corners=True)
    return value*value.new_tensor([(size[1]-1)/(width-1),(size[0]-1)/(height-1)])[None,:,None,None]
