import torch
import numpy as np
import scipy
import matplotlib.pyplot as plt
from xgutils import *
from xgutils.vis import fresnelvis, visutil

def vis_VoxelXRay(voxel, axis=0, duration=10., target_path="/studio/temp/xray.mp4"):
    if(len(voxel.shape)==1):
        voxel = nputil.array2NDCube(voxel, N=3)
    imgs = []
    for i in sysutil.progbar(range(voxel.shape[axis])):
        nvox = voxel.copy()
        nvox[i:,:,:]=0
        voxv, voxf = geoutil.array2mesh(nvox.reshape(-1), thresh=.5, coords=nputil.makeGrid([-1,-1,-1],[1,1,1],[64,64,64], indexing="ij"))
        #voxv[]
        dflt_camera = fresnelvis.dflt_camera
        dflt_camera["camPos"]=np.array([2,2,2])
        dflt_camera["resolution"]=(256,256)
        img = fresnelvis.renderMeshCloud({"vert":voxv,"face":voxf}, **dflt_camera, axes=True)
        imgs.append(img)
    sysutil.imgarray2video(target_path, imgs, duration=duration)

def OctreePlot3D(tree, dim, depth, **kwargs):
    assert dim==2
    boxcenter, boxlen, tdepth = ptutil.ths2nps(ptutil.tree2bboxes(torch.from_numpy(tree), dim=dim, depth=depth))    
    maxdep = tdepth.max()
    renderer = fresnelvis.FresnelRenderer(camera_kwargs=dict(camPos=np.array([1.5,2,2]), resolution=(1024,1024)))#.add_mesh({"vert":vert, "face":face})
    for i in range(len(tdepth)):
        dep=tdepth[i]
        length = boxlen[i]
        bb_min = boxcenter[i]-boxlen[i]
        bb_max = boxcenter[i]+boxlen[i]
        lw=1+.5*np.exp(-dep)
        #rect = patches.Rectangle(corner, 2*length, 2*length, linewidth=lw, edgecolor=plt.cm.plasma(dep/maxdep), facecolor='none')
        renderer.add_bbox(bb_min=bb_min, bb_max=bb_max, color=plt.cm.plasma(dep/maxdep)[:3], radius=0.001*dep**1.5, solid=.0)
    img = renderer.render()
    return img

#def CloudPlot

def SparseVoxelPlot(sparse_voxel, depth=4, varying_color=False, camera_kwargs=dict(camPos=np.array([2,2,2]), resolution=(512,512))):
    resolution = camera_kwargs["resolution"]
    if len(sparse_voxel)==0:
        return np.zeros((resolution[0], resolution[1], 3))
    grid_dim = 2**depth
    box_len  = 2/grid_dim/2

    renderer = fresnelvis.FresnelRenderer(camera_kwargs=camera_kwargs)#.add_mesh({"vert":vert, "face":face})
    voxel_inds   = ptutil.unravel_index( torch.from_numpy(sparse_voxel), shape=(2**depth,)*3 )
    voxel_coords = ptutil.ths2nps(ptutil.index2point(voxel_inds, grid_dim=grid_dim))

    color = fresnelvis.gray_color
    percentage = np.arange(len(voxel_coords)) / len(voxel_coords)
    if varying_color==True:
        color = plt.cm.plasma(percentage)[...,:3]
    renderer.add_box(center=voxel_coords, spec=np.zeros((3))+box_len, color=color, solid=0.)
    img = renderer.render()
    return img
