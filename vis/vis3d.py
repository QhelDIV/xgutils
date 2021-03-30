import numpy as np
import scipy
import matplotlib.pyplot as plt
from xgutils import *
from xgutils.vis import fresnelvis, visutil

def vis_VoxelXRay(voxel, axis=0, duration=10.):
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
        img = fresnelvis.renderMeshCloud({"vert":voxv,"face":voxf}, samples=8, **dflt_camera, axes=True)
        imgs.append(img)
    sysutil.imgarray2video("/studio/temp/xray.mp4", imgs, duration=duration)