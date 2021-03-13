import numpy as np
import fresnel
import matplotlib.pyplot as plt
import math

import visutil as vis

def addAxes(scene, radius=[0.01,0.01,0.01]):
    axs = fresnel.geometry.Cylinder(scene, N=3)
    axs.material = fresnel.material.Material(solid=1.)
    axs.material.primitive_color_mix = 1.0
    axs.points[:] = [[[0,0,0],[1,0,0]],
                        [[0,0,0],[0,1,0]],
                        [[0,0,0],[0,0,1]]]
    axs.radius[:] = radius                 
    axs.color[:] =  [[[1,0,0],[1,0,0]],
                        [[0,1,0],[0,1,0]],
                        [[0,0,1],[0,0,1]]]

def renderMeshCloud(    mesh=None, mesh_outline_width=None, meshflat=False,  # mesh settings
                        cloud=None, cloudR=0.006, cloudC=None,  # pc settings
                        camPos=None, camLookat=None, camUp=np.array([0,0,1]), camHeight=1.,  # camera settings
                        samples=8, axes=False, resolution=(1024,1024),  # render settings
                        **kwargs):
    device = fresnel.Device()

    scene = fresnel.Scene(device)
    if mesh is not None:
        mesh = fresnel.geometry.Mesh(scene,vertices=mesh['vert'][mesh['face']].reshape(-1,3) ,N=1)
        mesh.material = fresnel.material.Material(color=fresnel.color.linear([0.7,0.7,0.7]), 
                                                    roughness=0.3,
                                                    specular=1.,
                                                    spec_trans=0.)
        if mesh_outline_width is not None:
            mesh.outline_width = mesh_outline_width
    if cloud is not None:
        cloud = fresnel.geometry.Sphere(scene, position = cloud, radius=cloudR)
        solid = .7 if mesh is not None else 0.
        cloud.material = fresnel.material.Material(solid=solid, \
                                                    color=fresnel.color.linear([1,0.0,0]),\
                                                    roughness=1.0,
                                                    specular=0.0)
        if cloudC is not None:
            #cloudC = vis.rescale(cloudC, cloudC)
            cloud.material.primitive_color_mix = 1.0
            cloud.color[:] = fresnel.color.linear(plt.cm.plasma(cloudC)[:,:3])
    if axes == True:
       addAxes(scene)
    if camPos is None or camLookat is None:
        scene.camera = fresnel.camera.fit(scene,margin=0)
    else:
        scene.camera = fresnel.camera.orthographic(camPos, camLookat, camUp, camHeight)
    scene.lights = fresnel.light.cloudy()
    #scene.lights = fresnel.light.rembrandt()
    #scene.lights = fresnel.light.lightbox()
    #scene.lights = fresnel.light.loop()
    scene.lights = fresnel.light.butterfly()
    #scene.lights[0].theta = 3

    tracer = fresnel.tracer.Path(device=device, w=resolution[0], h=resolution[1])
    tracer.sample(scene, samples=samples, light_samples=8)
    #tracer.resize(w=450, h=450)
    #tracer.aa_level = 3
    image = tracer.render(scene)[:]
    return image