import numpy as np
import fresnel
import matplotlib.pyplot as plt
import math

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
def addBBox(scene, bb_min=np.array([-1,-1,-1.]), bb_max=np.array([1,1,1.])):
    axs = fresnel.geometry.Cylinder(scene, N=12)
    axs.material = fresnel.material.Material(solid=1.)
    axs.material.primitive_color_mix = 1.0
    pts = []
    xi,yi,zi = bb_min
    xa,ya,za = bb_max
    axs.points[:] = [   [[xi,yi,zi],[xa,yi,zi]],
                        [[xi,yi,zi],[xi,ya,zi]],
                        [[xi,yi,zi],[xi,yi,za]], # 
                        [[xi,ya,za],[xa,ya,za]],
                        [[xi,ya,za],[xi,yi,za]],
                        [[xi,ya,za],[xi,ya,zi]], #
                        [[xa,ya,zi],[xi,ya,zi]],
                        [[xa,ya,zi],[xa,yi,zi]],
                        [[xa,ya,zi],[xa,ya,za]], #
                        [[xa,yi,za],[xi,yi,za]], 
                        [[xa,yi,za],[xa,ya,za]],
                        [[xa,yi,za],[xa,yi,zi]], #
                    ]
    axs.radius[:] = 0.005
    axs.color[:] =  [ [[.5,0,0],[.5,0,0]] ] * 12
dflt_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
    camUp=np.array([0,1,0]),camHeight=2.414,resolution=(128,128))
def renderMeshCloud(    mesh=None, mesh_outline_width=None, meshflat=False,  # mesh settings
                        cloud=None, cloudR=0.006, cloudC=None,  # pc settings
                        camPos=None, camLookat=None, camUp=np.array([0,0,1]), camHeight=1.,  # camera settings
                        samples=8, axes=False, bbox=False, resolution=(1024,1024),  # render settings
                        **kwargs):
    device = fresnel.Device()

    scene = fresnel.Scene(device)
    if mesh is not None and mesh['vert'].shape[0]>0:
        mesh = fresnel.geometry.Mesh(scene,vertices=mesh['vert'][mesh['face']].reshape(-1,3) ,N=1)
        mesh.material = fresnel.material.Material(color=fresnel.color.linear([0.7,0.7,0.7]), 
                                                    roughness=0.3,
                                                    specular=1.,
                                                    spec_trans=0.)
        if mesh_outline_width is not None:
            mesh.outline_width = mesh_outline_width
    if cloud is not None and cloud.shape[0]>0:
        cloud = fresnel.geometry.Sphere(scene, position = cloud, radius=cloudR)
        solid = .7 if mesh is not None else 0.
        cloud.material = fresnel.material.Material(solid=solid, \
                                                    color=fresnel.color.linear([1,0.0,0]),\
                                                    roughness=1.0,
                                                    specular=0.0)
        if cloudC is not None:
            cloud.material.primitive_color_mix = 1.0

            cloud.color[:] = fresnel.color.linear(plt.cm.plasma(cloudC)[:,:3])
    if axes == True:
        addAxes(scene)
    if bbox == True:
        addBBox(scene)
    if camPos is None or camLookat is None:
        scene.camera = fresnel.camera.fit(scene,margin=0)
    else:
        scene.camera = fresnel.camera.orthographic(camPos, camLookat, camUp, camHeight)
    scene.lights = fresnel.light.cloudy()
    scene.lights = fresnel.light.rembrandt()
    #scene.lights = fresnel.light.lightbox()
    #scene.lights = fresnel.light.loop()
    #scene.lights = fresnel.light.butterfly()
    #scene.lights[0].theta = 3

    tracer = fresnel.tracer.Path(device=device, w=resolution[0], h=resolution[1])
    tracer.sample(scene, samples=samples, light_samples=8)
    #tracer.resize(w=450, h=450)
    #tracer.aa_level = 3
    image = tracer.render(scene)[:]
    return image

#def plot_mesh_array(array, **kwargs):
    
