import numpy as np
import fresnel
import matplotlib.pyplot as plt
import math
import copy

dflt_camera = dict(camPos=np.array([2,2,2]), camLookat=np.array([0.,0.,0.]),\
    camUp=np.array([0,1,0]),camHeight=2.414, fit_camera=False, \
    light_samples=16, samples=32, resolution=(256,256))
gold_color = np.array([253, 204, 134])/256
gray_color = np.array([0.7, 0.7, 0.7])
red_color  = np.array([1.,  0.,  0.])

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
def addBBox(scene, bb_min=np.array([-1,-1,-1.]), bb_max=np.array([1,1,1.]), color=red_color, radius=0.005, solid=1.):
    axs = fresnel.geometry.Cylinder(scene, N=12)
    axs.material = fresnel.material.Material(   color = fresnel.color.linear(color),
                                                solid=solid,
                                                spec_trans=.4)
    #axs.material.primitive_color_mix = 1.0
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
    axs.radius[:] = radius
    axs.color[:] =  [ [[.5,0,0],[.5,0,0]] ] * 12
def addBox(scene, center, spec=(1,1,1), color=gray_color, solid=0., **kwargs):
    X, Y, Z = spec[0], spec[1], spec[2]
    poly_info = fresnel.util.convex_polyhedron_from_vertices([
        [-X, -Y, -Z],
        [-X, -Y, Z],
        [-X, Y, -Z],
        [-X, Y, Z],
        [X, -Y, -Z],
        [X, -Y, Z],
        [X, Y, -Z],
        [X, Y, Z],
    ])
    geometry = fresnel.geometry.ConvexPolyhedron(scene,
                                             poly_info,
                                             position=center,
                                             outline_width=0)#0.015)
    geometry.material = fresnel.material.Material( \
                                roughness=1.0,
                                specular=0.)
    #if len(color)!=3:
    geometry.material.primitive_color_mix = 1.0
    geometry.color[:] = color
    # geometry.outline_material = fresnel.material.Material( \
    #                             color=(0.95, 0.93, 0.88),
    #                             roughness=0.1,
    #                             metal=1.0)


def renderMeshCloud(    mesh=None, meshC=gray_color, mesh_outline_width=None, meshflat=False,  # mesh settings
                        cloud=None, cloudR=0.006, cloudC=None,  # pc settings
                        camPos=None, camLookat=None, camUp=np.array([0,0,1]), camHeight=1.,  # camera settings
                        samples=32, axes=False, bbox=False, resolution=(1024,1024),  # render settings
                        lights="rembrandt", **kwargs):
    device = fresnel.Device()
    scene = fresnel.Scene(device)
    if mesh is not None and mesh['vert'].shape[0]>0:
        mesh = fresnel.geometry.Mesh(scene,vertices=mesh['vert'][mesh['face']].reshape(-1,3) ,N=1)
        mesh.material = fresnel.material.Material(color=fresnel.color.linear(meshC),
                                                    roughness=0.3,
                                                    specular=1.,
                                                    spec_trans=0.)
        if mesh_outline_width is not None:
            mesh.outline_width = mesh_outline_width
    if cloud is not None and cloud.shape[0]>0:
        cloud = fresnel.geometry.Sphere(scene, position = cloud, radius=cloudR)
        solid = .7 if mesh is not None else 0.
        cloud_flat_color = gold_color
        if cloudC is not None and len(cloudC)==3:
            cloud_flat_color = cloudC
        cloud.material = fresnel.material.Material(solid=solid, \
                                                    color=fresnel.color.linear(cloud_flat_color),\
                                                    roughness=1.0,
                                                    specular=0.0)
        if cloudC is not None and len(cloudC)!=3:
            cloud.material.primitive_color_mix = 1.0
            cloud.color[:] = fresnel.color.linear(plt.cm.plasma(cloudC)[:,:3])
    if axes == True:
        addAxes(scene)
    if bbox == True:
        addBBox(scene)
    if camPos is None or camLookat is None:
        print("Fitting")
        scene.camera = fresnel.camera.fit(scene,margin=0)
    else:
        scene.camera = fresnel.camera.orthographic(camPos, camLookat, camUp, camHeight)
    if lights == "cloudy":
        scene.lights = fresnel.light.cloudy()
    if lights == "rembrandt":
        scene.lights = fresnel.light.rembrandt()
    if lights == "lightbox":
        scene.lights = fresnel.light.lightbox()
    if lights == "loop":
        scene.lights = fresnel.light.loop()
    if lights == "butterfly":
        scene.lights = fresnel.light.butterfly()
    #scene.lights[0].theta = 3

    tracer = fresnel.tracer.Path(device=device, w=resolution[0], h=resolution[1])
    tracer.sample(scene, samples=samples, light_samples=8)
    #tracer.resize(w=450, h=450)
    #tracer.aa_level = 3
    image = tracer.render(scene)[:]
    return image
class FresnelRenderer():
    def __init__(self, **kwargs):
        self.setup_scene(**kwargs)
    def setup_scene(self, camera_kwargs={}, lights="rembrandt"):
        device = fresnel.Device()
        scene = fresnel.Scene(device)

        self.camera_opt = camera_opt = copy.deepcopy(dflt_camera)
        camera_opt.update(camera_kwargs)

        if camera_opt["fit_camera"]==True:
            print("Camera is not setup, now auto-fit camera")
            scene.camera = fresnel.camera.fit(scene,margin=0)
        else:
            camPos    = camera_opt["camPos"]
            camLookat = camera_opt["camLookat"]
            camUp     = camera_opt["camUp"]
            camHeight = camera_opt["camHeight"]
            scene.camera = fresnel.camera.orthographic(camPos, camLookat, camUp, camHeight)
        # setup lightings
        if lights == "cloudy":
            scene.lights = fresnel.light.cloudy()
        if lights == "rembrandt":
            scene.lights = fresnel.light.rembrandt()
        if lights == "lightbox":
            scene.lights = fresnel.light.lightbox()
        if lights == "loop":
            scene.lights = fresnel.light.loop()
        if lights == "butterfly":
            scene.lights = fresnel.light.butterfly()

        self.scene, self.device = scene, device
    def add_cloud(self, cloud, radius=0.006, color=None, solid=0., name=None):
        scene = self.scene
        cloud = fresnel.geometry.Sphere(scene, position = cloud, radius=radius)
        cloud_flat_color = gold_color
        if color is not None and len(color)==3:
            cloud_flat_color = color
        cloud.material = fresnel.material.Material(solid=solid, \
                                                    color=fresnel.color.linear(cloud_flat_color),\
                                                    roughness=1.0,
                                                    specular=0.0)
        if color is not None and len(color)!=3:
            cloud.material.primitive_color_mix = 1.0
            cloud.color[:] = fresnel.color.linear(plt.cm.plasma(color)[:,:3])
    def add_mesh(self, mesh, color = gray_color, outline_width=None, name=None):
        scene = self.scene
        mesh = fresnel.geometry.Mesh(scene,vertices=mesh['vert'][mesh['face']].reshape(-1,3) ,N=1)
        mesh.material = fresnel.material.Material(color=fresnel.color.linear(color),
                                                    roughness=0.3,
                                                    specular=1.,
                                                    spec_trans=0.)
        if outline_width is not None:
            mesh.outline_width = outline_width
        return self
    def add_bbox(self, *args, **kwargs):
        addBBox(self.scene, *args, **kwargs)
        return self
    def add_box(self, *args, **kwargs):
        addBox(self.scene, *args, **kwargs)
        return self
    def render(self):
        scene = self.scene
        resolution = self.camera_opt["resolution"]
        samples = self.camera_opt["samples"]
        light_samples = self.camera_opt["light_samples"]
        tracer = fresnel.tracer.Path(device=self.device, w=resolution[0], h=resolution[1])
        tracer.sample(scene, samples=samples, light_samples=light_samples)
        #tracer.resize(w=450, h=450)
        #tracer.aa_level = 3
        image = tracer.render(scene)[:]
        return image
#def plot_mesh_array(array, **kwargs):
    
