import re
import os
import sys
import igl
import mcubes
import numpy as np

from skimage.measure import find_contours
from skimage import morphology
from xgutils import nputil, ptutil, sysutil
def length(x):
    return np.linalg.norm(x)
def point2lineDistance(q, p1, p2):
    d = np.linalg.norm(np.cross(p2-p1, p1-q))/np.linalg.norm(p2-p1)
    return d
def get2DRotMat(theta=90, mode='degree'):
    if mode == 'degree':
        theta = np.radians(theta)
    return np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
def pointSegDistance(q, p1, p2):
    line_vec = p2-p1
    pnt_vec = q-p1
    line_len = np.linalg.norm(line_vec)
    line_unitvec = normalize(line_vec)
    pnt_vec_scaled = pnt_vec * 1.0/line_len
    t = np.dot(line_unitvec, pnt_vec_scaled)
    if t < 0.0:
        t = 0.0
    elif t > 1.0:
        t = 1.0
    nearest = line_vec * t
    dist = length(nearest - pnt_vec)
    nearest = nearest + p1
    return (dist, nearest)

def sampleTriangle(vertices, sampleNum=10, noVert=False):
    # vertices: numpy array of 
    if noVert == False:
        rd_a, rd_b = np.random.rand(sampleNum-3), np.random.rand(sampleNum-3)
    else:
        rd_a, rd_b = np.random.rand(sampleNum), np.random.rand(sampleNum)
    larger_than_1 = (rd_a + rd_b > 1.)
    rd_a[larger_than_1] = 1 - rd_a[larger_than_1]
    rd_b[larger_than_1] = 1 - rd_b[larger_than_1]
    if noVert == False:
        rd_a = np.r_[0,1,0,rd_a]
        rd_b = np.r_[0,0,1,rd_b]
    samples = np.array([vertices[0] + rd_a[i]*(vertices[1]-vertices[0]) + rd_b[i]*(vertices[2]-vertices[0]) \
                            for i in range(sampleNum)])
    return samples
def randQuat(N=1):
    #Generates uniform random quaternions
    #James J. Kuffner 2004 
    #A random array 3xN
    s = np.random.rand(3,N)
    sigma1 = np.sqrt(1.0 - s[0])
    sigma2 = np.sqrt(s[0])
    theta1 = 2*np.pi*s[1]
    theta2 = 2*np.pi*s[2]
    w = np.cos(theta2)*sigma2
    x = np.sin(theta1)*sigma1
    y = np.cos(theta1)*sigma1
    z = np.sin(theta2)*sigma2
    return np.array([w, x, y, z])
def multQuat(Q1,Q2):
    # https://stackoverflow.com/a/38982314/5079705
    w0,x0,y0,z0 = Q1   # unpack
    w1,x1,y1,z1 = Q2
    return([-x1*x0 - y1*y0 - z1*z0 + w1*w0, x1*w0 + y1*z0 - z1*y0 +
    w1*x0, -x1*z0 + y1*w0 + z1*x0 + w1*y0, x1*y0 - y1*x0 + z1*w0 +
    w1*z0])
def conjugateQuat(Q):
    return np.array([Q[0],-Q[1],-Q[2],-Q[3]])
def applyQuat(V, Q):
    P = np.array([0., V[0], V[1], V[2]])
    nP = multQuat(Q, multQuat(P, conjugateQuat(Q)) )
    return nP[1:4]
def fibonacci_sphere(samples=1000):
    rnd = 1.

    points = []
    offset = 2./samples
    increment = np.pi * (3. - np.sqrt(5.));

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2);
        r = np.sqrt(1 - np.power(y,2))

        phi = ((i + rnd) % samples) * increment

        x = np.cos(phi) * r
        z = np.sin(phi) * r

        points.append([x,y,z])

    return points
# mesh operation
def mergeMeshes(meshes):
    verts = []
    faces = []
    total_vert_num = 0
    for mesh in meshes:
        verts.append(mesh["vert"])
        faces.append(mesh["face"]+total_vert_num)
        if len(faces[-1].shape)==1:
            faces[-1] = faces[-1].reshape(1,-1)
        total_vert_num += len(verts[-1])
        assert len(verts[-1].shape)==2 and len(faces[-1].shape)==2, "Invalid mesh!" 
    verts = np.concatenate(verts)
    faces = np.concatenate(faces)
    return {"vert":verts, "face":faces}
def filterMesh(vert, face, filterV):
    v_keep_mask = filterV
    v_del_mask  = 1-filterV
    
    newIndV = np.zeros(vert.shape[0])-1
    newIndV[v_keep_ind] = np.arange(v_keep_ind.sum()).astype(int)
    nv = vert[v_keep_ind]
    f_keep_mask = (v_keep_mask[face].sum(axis=-1)==3)
    nf = face[f_keep_mask]
    nf = newIndV[nf]
    assert (nf>0).all(), "New face list contains removed vertices" 
    return nv, nf
    # TODO filterF
# reconstruction & sampling
def array2mesh(array, thresh=0., dim=3, coords=None, bbox=np.array([[-1,-1],[1,1]]), return_coords=False, \
                if_decimate=False, decimate_face=4096, cart_coord=True):
    """from 1-D array to 3D mesh

    Args:
        array (np.ndarray): 1-D array
        thresh (float, optional): threshold. Defaults to 0..
        dim (int, optional): 2 or 3, curve or mesh. Defaults to 3.
        coords (np.ndarray, optional): array's coordinates (num_points, x_dim). Defaults to None.
        bbox (np.ndarray, optional): bounding box of coords. Defaults to np.array([[-1,-1],[1,1]]).
        return_coords (bool, optional): whether return the coords. Defaults to False.
        decimate_face (int, optional): whether to simplify the mesh. Defaults to 4096.
        cart_coord (bool, optional): cartesian coordinate in array form, x->i, y->j,... and all varibles increases monotonically. Defaults to True.

    Returns:
        tuple: `verts`, `faces`, `coords` or `verts`, `faces` according to `return_coords`
    """
    grid = nputil.array2NDCube(array, N=dim)
    if   dim==3:
        verts, faces = mcubes.marching_cubes(grid, thresh)
        if cart_coord == False:
            verts = verts[:,[1,0,2]]
        verts = verts/(grid.shape[0]-1) # rearrange order and rescale
    elif dim==2:
        contours = find_contours(grid, thresh)
        vcount, points, edges = 0, [], []
        for contour in contours:
            ind = np.arange( len(contour) )
            points.append(contour)
            edges.append( np.c_[vcount+ind, vcount+(ind+1)%len(contour)] )
            vcount += len(contour)
        if len(contours) == 0:
            return None, None
        verts = np.concatenate(points, axis=0)[:,[1,0]] / (grid.shape[0]-1)
        #verts = verts[:,[1,0]]
        faces  = np.concatenate(edges,  axis=0)
        #levelset_samples = igl.sample_edges(points, edges, 10)
    if coords is not None:
        bbmin, bbmax = nputil.arrayBBox(coords)
    else:
        bbmin, bbmax = bbox
        coords = nputil.makeGrid(bb_min=bbmin, bb_max=bbmax, shape=grid.shape)
    verts = verts*(bbmax-bbmin) + bbmin
    verts, faces = verts, faces.astype(int)
    if if_decimate==True:
        if dim!=3:
            print("Warning! decimation only works for 3D")
        elif faces.shape[0]>decimate_face: # Only decimate when appropriate
            reached, verts, faces, _, _ = igl.decimate(verts, faces, decimate_face)
            faces = faces.astype(int)
    if return_coords==True:
        return verts, faces, coords
    else:
        return verts, faces
def array2curve(array, thresh=0., coords=None):
    pass
def sampleMesh(vert, face, sampleN):
    sampled = None
    if vert.shape[-1]==3:
        resample = True
        while resample:
            try:
                B,FI    = igl.random_points_on_mesh(sampleN, vert, face)
                sampled =   B[:,0:1]*vert[face[FI,0]] + \
                            B[:,1:2]*vert[face[FI,1]] + \
                            B[:,2:3]*vert[face[FI,2]]
                resample=False
                if sampled.shape[0] != sampleN:
                    print('Failed to sample "sampleN" points, now resampling...', file=sys.__stdout__)
                    resample=True
            except Exception as e:
                print('Error encountered during mesh sampling:', e, file=sys.__stdout__)
                print('Now resampling...', file=sys.__stdout__)
                resample = True
    elif vert.shape[-1]==2:
        edge = face
        fac = 2 * np.ceil(sampleN / vert.shape[0]).astype(int)
        sampled = igl.sample_edges(vert, edge, fac)
        choice = np.random.choice(sampled.shape[0], sampleN, replace=False)
        sampled = sampled[choice]

    return sampled
sampleShape = sampleMesh

# geometry
def signed_distance(queries, vert, face): # remove NAN's
    S, I, C = igl.signed_distance(queries, vert,face)
    if len(S.shape)==0:
        S = S.reshape(1)
    return np.nan_to_num(S), I, C
def shape2sdf(shapePath, shapeInd, gridDim=256, disturb=False):
    vert = H5Var(shapePath, 'vert')[shapeInd]
    face = H5Var(shapePath, 'face')[shapeInd]
    x = y = z = np.linspace(0,1,gridDim)
    grid = np.stack(np.meshgrid(x,y,z,sparse=False), axis=-1)
    all_samples = grid.reshape(-1,3)
    if disturb==True:
        disturbation = np.random.rand(all_samples.shape[0],3)/gridDim
        all_samples += disturbation
    S, I, C = signed_distance(all_samples, vert, face)
    sdfPairs = np.concatenate([all_samples,S[:,None]], axis=-1)
    return sdfPairs
def mesh2sdf(vert, face, gridDim=64, disturb=False):
    x = y = z = np.linspace(0,1,gridDim)
    grid = np.stack(np.meshgrid(x,y,z,sparse=False), axis=-1)
    all_samples = grid.reshape(-1,3)
    if disturb==True:
        disturbation = np.random.rand(all_samples.shape[0],3)/gridDim
        all_samples += disturbation
    S, I, C = signed_distance(all_samples, vert, face)
    sdfPairs = np.concatenate([all_samples,S[:,None]], axis=-1)
    return sdfPairs
def pc2sdf():
    #TODO
    pass

# open3d related
class Open3D_Toolbox():
    def __init__(self):
        import open3d as o3d
        self.o3d = o3d
    def poisson_recon(self, cloud, estimate_normals=True, depth=6, quantile=.3, knn=10):
        o3d=self.o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        #with o3d.utility.VerbosityContextManager(
        #    o3d.utility.VerbosityLevel.Debug) as cm:
        poi_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
        vertices_to_remove = densities<(np.quantile(densities, quantile))
        poi_mesh.remove_vertices_by_mask(vertices_to_remove)
        return np.asarray(poi_mesh.vertices), np.asarray(poi_mesh.triangles)
    def ball_pivoting(self, cloud, radii=[0.01, 0.02, 0.04], knn=30):
        o3d=self.o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))  # invalidate existing normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))

        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting( pcd, o3d.utility.DoubleVector(radii) )
        return np.asarray(rec_mesh.vertices), np.asarray(rec_mesh.triangles)

class Meshlab_Toolbox():
    def __init__(self):
        import pymeshlab
        self.mlab = pymeshlab
    def poisson_recon(self, cloud, estimate_normals=True, depth=6, fulldepth=4, knn=10):
        pymeshlab = self.mlab
        temp_dir = os.path.expanduser('~/.temp/meshlab/')
        sysutil.mkdirs(temp_dir)
        cloud_path = os.path.join(temp_dir, "cloud.pts")
        recon_path = os.path.join(temp_dir, "recon.ply")

        np.savetxt(cloud_path, cloud)

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(cloud_path)
        ms.compute_normals_for_point_sets(k=20)
        ms.surface_reconstruction_screened_poisson(depth=depth, fulldepth=fulldepth)
        ms.save_current_mesh(recon_path)
        vert, face = igl.read_triangle_mesh(recon_path)
        return vert, face

# grid sampling (inefficient)
def shapes2sdfs(shapePath, sdfPath, indices=np.arange(10), gridDim=256, disturb=False):
    #shapeDict = readh5(shapePath)
    #verts, faces = shapeDict['vert'], shapeDict['face']
    if os.path.exists(sdfPath):
        if '.h5' == sdfPath[-3:]:
            os.remove(sdfPath)
        else:
            raise ValueError('sdfPath must ended with .h5')
    args = [indices]
    func = lambda index:shape2sdf(shapePath, index,gridDim=gridDim, disturb=disturb)
    batchFunc=lambda batchOut: [H5Var(sdfPath, 'SDF').append(np.array(batchOut))]
    ret = np.array(parallelMap(func, args, zippedIn=False, batchFunc=batchFunc))[0]
    #print(ret.shape)
    #writeh5(sdfPath, {'SDF':sdf})
    return ret

from scipy.spatial import cKDTree
def points_dist(p1, p2):
    #from chamfer_distance import ChamferDistance
    #chamfer_dist = ChamferDistance()
    '''distance from p1 to p2'''
    #print(p1.shape, p2.shape)
    #d1, d2  = chamfer_dist( ptutil.np2th(p1), ptutil.np2th(p2) )
    tree = cKDTree(p2)
    dist, ind = tree.query(p1, k = 1)

    return dist
def points_sdf(targetx, sign, ref):
    dist = points_dist(targetx, ref)
    dist = np.sign(sign)*dist
    return dist
# voxelization
def morph_voxelization(vert, face, sampleN=1000000, grid_dim=128, selem_size=6):
    """ Morphological voxelization. Given arbitrary triangle soup, return the watertight voxelization of it.
        First sample cloud from mesh, voxelize the cloud, dilate, floodfill, erose. Note that dilate+erose=closing
    """
    vmin, vmax = np.abs(vert).min(), np.abs(vert).max()
    if vmax<1.:
        print(f"Warning: Mesh should be fallen into [-1,1]^3 bounding box! vmin:{vmin} vmax:{vmax}")
    samples = sampleMesh(vert, face, sampleN)
    voxel, coords = ptutil.ths2nps(ptutil.point2voxel(samples[None,...], grid_dim=grid_dim, ret_coords=True))
    voxel, coords = voxel[0], coords[0]
    
    selem = morphology.ball(selem_size)
    dilated = morphology.binary_dilation(voxel, selem)
    mask    = 1-morphology.flood(dilated, (0, 0, 0))
    erosed  = morphology.binary_erosion(mask, selem)
    water_tight_voxel = erosed
    return water_tight_voxel, coords
# coordinate transforms
def shapenetv1_to_shapenetv2(voxel):
    return np.flip(np.transpose(voxel, (2,1,0)),2).copy()
def shapenetv2_to_nnrecon(voxel):
    return np.flip(np.transpose(voxel, (1,0,2)),2).copy()
def shapenetv2_to_cart(voxel):
    return np.flip(voxel, 2).copy()
def nnrecon_to_cart(voxel):
    return np.flip(np.transpose(voxel, (2,1,0)),0).copy()
def cart_to_nnrecon(voxel):
    return np.flip(np.transpose(voxel, (1,0,2)),1).copy()

# SDF functions
def boxSDF(queries, spec, center=None):
    ''' queries: NxD array
        spec:    D array
        center:  D array
    '''
    if center is None:
        center = np.zeros(spec.shape)
    b = spec[None,...]
    c = center[None,...]
    queries -= c
    q = np.abs(queries) - b
    sd = q.max(axis=-1)
    #sd = sd*(sd>0)
    sd = np.linalg.norm(q*(q>0), axis=-1) + sd*(sd<0)
    return sd
def batchBoxSDF(queries, spec, center=None):
    ''' queries: NxD array
        spec:    MxD array
        center:  MxD array
        return:
            MxN array
    '''
    if center is None:
        center = np.zeros(spec.shape)
    b = spec[:,None,:]
    c = center[:,None,:]
    queries = queries[None,...] - c
    q = np.abs(queries) - b
    sd = q.max(axis=-1)
    #sd = sd*(sd>0)
    sd = np.linalg.norm(q*(q>0), axis=-1) + sd*(sd<0)
    return sd

# Obsolete
def extract_levelset(target_x=None, target_y=None, sampleN=256,**kwargs):
    print("Warning, extract_levelset is obsolete now! Use array2mesh & sampleMesh instead.")
    dim = target_x.shape[-1]
    if dim == 3:
        shape   = LevelsetVisual(opt=None).visualize( 
            target_y = target_y, 
            target_x = target_x,
            name = 'levelset')['shape']['levelset']
        vert, face = shape['vert'], shape['face']
        levelset_samples = sampleMesh(vert, face, sampleN)
    elif dim == 2:
        # TODO
        vert, edge = array2mesh(target_y, thresh=0., dim=2, coords=target_x)
        fac = 2 * np.ceil(sampleN / vert.shape[0]).astype(int)
        levelset_samples = igl.sample_edges(vert, edge, fac)
        levelset_samples = np.random.choice(levelset_samples.shape[0], sampleN, replace=False)
    return levelset_samples
