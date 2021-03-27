import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from xgutils import nputil, sysutil
def sampleNDSphere(shape):
    """generate random sample on n-d sphere

    Args:
        shape ([...,dim] np.array): generate samples as shape, 
            the last dimension dim is the dimension of the sphere

    Returns:
        np.array: samples
    """    
    u = torch.randn(shape)
    d=(u**2).sum(-1,keepdim=True)**.5
    u = u/d
    return u
# Type conversions
def th2np(tensor):
    if type(tensor) is np.ndarray:
        return tensor
    if type(tensor) is torch.Tensor:
        return tensor.detach().cpu().numpy()
    if issubclass(type(tensor), torch.distributions.distribution.Distribution):
        return thdist2np(tensor)
    else:
        return tensor
def np2th(array, device='cuda'):
    tensor = array
    if type(array) is not torch.Tensor:
        tensor = torch.tensor(array).float()
    if type(tensor) is torch.Tensor:
        if device=='cuda':
            return tensor.cuda()
        return tensor.cpu()
    else:
        return array
def nps2ths(arrays, device="cuda"):
    if type(arrays) is dict:
        dic={}
        for key in arrays:
            array = arrays[key]
            dic[key] = np2th(array, device=device)
        return dic
    else:
        tensors = []
        for array in arrays:
            tensors.append(np2th(array, device=device))
        return tuple(tensors)
def ths2nps(tensors):
    if type(tensors) is dict:
        dic={}
        for key in tensors:
            if type(tensors[key]) is dict or type(tensors[key]) is list:
                dic[key] = ths2nps(tensors[key])
            else:
                dic[key] = th2np( tensors[key] )
        return dic
    elif type(tensors) is list or type(tensors) is tuple:
        arrays = []
        for tensor in tensors:
            if type(tensor) is dict or type(tensor) is list:
                arrays.append(ths2nps(tensor))
            else:
                arrays.append(th2np(tensor))
        return tuple(arrays)
    else:
        return th2np(tensors)
def th2device(tensor, device='cpu'): # ['cpu', 'cuda']
    if type(tensor) is torch.Tensor:
        return tensors.detach().cpu() if device=='cpu' else tensors.cuda()
    elif issubclass(type(tensor), torch.distributions.distribution.Distribution):
        if type(tensor) is torch.distributions.independent.Independent:
            reinterpreted_batch_ndims = tensor.reinterpreted_batch_ndims
            dist = th2device(tensor.base_dist)
            return torch.distributions.independent.Independent(dist, reinterpreted_batch_ndims )
        elif type(tensor) is torch.distributions.normal.Normal:
            loc, scale = ths2device([tensor.loc, tensor.scale])
            return torch.distributions.normal.Normal(loc=loc, scale=scale)
    else:
        raise TypeError(f'type {type(tensor)} is not supported')
def ths2device(tensors, device='cpu'):
    if type(tensors) is dict:
        tensors_device={}
        for key in tensors:
            tensor = tensors[key]
            tensors_device[key] = tensor.float().cuda()
        return tensors_device
    else:
        tensorsCUDA = [tensor.cuda() for tensor in tensors]
        return tuple(tensorsCUDA)
def ths2cuda(tensors):
    if type(tensors) is dict:
        tensorsCUDA={}
        for key in tensors:
            tensor = tensors[key]
            tensorsCUDA[key] = tensor.float().cuda()
        return tensorsCUDA
    else:
        tensorsCUDA = [tensor.cuda() for tensor in tensors]
        return tuple(tensorsCUDA)
def ths2cpu(tensors):
    pass
def batch_select(batch, index=0):
    if type(batch) is dict:
        return dict([(key,batch[key][index]) for key in batch.keys()])
    else:
        return batch
def thdist2np(dist):
    item = {'mean': th2np(dist.mean), \
            'variance':th2np(dist.variance), \
            'entropy':th2np(dist.entropy())}
    return item
def simple_gather(tensor, axis, ind):
    pass # TODO
    # ind = torch.tensor(ind,dtype=int).view(..,).expand(-1, -1, dim_x)
    # indY = torch.tensor(ind,dtype=int).unsqueeze(-1).expand(-1, -1, dim_y)
    # subX = X.gather(1, indX)
    # subY = Y.gather(1, indY)

def plot_grad_flow(named_parameters):
    ave_grads = []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
    plt.plot(ave_grads, alpha=0.3, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, linewidth=1, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(xmin=0, xmax=len(ave_grads))
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
def plot_grad_flow_v2(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([matplotlib.lines.Line2D([0], [0], color="c", lw=4),
                matplotlib.lines.Line2D([0], [0], color="b", lw=4),
                matplotlib.lines.Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def batch_dict(dict_list):
    keys = dict_list[0]
    dataDict = {}
    for key in keys:
        dataDict[key] = []
    for item in dict_list:
        for key in keys:
            dataDict[key].append( item[key] )
    for key in keys:
        dataDict[key] = np.array(dataDict[key])
    return dataDict
# dataset related
def dataset_to_h5(dset, outdir='~/temp/temp.h5'):
    item = dset[0]
    dict_list = []
    for i in sysutil.progbar( range(len(dset)) ):
        dict_list.append(dset[i])
    dataDict = batch_dict(dict_list)
    for key in dataDict.keys():
        if type(dataDict[key]) is torch.Tensor:
            dataDict[key] = th2np(dataDict[key])
    nputil.writeh5(outdir, dataDict)
def dataset_generator(dset, data_indices=[0,1,2], device="cuda"):
    for ind in data_indices:
        dataitem = dset.__getitem__(ind)
        batch = {}
        for key in dataitem:
            datakey = dataitem[key]
            if type(datakey) is not np.ndarray and type(datakey) is not torch.Tensor:
                continue
            datakey = dataitem[key][None,...]
            if type(datakey) is np.ndarray:
                datakey = torch.from_numpy(datakey)
            batch[key] = datakey.to(device)
        yield batch

# Fold and Unfold
def unfold_cube(tensor, last_dims=2, size=2, step=2, flatten=True):
    unfolded = tensor
    batch_shape= tensor.shape[:-last_dims]
    batch_dims = len(batch_shape)
    for di in range(last_dims):
        unfolded = unfolded.unfold(batch_dims+di, size=size, step=step)
    if flatten==True:
        total_size = np.array(tensor.shape[-last_dims:]).prod()
        unfold_size = unfolded.shape[-1]**last_dims
        unfolded = unfolded.reshape(*(unfolded.shape[:-2*last_dims]), total_size//unfold_size, unfold_size)
    return unfolded
def fold_cube(unfolded, N=3):
    batch_shape = unfolded.shape[:-2]
    batch_dims  = len(batch_shape)
    vox_dim = np.ceil( np.power(unfolded.shape[-1], 1./N) ).astype(int)
    unfolded = unfolded.reshape(*batch_shape,*((vox_dim,)*(2*N)))
    folded = unfolded
    for i in range(N):
        folded = torch.cat(torch.split(folded,1,dim=batch_dims+i), dim=batch_dims+N+i)
    for i in range(N):
        folded = torch.squeeze(folded, dim=batch_dims)

    return folded

def compress_voxels(voxel, packbits=True):
    assert(voxel.shape[-1]==256), "Only 256-> 16x16 dims is supported"
    divided = unfold_cube(torch.from_numpy(voxel), last_dims=3, size=16, step=16).numpy()
    empty   = (divided.sum(axis=-1)==0)
    full    = (divided.sum(axis=-1)==16**3)
    partial = np.logical_and(1-full, 1-empty)
    empty_idx, full_idx, partial_idx = np.where(empty)[0], np.where(full)[0], np.where(partial)[0]
    shape_vocab = np.zeros((1+1+len(partial_idx), 16*16*16), dtype=np.bool)
    vocab_idx   = np.zeros((16*16*16), dtype=np.int16)
    # 0: empty, 1: full, >1: various parts
    shape_vocab[1] = 1
    shape_vocab[2+np.arange(len(partial_idx))] = divided[partial_idx]
    vocab_idx[partial_idx] = 2+np.arange(len(partial_idx))
    vocab_idx[full_idx]    = 1
    #shape_vocab = shape_vocab.astype(bool)
    assert ((shape_vocab[vocab_idx] != divided).sum()==0), "Invalid compression"
    if packbits==True:
        shape_vocab = np.packbits(shape_vocab, axis=-1)
    return shape_vocab, vocab_idx # uint8, int16
def decompress_voxels(shape_vocab, vocab_idx, unpackbits=True):
    # 20x + faster than compress_voxels
    if unpackbits==True:
        shape_vocab = np.unpackbits(shape_vocab, axis=-1)
    unfolded = shape_vocab[vocab_idx]
    folded   = fold_cube(torch.from_numpy(unfolded), N=3).numpy()
    return folded

def fold_unittest():
    testth = torch.tensor([[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15.]])
    unfolded = unfold_cube(testth, size=2, step=2, last_dims=2)
    folded   = fold_cube(unfolded, N=2)
    assert (testth!=folded).sum()==0

    voxels = np.random.rand(256,256,256) > .5
    shape_vocab, vocab_idx = compress_voxels(voxels)
    print("shape_vocab, vocab_idx", shape_vocab.shape, shape_vocab.dtype, vocab_idx.shape, vocab_idx.dtype)
    decompress             = decompress_voxels(shape_vocab, vocab_idx)
    assert (voxels!=decompress).sum()==0
    print("All past")

# point to index and reverse
def ravel_index(tensor, shape):
    raveled = torch.zeros(*tensor.shape[:-1]).type_as(tensor)
    if tensor.shape[-1]==2:
        raveled = tensor[..., 0]*shape[1] + tensor[..., 1]
    elif tensor.shape[-1]==3:
        raveled = tensor[..., 0]*shape[1]*shape[2] + tensor[..., 1]*shape[2] + tensor[..., 2]
    else:
        raise ValueError("shape must be 2 or 3 dimensional")
    return raveled
def unravel_index(tensor, shape):
    unraveled = torch.zeros(*tensor.shape, len(shape)).type_as(tensor)
    if len(shape)==2:
        unraveled[..., 0] = tensor // shape[1]
        unraveled[..., 1] = tensor %  shape[1]
    elif len(shape)==3:
        s12 = shape[1]*shape[2]
        unraveled[..., 0] = tensor // s12
        unraveled[..., 1] = tensor %  s12 // shape[2]
        unraveled[..., 2] = tensor %  s12 %  shape[2]
    else:
        raise ValueError("shape must be 2 or 3 dimensional")
    return unraveled
def ravel_unittest():
    idx = np.arange(9)
    npunravel = np.array(np.unravel_index(idx, (3,3))).swapaxes(0,-1)
    unraveled = unravel_index(torch.from_numpy(idx)[None,...], (3,3))
    assert ( npunravel==(unraveled[0].numpy())).all(), print(npunravel,"\n",unraveled[0].numpy())
    raveled   = ravel_index(unraveled, (3,3))
    assert ( idx==(raveled[0].numpy())).all(), print(idx,"\n",raveled[0].numpy())
    
    idx = np.arange(27)
    shape = (3,3,3)
    npunravel = np.array(np.unravel_index(idx, shape)).swapaxes(0,-1)
    unraveled = unravel_index(torch.from_numpy(idx)[None,...], shape)
    assert ( npunravel==(unraveled[0].numpy())).all(), print(npunravel,"\n",unraveled[0].numpy())
    raveled   = ravel_index(unraveled, shape)
    assert ( idx==(raveled[0].numpy())).all(), print(idx,"\n",raveled[0].numpy())
    print(unraveled)

def point2index(points, grid_dim=32, ravel=False, ret_relative=False):
    """Convert points in [-1,1]*dim to indices of (grid_dim,)*dim grid.
    The grid is generated using 'in' mode (pixel/voxel mode)

    Args:
        points (torch.Tensor): [*,dim]
        grid_dim (int, optional): dimension of the grid. Defaults to 32.

    Returns:
        [*,dim]: the returned indices
    """
    pt_dim = points.shape[-1]
    offset = 1./grid_dim
    eps = 1e-5 # scale (1,1,...) slightly inside to avoid lying on the boundary
    # max loc 1 is corresponding to  2-2*offset, we need to multiply (grid_dim-1)/(2-2*offset)
    scale = (grid_dim-1-eps)/(2-2*offset)
    shift_points = (points + 1 - offset) * scale
    float_index = torch.clamp(torch.round(shift_points), min=0.0, max=grid_dim-1)
    index = float_index.long()
    if ravel==True:
        index = ravel_index(index, shape=(grid_dim,)*pt_dim)
    if ret_relative==True:
        grid_points   = float_index / scale - 1 + offset
        relative_dist = points - grid_points
        return index, grid_points, relative_dist
    else:
        return index
def index2point(index, grid_dim=32):
    offset = 1./grid_dim
    eps = 1e-5
    scale = (grid_dim-1-eps)/(2-2*offset)
    points = index.float() / scale - 1 + offset
    return points
def p2i_unittest():
    points = torch.rand(2000,2) * 2 - 1. #*1.6 - .8
    rind, grid_points, relative_dist = point2index(points, ret_relative=True, grid_dim=4)
    fig, ax = visutil.newPlot(plotrange=np.array([[-1.,1.],[-1.,1.]]))
    ax.scatter(points.numpy()[:,0],      points.numpy()[:,1], s=2)
    ax.scatter(grid_points.numpy()[:,0], grid_points.numpy()[:,1], zorder=4)
    ax.quiver(grid_points.numpy()[:,0], grid_points.numpy()[:,1], relative_dist.numpy()[:,0], relative_dist.numpy()[:,1], angles='xy', scale_units='xy', scale=1)
    plt.show()
    print((grid_points),(points), (relative_dist))
def point2voxel(points, grid_dim=32, ret_coords=False):
    """Voxelize point cloud, [i][j][k] correspond to x, y, z directly

    Args:
        points (torch.Tensor): [B,num_pts,x_dim]
        grid_dim (int, optional): grid dimension. Defaults to 32.

    Returns:
        torch.Tensor: [B,(grid_dim,)*x_dim]
    """
    voxel = torch.zeros(points.shape[0], *((grid_dim,)*points.shape[-1])).type_as(points)
    inds = point2index(points, grid_dim)
    # make all the indices flat to avoid using for loop for batch
    # (B*num_points, x_dim)
    inds_flat = inds.view(-1,points.shape[-1])
    # [1,2,3] becomes [1,1,1,...,2,2,2,...,3,3,3,...]
    binds = torch.repeat_interleave(torch.arange(points.shape[0]).type_as(points).long(), points.shape[1])
    if points.shape[-1]==2:
        voxel[binds, inds_flat[:,0], inds_flat[:,1]] = 1
    if points.shape[-1]==3:
        voxel[binds, inds_flat[:,0], inds_flat[:,1], inds_flat[:,2]] = 1
    if ret_coords==True:
        x_dim = points.shape[-1]
        coords = nputil.makeGrid(bb_min=[-1,]*x_dim, bb_max=[1,]*x_dim, shape=[grid_dim,]*x_dim, indexing="ij")
        coords = torch.from_numpy(coords[None,...])
        return voxel, coords
    else:
        return voxel

