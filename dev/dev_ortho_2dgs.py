import math

import pytest
import torch
from typing_extensions import Tuple

device = torch.device("cuda:0")


def expand(data: dict, batch_dims: Tuple[int, ...]):
    # append multiple batch dimensions to the front of the tensor
    # eg. x.shape = [N, 3], batch_dims = (1, 2), return shape is [1, 2, N, 3]
    # eg. x.shape = [N, 3], batch_dims = (), return shape is [N, 3]
    ret = {}
    for k, v in data.items():
        if isinstance(v, torch.Tensor) and len(batch_dims) > 0:
            new_shape = batch_dims + v.shape
            ret[k] = v.expand(new_shape)
        else:
            ret[k] = v
    return ret


def gen_test_data():
    C = 3
    N = 1000
    means = torch.randn(N, 3, device=device)
    quats = torch.randn(N, 4, device=device)
    quats = torch.nn.functional.normalize(quats, dim=-1)
    scales = torch.ones(N, 3, device=device)
    scales[..., :2] *= 0.1
    opacities = torch.rand(C, N, device=device) * 0.5
    colors = torch.rand(C, N, 3, device=device)
    viewmats = torch.broadcast_to(torch.eye(4, device=device), (C, 4, 4))
    # W, H = 24, 20
    W, H = 640, 480
    fx, fy, cx, cy = W, W, W // 2, H // 2
    Ks = torch.broadcast_to(
        torch.tensor([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], device=device),
        (C, 3, 3),
    )
    return {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": Ks,
        "width": W,
        "height": H,
    }

test_data = gen_test_data()


from gsplat.cuda._torch_impl_ortho_2dgs import _rasterize_to_pixels_ortho_2dgs
from gsplat.cuda._wrapper import (
    fully_fused_projection_2dgs,
    isect_offset_encode,
    isect_tiles,
    rasterize_to_pixels_2dgs,
)

channels = 3
batch_dims = (2,)

torch.manual_seed(42)

N = test_data["means"].shape[-2]
C = test_data["viewmats"].shape[-3]
I = math.prod(batch_dims) * C
test_data.update(
    {
        "colors": torch.rand(C, N, channels, device=device),
        "backgrounds": torch.rand(C, channels, device=device),
    }
)

test_data = expand(test_data, batch_dims)
Ks = test_data["Ks"]
viewmats = test_data["viewmats"]
height = test_data["height"]
width = test_data["width"]
quats = test_data["quats"]
scales = test_data["scales"]
means = test_data["means"]
opacities = test_data["opacities"]
colors = test_data["colors"]
backgrounds = test_data["backgrounds"]

radii, means2d, depths, M_i2u, normals, depth_grads = fully_fused_projection_2dgs(
    means, quats, scales, viewmats, Ks, width, height, camera_model="ortho", near_plane=-1e6, far_plane=1e6
)
colors = torch.cat([colors, depths[..., None]], dim=-1)
backgrounds = torch.zeros(batch_dims + (C, channels + 1), device=device)

# Identify intersecting tiles
tile_size = 16
tile_width = math.ceil(width / float(tile_size))
tile_height = math.ceil(height / float(tile_size))
_, isect_ids, flatten_ids = isect_tiles(means2d, radii, depths, tile_size, tile_width, tile_height)
isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))
densify = torch.zeros_like(means2d, device=means2d.device)

means2d.requires_grad = True
M_i2u.requires_grad = True
colors.requires_grad = True
opacities.requires_grad = True
backgrounds.requires_grad = True
normals.requires_grad = True
densify.requires_grad = True

(
    render_colors,
    render_alphas,
    render_normals,
    _,
    _,
) = rasterize_to_pixels_2dgs(
    means2d,
    M_i2u,
    colors,
    opacities,
    normals,
    depth_grads,
    densify,
    width,
    height,
    tile_size,
    isect_offsets,
    flatten_ids,
    backgrounds=backgrounds,
    distloss=True,
    camera_model="ortho",
)

_render_colors, _render_alphas, _render_normals = _rasterize_to_pixels_ortho_2dgs(
    means2d,
    M_i2u,
    colors,
    normals,
    depth_grads,
    opacities,
    width,
    height,
    tile_size,
    isect_offsets,
    flatten_ids,
    backgrounds=backgrounds,
)

v_render_colors = torch.rand_like(render_colors)
v_render_alphas = torch.rand_like(render_alphas)
v_render_normals = torch.rand_like(render_normals)

(
    v_means2d,
    v_ray_transforms,
    v_colors,
    v_opacities,
    v_backgrounds,
    v_normals,
) = torch.autograd.grad(
    (render_colors * v_render_colors).sum()
    + (render_alphas * v_render_alphas).sum()
    + (render_normals * v_render_normals).sum(),
    (means2d, M_i2u, colors, opacities, backgrounds, normals),
)

(
    _v_means2d,
    _v_ray_transforms,
    _v_colors,
    _v_opacities,
    _v_backgrounds,
    _v_normals,
) = torch.autograd.grad(
    (_render_colors * v_render_colors).sum()
    + (_render_alphas * v_render_alphas).sum()
    + (_render_normals * v_render_normals).sum(),
    (means2d, M_i2u, colors, opacities, backgrounds, normals),
)

# assert close forward
torch.testing.assert_close(render_colors, _render_colors, atol=1e-3, rtol=1e-3)
torch.testing.assert_close(render_alphas, _render_alphas, atol=1e-3, rtol=1e-3)
torch.testing.assert_close(render_normals, _render_normals, atol=1e-3, rtol=1e-3)

# assert close backward
torch.testing.assert_close(v_means2d, _v_means2d, rtol=1e-3, atol=1e-3)

torch.testing.assert_close(v_ray_transforms, _v_ray_transforms, rtol=2e-1, atol=5e-2)
torch.testing.assert_close(v_colors, _v_colors, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(v_opacities, _v_opacities, rtol=1e-3, atol=1e-3)
torch.testing.assert_close(v_backgrounds, _v_backgrounds, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(v_normals, _v_normals, rtol=1e-3, atol=1e-3)
