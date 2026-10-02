import pytest
import torch

from gsplat.cuda._torch_impl import _persp_skew_proj, _spherical_harmonics
from gsplat.cuda._torch_impl_2dgs import _fully_fused_projection_2dgs
from gsplat.cuda._wrapper import (
    fully_fused_projection,
    fully_fused_projection_2dgs,
    proj,
    rasterize_to_indices_in_range_2dgs,
    rasterize_to_pixels_2dgs,
)
from gsplat.rendering import rasterization, rasterization_2dgs
from gsplat.utils import depth_to_points_skew

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def scene(skew=24.0, batch=False):
    generator = torch.Generator(device="cuda").manual_seed(73)
    means = torch.randn(32, 3, generator=generator, device="cuda") * 0.1
    means[:, 2] += 3.0
    quats = torch.randn(32, 4, generator=generator, device="cuda")
    scales = torch.full((32, 3), 0.03, device="cuda")
    viewmats = torch.eye(4, device="cuda").repeat(2, 1, 1)
    viewmats[1, 0, 3] = 0.1
    intrinsics = torch.tensor([[80.0, skew, 32.0], [0.0, 85.0, 24.0], [0.0, 0.0, 1.0]], device="cuda").repeat(2, 1, 1)
    inputs = [means, quats, scales, viewmats]
    if batch:
        inputs = [tensor.unsqueeze(0).repeat(2, *([1] * tensor.ndim)) for tensor in inputs]
        intrinsics = intrinsics.unsqueeze(0).repeat(2, 1, 1, 1)
    return [tensor.requires_grad_() for tensor in inputs], intrinsics


@pytest.mark.parametrize("skew", [0.0, 24.0, -37.0])
@pytest.mark.parametrize("clipped", [False, True])
def test_ewa_projection_and_gradients(skew, clipped):
    inputs, intrinsics = scene(skew)
    means = inputs[0].detach().unsqueeze(0).repeat(2, 1, 1).requires_grad_()
    if clipped:
        with torch.no_grad():
            means[:, :8, 0] = 7.0
            means[:, 8:16, 1] = -8.0
    generator = torch.Generator(device="cuda").manual_seed(74)
    factor = torch.randn(2, 32, 3, 3, generator=generator, device="cuda") * 0.04
    covars = (factor @ factor.transpose(-1, -2)).requires_grad_()
    actual = proj(means, covars, intrinsics, 64, 48, camera_model="pinhole_skew")
    expected = _persp_skew_proj(means, covars, intrinsics, 64, 48)
    weights = [torch.randn(tensor.shape, generator=generator, device="cuda") for tensor in actual]
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference, rtol=2e-5, atol=2e-5)
    actual_grads = torch.autograd.grad(
        sum((tensor * weight).sum() for tensor, weight in zip(actual, weights)), (means, covars)
    )
    expected_grads = torch.autograd.grad(
        sum((tensor * weight).sum() for tensor, weight in zip(expected, weights)), (means, covars)
    )
    for grad, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, reference, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("skew", [0.0, 24.0, -37.0])
def test_2dgs_projection_and_gradients(batch, packed, skew):
    inputs, intrinsics = scene(skew, batch)
    actual = fully_fused_projection_2dgs(*inputs, intrinsics, 64, 48, packed=packed, camera_model="pinhole_skew")
    reference_scales = torch.cat((inputs[2][..., :2], torch.ones_like(inputs[2][..., 2:3])), dim=-1)
    expected = _fully_fused_projection_2dgs(inputs[0], inputs[1], reference_scales, inputs[3], intrinsics, 64, 48)
    if packed:
        batch_ids, camera_ids, gaussian_ids, indptr, radii, *outputs = actual
        if batch:
            expected_outputs = [tensor[batch_ids, camera_ids, gaussian_ids] for tensor in expected[1:]]
        else:
            expected_outputs = [tensor[camera_ids, gaussian_ids] for tensor in expected[1:]]
    else:
        radii, *outputs = actual
        expected_outputs = expected[1:]
    assert torch.all(radii > 0)
    weights = []
    generator = torch.Generator(device="cuda").manual_seed(75)
    for output, reference in zip(outputs, expected_outputs):
        torch.testing.assert_close(output, reference, rtol=2e-4, atol=2e-4)
        weights.append(torch.randn(output.shape, generator=generator, device="cuda") / output.numel())
    actual_grads = torch.autograd.grad(sum((tensor * weight).sum() for tensor, weight in zip(outputs, weights)), inputs)
    expected_grads = torch.autograd.grad(
        sum((tensor * weight).sum() for tensor, weight in zip(expected_outputs, weights)), inputs
    )
    for grad, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, reference, rtol=3e-3, atol=2e-4)


@pytest.mark.parametrize("backend", ["3dgs", "2dgs"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("render_mode", ["RGB", "D", "ED"])
def test_zero_skew_render_preserves_fast_path(backend, packed, render_mode):
    inputs, intrinsics = scene(0.0)
    means, quats, scales, viewmats = inputs
    colors = torch.sigmoid(means.detach())
    opacities = torch.full((32,), 0.4, device="cuda", requires_grad=True)
    renderer = rasterization if backend == "3dgs" else rasterization_2dgs
    common = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=intrinsics,
        width=64,
        height=48,
        packed=packed,
        tile_size=16,
        render_mode=render_mode,
        backgrounds=torch.full((2, 3), 0.2, device="cuda"),
    )
    fast = renderer(**common, camera_model="pinhole")
    skew = renderer(**common, camera_model="pinhole_skew")
    for output, reference in zip(skew[:2], fast[:2]):
        torch.testing.assert_close(output, reference, rtol=3e-4, atol=3e-5)
    fast_grads = torch.autograd.grad(fast[0].square().sum() + fast[1].square().sum(), (*inputs, opacities))
    skew_grads = torch.autograd.grad(skew[0].square().sum() + skew[1].square().sum(), (*inputs, opacities))
    for grad, reference in zip(skew_grads, fast_grads):
        torch.testing.assert_close(grad, reference, rtol=5e-3, atol=2e-3)


@pytest.mark.parametrize("packed", [False, True])
def test_ewa_fused_packed_matches_dense(packed):
    inputs, intrinsics = scene()
    means, quats, scales, viewmats = inputs
    actual = fully_fused_projection(
        means, None, quats, scales, viewmats, intrinsics, 64, 48, packed=packed, camera_model="pinhole_skew"
    )
    dense = fully_fused_projection(
        means, None, quats, scales, viewmats, intrinsics, 64, 48, camera_model="pinhole_skew"
    )
    if packed:
        batch_ids, camera_ids, gaussian_ids, indptr, radii, *outputs = actual
        references = [tensor[camera_ids, gaussian_ids] for tensor in dense[1:] if tensor is not None]
        outputs = [tensor for tensor in outputs if tensor is not None]
    else:
        radii, *outputs = actual
        references = [tensor for tensor in dense[1:] if tensor is not None]
        outputs = [tensor for tensor in outputs if tensor is not None]
    assert torch.all(radii > 0)
    for output, reference in zip(outputs, references):
        torch.testing.assert_close(output, reference)
    actual_grads = torch.autograd.grad(sum(tensor.square().mean() for tensor in outputs), inputs)
    dense_grads = torch.autograd.grad(sum(tensor.square().mean() for tensor in references), inputs)
    for grad, reference in zip(actual_grads, dense_grads):
        torch.testing.assert_close(grad, reference, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("backend", ["3dgs", "2dgs"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("render_mode", ["RGB", "D", "ED"])
def test_nonzero_skew_render_and_gradients(backend, packed, render_mode):
    inputs, intrinsics = scene(-37.0)
    means, quats, scales, viewmats = inputs
    shear = torch.eye(4, device="cuda").repeat(2, 1, 1)
    shear[:, 0, 1] = intrinsics[:, 0, 1] / intrinsics[:, 0, 0]
    reference_intrinsics = intrinsics.clone()
    reference_intrinsics[:, 0, 1] = 0.0
    renderer = rasterization if backend == "3dgs" else rasterization_2dgs
    colors = torch.sigmoid(means.detach())
    opacities = torch.full((32,), 0.4, device="cuda", requires_grad=True)
    common = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        width=64,
        height=48,
        packed=packed,
        tile_size=16,
        render_mode=render_mode,
        backgrounds=torch.full((2, 3), 0.2, device="cuda"),
    )
    actual = renderer(**common, viewmats=viewmats, Ks=intrinsics, camera_model="pinhole_skew")
    reference = renderer(**common, viewmats=shear @ viewmats, Ks=reference_intrinsics, camera_model="pinhole")
    for output, expected in zip(actual[:2], reference[:2]):
        torch.testing.assert_close(output, expected, rtol=3e-4, atol=3e-5)
    valid = actual[1].detach() > 0.01 if render_mode == "ED" else torch.ones_like(actual[1])
    actual_grads = torch.autograd.grad(
        (actual[0].square() * valid).sum() + actual[1].square().sum(), (*inputs, opacities)
    )
    expected_grads = torch.autograd.grad(
        (reference[0].square() * valid).sum() + reference[1].square().sum(), (*inputs, opacities)
    )
    for grad, expected in zip(actual_grads, expected_grads):
        if render_mode == "ED":
            tolerance = max(3e-3, expected.abs().max().item() * 5e-5)
            torch.testing.assert_close(grad, expected, rtol=5e-3, atol=tolerance)
            assert (grad - expected).norm() <= expected.norm() * 1e-4 + 1e-5
        else:
            torch.testing.assert_close(grad, expected, rtol=5e-3, atol=3e-3)


@pytest.mark.parametrize("backend", ["3dgs", "2dgs"])
def test_skew_sparse_projection_gradients(backend):
    inputs, intrinsics = scene()
    if backend == "2dgs":
        project = lambda sparse: fully_fused_projection_2dgs(
            *inputs, intrinsics, 64, 48, packed=True, sparse_grad=sparse, camera_model="pinhole_skew"
        )
    else:
        means, quats, scales, viewmats = inputs
        project = lambda sparse: fully_fused_projection(
            means,
            None,
            quats,
            scales,
            viewmats,
            intrinsics,
            64,
            48,
            packed=True,
            sparse_grad=sparse,
            camera_model="pinhole_skew",
        )
    sparse_outputs = [tensor for tensor in project(True) if tensor is not None and tensor.is_floating_point()]
    dense_outputs = [tensor for tensor in project(False) if tensor is not None and tensor.is_floating_point()]
    sparse_grads = torch.autograd.grad(sum(tensor.square().mean() for tensor in sparse_outputs), inputs)
    dense_grads = torch.autograd.grad(sum(tensor.square().mean() for tensor in dense_outputs), inputs)
    for index, (grad, expected) in enumerate(zip(sparse_grads, dense_grads)):
        assert grad.is_sparse == (index < 3)
        torch.testing.assert_close(grad.to_dense(), expected, rtol=3e-4, atol=3e-4)


@pytest.mark.parametrize("backend", ["3dgs", "2dgs"])
@pytest.mark.parametrize("packed", [False, True])
def test_skew_sh_uses_rigid_camera_directions(backend, packed):
    inputs, intrinsics = scene(-37.0)
    means, quats, scales, viewmats = inputs
    generator = torch.Generator(device="cuda").manual_seed(95)
    coefficients = (torch.randn(32, 4, 3, generator=generator, device="cuda") * 0.1).requires_grad_()
    opacities = torch.full((32,), 0.4, device="cuda", requires_grad=True)
    origins = -torch.einsum("cij,cj->ci", viewmats[:, :3, :3].transpose(-1, -2), viewmats[:, :3, 3])
    directions = means[None, :, :] - origins[:, None, :]
    reference_colors = (_spherical_harmonics(1, directions, coefficients) + 0.5).clamp_min(0)
    renderer = rasterization if backend == "3dgs" else rasterization_2dgs
    common = dict(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        viewmats=viewmats,
        Ks=intrinsics,
        width=64,
        height=48,
        packed=packed,
        camera_model="pinhole_skew",
    )
    actual = renderer(**common, colors=coefficients, sh_degree=1)
    expected = renderer(**common, colors=reference_colors)
    for output, reference in zip(actual[:2], expected[:2]):
        torch.testing.assert_close(output, reference, rtol=3e-4, atol=3e-5)
    parameters = (*inputs, opacities, coefficients)
    actual_grads = torch.autograd.grad(actual[0].square().sum(), parameters)
    expected_grads = torch.autograd.grad(expected[0].square().sum(), parameters)
    for grad, reference in zip(actual_grads, expected_grads):
        torch.testing.assert_close(grad, reference, rtol=5e-3, atol=3e-3)


def test_skew_depth_roundtrip():
    _, intrinsics = scene()
    depth = torch.full((2, 8, 10, 1), 3.0, device="cuda")
    points = depth_to_points_skew(depth, torch.eye(4, device="cuda").repeat(2, 1, 1), intrinsics)
    projected = torch.einsum("cij,chwj->chwi", intrinsics, points)
    projected = projected[..., :2] / projected[..., 2:3]
    horizontal, vertical = torch.meshgrid(
        torch.arange(10, device="cuda") + 0.5, torch.arange(8, device="cuda") + 0.5, indexing="xy"
    )
    torch.testing.assert_close(
        projected, torch.stack((horizontal, vertical), dim=-1).expand(2, -1, -1, -1), atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize("packed", [False, True])
def test_skew_low_level_pixel_rasterization(packed):
    inputs, intrinsics = scene()
    means, quats, scales, viewmats = inputs
    colors = torch.sigmoid(means.detach())
    opacities = torch.full((32,), 0.4, device="cuda")
    rendered = rasterization_2dgs(
        means,
        quats,
        scales,
        opacities,
        colors,
        viewmats,
        intrinsics,
        64,
        48,
        packed=packed,
        camera_model="pinhole_skew",
    )
    metadata = rendered[-1]
    common = dict(
        means2d=metadata["means2d"],
        ray_transforms=metadata["ray_transforms"],
        opacities=metadata["opacities"],
        image_width=64,
        image_height=48,
        tile_size=16,
        isect_offsets=metadata["isect_offsets"],
        flatten_ids=metadata["flatten_ids"],
    )
    projected_colors = colors[metadata["gaussian_ids"]] if packed else colors.unsqueeze(0).expand(2, -1, -1)
    actual = rasterize_to_pixels_2dgs(
        **common,
        colors=projected_colors,
        normals=metadata["normals"],
        densify=metadata["gradient_2dgs"],
        packed=packed,
        camera_model="pinhole_skew",
    )
    for output, reference in zip(actual[:2], rendered[:2]):
        torch.testing.assert_close(output, reference)
    if not packed:
        range_args = dict(
            **common,
            range_start=0,
            range_end=100,
            transmittances=torch.ones(2, 48, 64, device="cuda"),
        )
        skew_ids = rasterize_to_indices_in_range_2dgs(**range_args, camera_model="pinhole_skew")
        fast_ids = rasterize_to_indices_in_range_2dgs(**range_args, camera_model="pinhole")
        for actual_ids, expected_ids in zip(skew_ids, fast_ids):
            torch.testing.assert_close(actual_ids, expected_ids)


@pytest.mark.parametrize("flag", ["with_ut", "with_eval3d"])
def test_unsupported_paths_fail_explicitly(flag):
    inputs, intrinsics = scene()
    means, quats, scales, viewmats = inputs
    with pytest.raises(ValueError, match="EWA"):
        rasterization(
            means,
            quats,
            scales,
            torch.ones(32, device="cuda"),
            torch.ones(32, 3, device="cuda"),
            viewmats,
            intrinsics,
            64,
            48,
            camera_model="pinhole_skew",
            **{flag: True},
        )
