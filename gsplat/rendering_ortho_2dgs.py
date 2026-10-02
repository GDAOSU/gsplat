import math
from typing import Optional, Tuple, Dict
from typing_extensions import Literal
import torch
from torch import Tensor
import torch.nn.functional as F
from .cuda._wrapper import (
    fully_fused_projection_2dgs,
    rasterize_to_pixels_2dgs,
    spherical_harmonics,
    isect_tiles,
    isect_offset_encode,
    has_2dgs,
)
from .utils import ortho_depth_to_normal
from .rendering import (
    RenderMode,
    render_mode_has_color,
    render_mode_has_depth,
    render_mode_has_depth_channel,
    render_mode_has_expected_depth,
    render_mode_has_only_depth_channel,
)


def rasterization_ortho_2dgs(
    means: Tensor,  # [..., N, 3]
    quats: Tensor,  # [..., N, 4]
    scales: Tensor,  # [..., N, 3]
    opacities: Tensor,  # [..., N]
    colors: Tensor,  # [..., (C,) N, D] or [..., (C,) N, K, 3]
    viewmats: Tensor,  # [..., C, 4, 4]
    Ks: Tensor,  # [..., C, 3, 3]
    width: int,
    height: int,
    near_plane: float = 0.01,
    far_plane: float = 1e10,
    radius_clip: float = 0.0,
    eps2d: float = 0.3,
    sh_degree: Optional[int] = None,
    packed: bool = False,
    tile_size: int = 16,
    backgrounds: Optional[Tensor] = None,
    render_mode: RenderMode = "RGB",
    sparse_grad: bool = False,
    absgrad: bool = False,
    distloss: bool = False,
    depth_mode: Literal["expected", "median"] = "expected",
    camera_model: Literal["ortho"] = "ortho",
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Dict]:
    """Rasterize a set of 2D Gaussians (N) to a batch of image planes (C).

    This function supports a handful of features, similar to the :func:`rasterization` function.

    .. warning::
        This function is currently not differentiable w.r.t. the camera intrinsics `Ks`.

    Args:
        means: The 3D centers of the Gaussians. [..., N, 3]
        quats: The quaternions of the Gaussians (wxyz convension). It's not required to be normalized. [..., N, 4]
        scales: The scales of the Gaussians. [..., N, 3]
        opacities: The opacities of the Gaussians. [..., N]
        colors: The colors of the Gaussians. [..., (C,) N, D] or [..., (C,) N, K, 3] for SH coefficients.
        viewmats: The world-to-cam transformation of the cameras. [..., C, 4, 4]
        Ks: The camera intrinsics. [..., C, 3, 3]
        width: The width of the image.
        height: The height of the image.
        near_plane: The near plane for clipping. Default is 0.01.
        far_plane: The far plane for clipping. Default is 1e10.
        radius_clip: Gaussians with 2D radius smaller or equal than this value will be
            skipped. This is extremely helpful for speeding up large scale scenes.
            Default is 0.0.
        eps2d: An epsilon added to the egienvalues of projected 2D covariance matrices.
            This will prevents the projected GS to be too small. For example eps2d=0.3
            leads to minimal 3 pixel unit. Default is 0.3.
        sh_degree: The SH degree to use, which can be smaller than the total
            number of bands. If set, the `colors` should be [(C,) N, K, 3] SH coefficients,
            else the `colors` should [(C,) N, D] post-activation color values. Default is None.
        packed: Whether to use packed mode which is more memory efficient but might or
            might not be as fast. Default is True.
        tile_size: The size of the tiles for rasterization. Default is 16.
            (Note: other values are not tested)
        backgrounds: The background colors. [C, D]. Default is None.
        render_mode: The rendering mode. Supported modes are "RGB", "d", "Ed", "D", "ED",
            "RGB-d", "RGB-Ed", "RGB+D", and "RGB+ED". "RGB" renders the colored image.
            Gaussian depth modes (D, ED, RGB+D, RGB+ED) use projection depth. Hit distance
            modes (d, Ed, RGB-d, RGB-Ed) compute along-ray distance. Expected modes (Ed, ED)
            are normalized by opacity. Default is "RGB".
        sparse_grad (Experimental): If true, the gradients for {means, quats, scales} will be stored in
            a COO sparse layout. This can be helpful for saving memory. Default is False.
        absgrad: If true, the absolute gradients of the projected 2D means
            will be computed during the backward pass, which could be accessed by
            `meta["means2d"].absgrad`. Default is False.
        channel_chunk: The number of channels to render in one go. Default is 32.
            If the required rendering channels are larger than this value, the rendering
            will be done looply in chunks.
        distloss: If true, use distortion regularization to get better geometry detail.
        depth_mode: render depth mode. Choose from expected depth and median depth.

    Returns:
        A tuple:

        **render_colors**: The rendered colors. [..., C, height, width, X].
        X depends on the `render_mode` and input `colors`. If `render_mode` is "RGB",
        X is D; if `render_mode` is "D" or "ED", X is 1; if `render_mode` is "RGB+D" or
        "RGB+ED", X is D+1.

        **render_alphas**: The rendered alphas. [..., C, height, width, 1].

        **render_normals**: The rendered normals. [..., C, height, width, 3].

        **surf_normals**: surface normal from depth. [..., C, height, width, 3]

        **render_distort**: The rendered distortions. [..., C, height, width, 1].
        L1 version, different from L2 version in 2DGS paper.

        **render_median**: The rendered median depth. [..., C, height, width, 1].

        **meta**: A dictionary of intermediate results of the rasterization.

    Examples:

    .. code-block:: python

        >>> # define Gaussians
        >>> means = torch.randn((100, 3), device=device)
        >>> quats = torch.randn((100, 4), device=device)
        >>> scales = torch.rand((100, 3), device=device) * 0.1
        >>> colors = torch.rand((100, 3), device=device)
        >>> opacities = torch.rand((100,), device=device)
        >>> # define cameras
        >>> viewmats = torch.eye(4, device=device)[None, :, :]
        >>> Ks = torch.tensor([
        >>>    [300., 0., 150.], [0., 300., 100.], [0., 0., 1.]], device=device)[None, :, :]
        >>> width, height = 300, 200
        >>> # render
        >>> colors, alphas, normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
        >>>    means, quats, scales, opacities, colors, viewmats, Ks, width, height
        >>> )
        >>> print (colors.shape, alphas.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 1])
        >>> print (normals.shape, surf_normals.shape)
        torch.Size([1, 200, 300, 3]) torch.Size([1, 200, 300, 3])
        >>> print (distort.shape, median_depth.shape)
        torch.Size([1, 200, 300, 1]) torch.Size([1, 200, 300, 1])
        >>> print (meta.keys())
        dict_keys(['camera_ids', 'gaussian_ids', 'radii', 'means2d', 'depths', 'ray_transforms',
        'opacities', 'normals', 'tile_width', 'tile_height', 'tiles_per_gauss', 'isect_ids',
        'flatten_ids', 'isect_offsets', 'width', 'height', 'tile_size', 'n_cameras', 'render_distort',
        'gradient_2dgs'])

    """

    batch_dims = means.shape[:-2]
    num_batch_dims = len(batch_dims)
    B = math.prod(batch_dims)
    N = means.shape[-2]
    C = viewmats.shape[-3]
    I = B * C
    device = means.device
    channels = colors.shape[-1]

    assert means.shape == batch_dims + (N, 3), means.shape
    assert quats.shape == batch_dims + (N, 4), quats.shape
    assert scales.shape == batch_dims + (N, 3), scales.shape
    assert opacities.shape == batch_dims + (N,), opacities.shape
    assert viewmats.shape == batch_dims + (C, 4, 4), viewmats.shape
    assert Ks.shape == batch_dims + (C, 3, 3), Ks.shape
    if distloss:
        assert render_mode_has_depth(render_mode), (
            f"distloss requires depth rendering, but render mode is {render_mode}"
        )

    if sh_degree is None:
        # treat colors as post-activation values, should be in shape [..., N, D] or [..., C, N, D]
        assert (colors.dim() == num_batch_dims + 2 and colors.shape[:-1] == batch_dims + (N,)) or (
            colors.dim() == num_batch_dims + 3 and colors.shape[:-1] == batch_dims + (C, N)
        ), colors.shape
    else:
        # treat colors as SH coefficients, should be in shape [..., N, K, 3] or [..., C, N, K, 3]
        # Allowing for activating partial SH bands
        assert (
            colors.dim() == num_batch_dims + 3 and colors.shape[:-2] == batch_dims + (N,) and colors.shape[-1] == 3
        ) or (
            colors.dim() == num_batch_dims + 4 and colors.shape[:-2] == batch_dims + (C, N) and colors.shape[-1] == 3
        ), colors.shape
        assert (sh_degree + 1) ** 2 <= colors.shape[-2], colors.shape

    # Compute Ray-Splat intersection transformation.
    proj_results = fully_fused_projection_2dgs(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        width,
        height,
        eps2d,
        near_plane,
        far_plane,
        radius_clip,
        packed,
        sparse_grad,
        camera_model,
    )

    if packed:
        (
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals,
        ) = proj_results
        opacities = opacities.view(B, N)[batch_ids, gaussian_ids]
        image_ids = batch_ids * C + camera_ids
    else:
        radii, means2d, depths, ray_transforms, normals = proj_results
        opacities = torch.broadcast_to(opacities[..., None, :], batch_dims + (C, N))  # [..., C, N]
        camera_ids, gaussian_ids = None, None
        image_ids = None

    densify = torch.zeros_like(means2d, dtype=means.dtype, requires_grad=True, device=device)
    # Identify intersecting tiles
    tile_width = math.ceil(width / float(tile_size))
    tile_height = math.ceil(height / float(tile_size))
    tiles_per_gauss, isect_ids, flatten_ids = isect_tiles(
        means2d,
        radii,
        depths,
        tile_size,
        tile_width,
        tile_height,
        packed=packed,
        n_images=I,
        image_ids=image_ids,
        gaussian_ids=gaussian_ids,
    )
    isect_offsets = isect_offset_encode(isect_ids, I, tile_width, tile_height)
    isect_offsets = isect_offsets.reshape(batch_dims + (C, tile_height, tile_width))

    # TODO: SH also suport N-D.
    # Compute the per-view colors
    # if not (
    #     colors.dim() == num_batch_dims + 3 and sh_degree is None
    # ):  # silently support [..., C, N, D] color.
    #     colors = (
    #         colors.view(B, N, -1)[batch_ids, gaussian_ids]
    #         if packed
    #         else colors[..., None, :, :].expand((-1,) * num_batch_dims + (C, -1, -1))
    #     )  # [nnz, D] or [..., C, N, 3]
    # else:
    #     if packed:
    #         colors = colors.view(B, C, N, -1)[batch_ids, camera_ids, gaussian_ids, :]
    if sh_degree is None:
        # Colors are post-activation values, with shape [..., N, D] or [..., C, N, D]
        if packed:
            if colors.dim() == num_batch_dims + 2:
                # Turn [..., N, D] into [nnz, D]
                colors = colors.view(B, N, -1)[batch_ids, gaussian_ids]
            else:
                # Turn [..., C, N, D] into [nnz, D]
                colors = colors.view(B, C, N, -1)[batch_ids, camera_ids, gaussian_ids]
        else:
            if colors.dim() == num_batch_dims + 2:
                # Turn [..., N, D] into [..., C, N, D]
                colors = torch.broadcast_to(colors[..., None, :, :], batch_dims + (C, N, -1))
            else:
                # colors is already [..., C, N, D]
                pass
    else:  # SH coefficients
        camtoworlds = torch.inverse(viewmats)
        sh_viewmats = torch.eye(4, dtype=viewmats.dtype, device=device).expand(viewmats.shape).clone()
        sh_viewmats[..., :3, 3] = -camtoworlds[..., :3, 3]
        if packed:
            sh_camera_ids = batch_ids * C + camera_ids
            sh_gaussian_ids = batch_ids * N + gaussian_ids
            shs = (
                colors.reshape(B, N, -1, 3)[batch_ids, gaussian_ids]
                if colors.dim() == num_batch_dims + 3
                else colors.reshape(B, C, N, -1, 3)[batch_ids, camera_ids, gaussian_ids]
            )
        else:
            sh_camera_ids = torch.arange(B * C, device=device).repeat_interleave(N)
            sh_gaussian_ids = torch.arange(B * N, device=device).reshape(B, 1, N).expand(B, C, N).reshape(-1)
            shs = (
                colors[..., None, :, :, :].expand(batch_dims + (C, N, colors.shape[-2], 3))
                if colors.dim() == num_batch_dims + 3
                else colors
            ).reshape(B * C * N, -1, 3)
        colors = spherical_harmonics(
            sh_degree,
            means.reshape(B * N, 3),
            sh_viewmats.reshape(B * C, 4, 4),
            shs,
            masks=(radii > 0).all(dim=-1).reshape(-1),
            batch_ids=torch.zeros_like(sh_camera_ids),
            camera_ids=sh_camera_ids,
            gaussian_ids=sh_gaussian_ids,
        )
        if not packed:
            colors = colors.reshape(batch_dims + (C, N, 3))
        # make it apple-to-apple with Inria's CUDA Backend.
        colors = torch.clamp_min(colors + 0.5, 0.0)

    # Rasterize to pixels
    if render_mode_has_depth_channel(render_mode) and render_mode_has_color(render_mode):
        colors = torch.cat((colors, depths[..., None]), dim=-1)

        if backgrounds is not None:
            backgrounds = torch.cat((backgrounds, torch.zeros_like(backgrounds[..., :1])), dim=-1)
    elif render_mode_has_only_depth_channel(render_mode):
        colors = depths[..., None]
        if backgrounds is not None:
            backgrounds = torch.zeros(batch_dims + (C, 1), device=backgrounds.device)
    else:  # RGB
        pass

    (
        render_colors,
        render_alphas,
        render_normals,
        render_distort,
        render_median,
    ) = rasterize_to_pixels_2dgs(
        means2d,
        ray_transforms,
        colors,
        opacities,
        normals,
        densify,
        width,
        height,
        tile_size,
        isect_offsets,
        flatten_ids,
        backgrounds=backgrounds,
        packed=packed,
        absgrad=absgrad,
        distloss=distloss,
        camera_model=camera_model,
    )
    render_normals_from_depth = None
    if render_mode_has_expected_depth(render_mode):
        # normalize the accumulated depth to get the expected depth
        render_colors = torch.cat(
            [
                render_colors[..., :-1],
                render_colors[..., -1:] / render_alphas.clamp(min=1e-10),
            ],
            dim=-1,
        )
    if render_mode_has_depth(render_mode):
        if depth_mode == "expected":
            depth_for_normal = render_colors[..., -1:]
        elif depth_mode == "median":
            depth_for_normal = render_median

        render_normals_from_depth = ortho_depth_to_normal(depth_for_normal, torch.linalg.inv(viewmats), Ks).squeeze(0)

    meta = {
        "camera_ids": camera_ids,
        "gaussian_ids": gaussian_ids,
        "radii": radii,
        "means2d": means2d,
        "depths": depths,
        "ray_transforms": ray_transforms,
        "opacities": opacities,
        "normals": normals,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "tiles_per_gauss": tiles_per_gauss,
        "isect_ids": isect_ids,
        "flatten_ids": flatten_ids,
        "isect_offsets": isect_offsets,
        "width": width,
        "height": height,
        "tile_size": tile_size,
        "n_cameras": C,
        "render_distort": render_distort,
        "gradient_2dgs": densify,  # This holds the gradient used for densification for 2dgs
    }

    return (
        render_colors,
        render_alphas,
        render_normals,
        render_normals_from_depth,
        render_distort,
        render_median,
        meta,
    )
