#include <ATen/Functions.h>
#include <ATen/TensorUtils.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/library.h>
#include "Common.h"
#include "Projection.h"
#include "Rasterization.h"
#include "Config.h"

namespace gsplat
{
#if GSPLAT_BUILD_2DGS
std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor
>
    ortho_projection_2dgs_fused_fwd(
        const at::Tensor &means,    // [..., N, 3]
        const at::Tensor &quats,    // [..., N, 4]
        const at::Tensor &scales,   // [..., N, 3]
        const at::Tensor &viewmats, // [..., C, 4, 4]
        const at::Tensor &Ks,       // [..., C, 3, 3]
        int64_t image_width,
        int64_t image_height,
        double eps2d,
        double near_plane,
        double far_plane,
        double radius_clip
    )
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    auto opt = means.options();
    at::DimVector batch_dims(means.sizes().slice(0, means.dim() - 2));
    uint32_t N = means.size(-2);    // number of gaussians
    uint32_t C = viewmats.size(-3); // number of cameras

    at::DimVector radii_shape(batch_dims);
    radii_shape.append({C, N, 2});
    at::Tensor radii = at::empty(radii_shape, opt.dtype(at::kInt));

    at::DimVector means2d_shape(batch_dims);
    means2d_shape.append({C, N, 2});
    at::Tensor means2d = at::empty(means2d_shape, opt);

    at::DimVector depths_shape(batch_dims);
    depths_shape.append({C, N});
    at::Tensor depths = at::empty(depths_shape, opt);

    at::DimVector ray_transforms_shape(batch_dims);
    ray_transforms_shape.append({C, N, 3, 3});
    at::Tensor ray_transforms = at::empty(ray_transforms_shape, opt);

    at::DimVector normals_shape(batch_dims);
    normals_shape.append({C, N, 3});
    at::Tensor normals = at::zeros(normals_shape, opt);

    launch_ortho_projection_2dgs_fused_fwd_kernel(
        // inputs
        means,
        quats,
        scales,
        viewmats,
        Ks,
        static_cast<uint32_t>(image_width),
        static_cast<uint32_t>(image_height),
        static_cast<float>(near_plane),
        static_cast<float>(far_plane),
        static_cast<float>(radius_clip),
        // outputs
        radii,
        means2d,
        depths,
        ray_transforms,
        normals
    );
    return std::make_tuple(radii, means2d, depths, ray_transforms, normals);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ortho_projection_2dgs_fused_bwd(
    // fwd inputs
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    // fwd outputs
    const at::Tensor &radii,          // [..., C, N, 2]
    const at::Tensor &ray_transforms, // [..., C, N, 3, 3]
    // grad outputs
    const at::Tensor &v_means2d,        // [..., C, N, 2]
    const at::Tensor &v_depths,         // [..., C, N]
    const at::Tensor &v_normals,        // [..., C, N, 3]
    const at::Tensor &v_ray_transforms, // [..., C, N, 3, 3]
    bool viewmats_requires_grad
)
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(radii);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_normals);
    CHECK_INPUT(v_ray_transforms);

    at::Tensor v_means  = at::zeros_like(means);
    at::Tensor v_quats  = at::zeros_like(quats);
    at::Tensor v_scales = at::zeros_like(scales);
    at::Tensor v_viewmats;
    if(viewmats_requires_grad)
    {
        v_viewmats = at::zeros_like(viewmats);
    }

    launch_ortho_projection_2dgs_fused_bwd_kernel(
        // inputs
        means,
        quats,
        scales,
        viewmats,
        Ks,
        static_cast<uint32_t>(image_width),
        static_cast<uint32_t>(image_height),
        radii,
        ray_transforms,
        v_means2d,
        v_depths,
        v_normals,
        v_ray_transforms,
        viewmats_requires_grad,
        // outputs
        v_means,
        v_quats,
        v_scales,
        v_viewmats
    );

    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}

std::tuple<
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor,
    at::Tensor
>
    ortho_projection_2dgs_packed_fwd(
        const at::Tensor &means,    // [..., N, 3]
        const at::Tensor &quats,    // [..., N, 4]
        const at::Tensor &scales,   // [..., N, 3]
        const at::Tensor &viewmats, // [..., C, 4, 4]
        const at::Tensor &Ks,       // [..., C, 3, 3]
        int64_t image_width,
        int64_t image_height,
        double near_plane,
        double far_plane,
        double radius_clip
    )
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);

    uint32_t N = means.size(-2);
    uint32_t B = means.numel() / (N * 3);
    uint32_t C = viewmats.size(-3);
    auto opt   = means.options();

    uint32_t nrows          = B * C;
    uint32_t ncols          = N;
    uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

    int32_t nnz;
    at::Tensor block_accum;
    if(B && C && N)
    {
        at::Tensor block_cnts = at::empty({nrows * blocks_per_row}, opt.dtype(at::kInt));
        launch_ortho_projection_2dgs_packed_fwd_kernel(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            static_cast<uint32_t>(image_width),
            static_cast<uint32_t>(image_height),
            static_cast<float>(near_plane),
            static_cast<float>(far_plane),
            static_cast<float>(radius_clip),
            c10::nullopt,
            block_cnts,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt,
            c10::nullopt
        );
        block_accum = at::cumsum(block_cnts, 0, at::kInt);
        nnz         = block_accum[-1].item<int32_t>();
    }
    else
    {
        nnz = 0;
    }

    at::Tensor indptr         = at::empty({B * C + 1}, opt.dtype(at::kInt));
    at::Tensor batch_ids      = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor camera_ids     = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor gaussian_ids   = at::empty({nnz}, opt.dtype(at::kLong));
    at::Tensor radii          = at::empty({nnz, 2}, opt.dtype(at::kInt));
    at::Tensor means2d        = at::empty({nnz, 2}, opt);
    at::Tensor depths         = at::empty({nnz}, opt);
    at::Tensor ray_transforms = at::empty({nnz, 3, 3}, opt);
    at::Tensor normals        = at::empty({nnz, 3}, opt);

    if(nnz)
    {
        launch_ortho_projection_2dgs_packed_fwd_kernel(
            means,
            quats,
            scales,
            viewmats,
            Ks,
            static_cast<uint32_t>(image_width),
            static_cast<uint32_t>(image_height),
            static_cast<float>(near_plane),
            static_cast<float>(far_plane),
            static_cast<float>(radius_clip),
            block_accum,
            c10::nullopt,
            indptr,
            batch_ids,
            camera_ids,
            gaussian_ids,
            radii,
            means2d,
            depths,
            ray_transforms,
            normals
        );
    }
    else
    {
        indptr.fill_(0);
    }

    return std::make_tuple(
        indptr, batch_ids, camera_ids, gaussian_ids, radii, means2d, depths, ray_transforms, normals
    );
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> ortho_projection_2dgs_packed_bwd(
    const at::Tensor &means,    // [..., N, 3]
    const at::Tensor &quats,    // [..., N, 4]
    const at::Tensor &scales,   // [..., N, 3]
    const at::Tensor &viewmats, // [..., C, 4, 4]
    const at::Tensor &Ks,       // [..., C, 3, 3]
    int64_t image_width,
    int64_t image_height,
    const at::Tensor &batch_ids,        // [nnz]
    const at::Tensor &camera_ids,       // [nnz]
    const at::Tensor &gaussian_ids,     // [nnz]
    const at::Tensor &ray_transforms,   // [nnz, 3, 3]
    const at::Tensor &v_means2d,        // [nnz, 2]
    const at::Tensor &v_depths,         // [nnz]
    const at::Tensor &v_ray_transforms, // [nnz, 3, 3]
    const at::Tensor &v_normals,        // [nnz, 3]
    bool viewmats_requires_grad,
    bool sparse_grad
)
{
    DEVICE_GUARD(means);
    CHECK_INPUT(means);
    CHECK_INPUT(quats);
    CHECK_INPUT(scales);
    CHECK_INPUT(viewmats);
    CHECK_INPUT(Ks);
    CHECK_INPUT(batch_ids);
    CHECK_INPUT(camera_ids);
    CHECK_INPUT(gaussian_ids);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(v_means2d);
    CHECK_INPUT(v_depths);
    CHECK_INPUT(v_normals);
    CHECK_INPUT(v_ray_transforms);

    auto opt     = means.options();
    uint32_t nnz = batch_ids.size(0);

    at::Tensor v_means, v_quats, v_scales, v_viewmats;
    if(sparse_grad)
    {
        v_means  = at::zeros({nnz, 3}, opt);
        v_quats  = at::zeros({nnz, 4}, opt);
        v_scales = at::zeros({nnz, 3}, opt);
    }
    else
    {
        v_means  = at::zeros_like(means, opt);
        v_quats  = at::zeros_like(quats, opt);
        v_scales = at::zeros_like(scales, opt);
    }
    if(viewmats_requires_grad)
    {
        v_viewmats = at::zeros_like(viewmats, opt);
    }

    launch_ortho_projection_2dgs_packed_bwd_kernel(
        means,
        quats,
        scales,
        viewmats,
        Ks,
        static_cast<uint32_t>(image_width),
        static_cast<uint32_t>(image_height),
        batch_ids,
        camera_ids,
        gaussian_ids,
        ray_transforms,
        v_means2d,
        v_depths,
        v_ray_transforms,
        v_normals,
        sparse_grad,
        v_means,
        v_quats,
        v_scales,
        v_viewmats.defined() ? at::optional<at::Tensor>(v_viewmats) : c10::nullopt
    );
    return std::make_tuple(v_means, v_quats, v_scales, v_viewmats);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    rasterize_to_pixels_ortho_2dgs_fwd(
        // Gaussian parameters
        const at::Tensor &means2d,                  // [..., N, 2] or [nnz, 2]
        const at::Tensor &ray_transforms,           // [..., N, 3, 3] or [nnz, 3, 3]
        const at::Tensor &colors,                   // [..., N, channels] or [nnz, channels]
        const at::Tensor &opacities,                // [..., N]  or [nnz]
        const at::Tensor &normals,                  // [..., N, 3] or [nnz, 3]
        const at::optional<at::Tensor> backgrounds, // [..., channels]
        const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
        // image size
        int64_t image_width,
        int64_t image_height,
        int64_t tile_size,
        // intersections
        const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
        const at::Tensor &flatten_ids   // [n_isects]
    )
{
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(normals);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    if(backgrounds.has_value())
    {
        CHECK_INPUT(backgrounds.value());
    }
    if(masks.has_value())
    {
        CHECK_INPUT(masks.value());
    }
    auto opt = means2d.options();

    at::DimVector image_dims(tile_offsets.sizes().slice(0, tile_offsets.dim() - 2));
    uint32_t channels = colors.size(-1);

    at::DimVector renders_dims(image_dims);
    renders_dims.append({image_height, image_width, channels});
    at::Tensor renders = at::empty(renders_dims, opt);

    at::DimVector alphas_dims(image_dims);
    alphas_dims.append({image_height, image_width, 1});
    at::Tensor alphas = at::empty(alphas_dims, opt);

    at::DimVector last_ids_dims(image_dims);
    last_ids_dims.append({image_height, image_width});
    at::Tensor last_ids = at::empty(last_ids_dims, opt.dtype(at::kInt));

    at::DimVector median_ids_dims(image_dims);
    median_ids_dims.append({image_height, image_width});
    at::Tensor median_ids = at::empty(median_ids_dims, opt.dtype(at::kInt));

    at::DimVector render_normals_dims(image_dims);
    render_normals_dims.append({image_height, image_width, 3});
    at::Tensor render_normals = at::empty(render_normals_dims, opt);

    at::DimVector render_distort_dims(image_dims);
    render_distort_dims.append({image_height, image_width, 1});
    at::Tensor render_distort = at::empty(render_distort_dims, opt);

    at::DimVector render_median_dims(image_dims);
    render_median_dims.append({image_height, image_width, 1});
    at::Tensor render_median = at::empty(render_median_dims, opt);

#    define __LAUNCH_KERNEL__(N)                             \
    case N:                                                  \
        launch_rasterize_to_pixels_ortho_2dgs_fwd_kernel<N>( \
            means2d,                                         \
            ray_transforms,                                  \
            colors,                                          \
            opacities,                                       \
            normals,                                         \
            backgrounds,                                     \
            masks,                                           \
            static_cast<uint32_t>(image_width),              \
            static_cast<uint32_t>(image_height),             \
            static_cast<uint32_t>(tile_size),                \
            tile_offsets,                                    \
            flatten_ids,                                     \
            renders,                                         \
            alphas,                                          \
            render_normals,                                  \
            render_distort,                                  \
            render_median,                                   \
            last_ids,                                        \
            median_ids                                       \
        );                                                   \
        break;

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    switch(channels)
    {
        __LAUNCH_KERNEL__(1)
        __LAUNCH_KERNEL__(2)
        __LAUNCH_KERNEL__(3)
        __LAUNCH_KERNEL__(4)
        __LAUNCH_KERNEL__(5)
        __LAUNCH_KERNEL__(8)
        __LAUNCH_KERNEL__(9)
        __LAUNCH_KERNEL__(16)
        __LAUNCH_KERNEL__(17)
        __LAUNCH_KERNEL__(32)
        __LAUNCH_KERNEL__(33)
        __LAUNCH_KERNEL__(64)
        __LAUNCH_KERNEL__(65)
        __LAUNCH_KERNEL__(128)
        __LAUNCH_KERNEL__(129)
        __LAUNCH_KERNEL__(256)
        __LAUNCH_KERNEL__(257)
        __LAUNCH_KERNEL__(512)
        __LAUNCH_KERNEL__(513)
    default: AT_ERROR("Unsupported number of channels: ", channels);
    }
#    undef __LAUNCH_KERNEL__

    return std::make_tuple(renders, alphas, render_normals, render_distort, render_median, last_ids, median_ids);
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor>
    rasterize_to_pixels_ortho_2dgs_bwd(
        // Gaussian parameters
        const at::Tensor &means2d,        // [..., N, 2] or [nnz, 2]
        const at::Tensor &ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
        const at::Tensor &colors,         // [..., N, channels] or [nnz, channels]
        const at::Tensor &opacities,      // [..., N] or [nnz]
        const at::Tensor &normals,        // [..., N, 3] or [nnz, 3]
        const at::Tensor &densify,
        const at::optional<at::Tensor> backgrounds, // [..., channels]
        const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
        // image size
        int64_t image_width,
        int64_t image_height,
        int64_t tile_size,
        // ray_crossions
        const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
        const at::Tensor &flatten_ids,  // [n_isects]
        // forward outputs
        const at::Tensor &render_colors, // [..., image_height, image_width, channels]
        const at::Tensor &render_alphas, // [..., image_height, image_width, 1]
        const at::Tensor &last_ids,      // [..., image_height, image_width]
        const at::Tensor &median_ids,    // [..., image_height, image_width]
        // gradients of outputs
        const at::Tensor &v_render_colors,  // [..., image_height, image_width, channels]
        const at::Tensor &v_render_alphas,  // [..., image_height, image_width, 1]
        const at::Tensor &v_render_normals, // [..., image_height, image_width, 3]
        const at::Tensor &v_render_distort, // [..., image_height, image_width, 1]
        const at::Tensor &v_render_median,  // [..., image_height, image_width, 1]
        // options
        bool absgrad
    )
{
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(colors);
    CHECK_INPUT(opacities);
    CHECK_INPUT(normals);
    CHECK_INPUT(densify);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);
    CHECK_INPUT(render_colors);
    CHECK_INPUT(render_alphas);
    CHECK_INPUT(last_ids);
    CHECK_INPUT(median_ids);
    CHECK_INPUT(v_render_colors);
    CHECK_INPUT(v_render_alphas);
    CHECK_INPUT(v_render_normals);
    CHECK_INPUT(v_render_distort);
    CHECK_INPUT(v_render_median);
    if(backgrounds.has_value())
    {
        CHECK_INPUT(backgrounds.value());
    }
    if(masks.has_value())
    {
        CHECK_INPUT(masks.value());
    }

    uint32_t channels = colors.size(-1);

    at::Tensor v_means2d        = at::zeros_like(means2d);
    at::Tensor v_ray_transforms = at::zeros_like(ray_transforms);
    at::Tensor v_colors         = at::zeros_like(colors);
    at::Tensor v_normals        = at::zeros_like(normals);
    at::Tensor v_opacities      = at::zeros_like(opacities);
    at::Tensor v_means2d_abs;
    if(absgrad)
    {
        v_means2d_abs = at::zeros_like(means2d);
    }
    at::Tensor v_densify = at::zeros_like(densify);

#    define __LAUNCH_KERNEL__(N)                                               \
    case N:                                                                    \
        launch_rasterize_to_pixels_ortho_2dgs_bwd_kernel<N>(                   \
            means2d,                                                           \
            ray_transforms,                                                    \
            colors,                                                            \
            opacities,                                                         \
            normals,                                                           \
            densify,                                                           \
            backgrounds,                                                       \
            masks,                                                             \
            static_cast<uint32_t>(image_width),                                \
            static_cast<uint32_t>(image_height),                               \
            static_cast<uint32_t>(tile_size),                                  \
            tile_offsets,                                                      \
            flatten_ids,                                                       \
            render_colors,                                                     \
            render_alphas,                                                     \
            last_ids,                                                          \
            median_ids,                                                        \
            v_render_colors,                                                   \
            v_render_alphas,                                                   \
            v_render_normals,                                                  \
            v_render_distort,                                                  \
            v_render_median,                                                   \
            absgrad ? c10::optional<at::Tensor>(v_means2d_abs) : c10::nullopt, \
            v_means2d,                                                         \
            v_ray_transforms,                                                  \
            v_colors,                                                          \
            v_opacities,                                                       \
            v_normals,                                                         \
            v_densify                                                          \
        );                                                                     \
        break;

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    switch(channels)
    {
        __LAUNCH_KERNEL__(1)
        __LAUNCH_KERNEL__(2)
        __LAUNCH_KERNEL__(3)
        __LAUNCH_KERNEL__(4)
        __LAUNCH_KERNEL__(5)
        __LAUNCH_KERNEL__(8)
        __LAUNCH_KERNEL__(9)
        __LAUNCH_KERNEL__(16)
        __LAUNCH_KERNEL__(17)
        __LAUNCH_KERNEL__(32)
        __LAUNCH_KERNEL__(33)
        __LAUNCH_KERNEL__(64)
        __LAUNCH_KERNEL__(65)
        __LAUNCH_KERNEL__(128)
        __LAUNCH_KERNEL__(129)
        __LAUNCH_KERNEL__(256)
        __LAUNCH_KERNEL__(257)
        __LAUNCH_KERNEL__(512)
        __LAUNCH_KERNEL__(513)
    default: AT_ERROR("Unsupported number of channels: ", channels);
    }
#    undef __LAUNCH_KERNEL__

    return std::make_tuple(v_means2d_abs, v_means2d, v_ray_transforms, v_colors, v_opacities, v_normals, v_densify);
}

std::tuple<at::Tensor, at::Tensor> rasterize_to_indices_ortho_2dgs(
    int64_t range_start,
    int64_t range_end,                // iteration steps
    const at::Tensor &transmittances, // [..., image_height, image_width]
    // Gaussian parameters
    const at::Tensor &means2d,        // [..., N, 2]
    const at::Tensor &ray_transforms, // [..., N, 3, 3]
    const at::Tensor &opacities,      // [..., N]
    // image size
    int64_t image_width,
    int64_t image_height,
    int64_t tile_size,
    // intersections
    const at::Tensor &tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor &flatten_ids   // [n_isects]
)
{
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    auto opt   = means2d.options();
    uint32_t N = means2d.size(-2);          // number of gaussians
    uint32_t I = means2d.numel() / (2 * N); // number of images

    uint32_t n_isects = flatten_ids.size(0);

    // First pass: count the number of gaussians that contribute to each pixel
    int64_t n_elems;
    at::Tensor chunk_starts;
    if(n_isects)
    {
        at::Tensor chunk_cnts = at::zeros({I * image_height * image_width}, opt.dtype(at::kInt));
        launch_rasterize_to_indices_ortho_2dgs_kernel(
            range_start,
            range_end,
            transmittances,
            means2d,
            ray_transforms,
            opacities,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            flatten_ids,
            c10::nullopt, // chunk_starts
            at::optional<at::Tensor>(chunk_cnts),
            c10::nullopt, // gaussian_ids
            c10::nullopt  // pixel_ids
        );
        at::Tensor cumsum = at::cumsum(chunk_cnts, 0, chunk_cnts.scalar_type());
        n_elems           = cumsum[-1].item<int64_t>();
        chunk_starts      = at::sub(cumsum, chunk_cnts);
    }
    else
    {
        n_elems = 0;
    }

    // Second pass: allocate memory and write out the gaussian and pixel ids.
    at::Tensor gaussian_ids = at::empty({n_elems}, opt.dtype(at::kLong));
    at::Tensor pixel_ids    = at::empty({n_elems}, opt.dtype(at::kLong));
    if(n_elems)
    {
        launch_rasterize_to_indices_ortho_2dgs_kernel(
            range_start,
            range_end,
            transmittances,
            means2d,
            ray_transforms,
            opacities,
            image_width,
            image_height,
            tile_size,
            tile_offsets,
            flatten_ids,
            at::optional<at::Tensor>(chunk_starts),
            c10::nullopt, // chunk_cnts
            at::optional<at::Tensor>(gaussian_ids),
            at::optional<at::Tensor>(pixel_ids)
        );
    }
    return std::make_tuple(gaussian_ids, pixel_ids);
}

TORCH_LIBRARY_FRAGMENT(gsplat, m)
{
    m.def(
        "ortho_projection_2dgs_fused_fwd(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int "
        "image_width, int image_height, float eps2d, float near_plane, float far_plane, float radius_clip) -> (Tensor, "
        "Tensor, Tensor, Tensor, Tensor)"
    );
    m.def(
        "ortho_projection_2dgs_fused_bwd(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int "
        "image_width, int image_height, Tensor radii, Tensor ray_transforms, Tensor v_means2d, Tensor v_depths, Tensor "
        "v_normals, Tensor v_ray_transforms, bool viewmats_requires_grad) -> (Tensor, Tensor, Tensor, Tensor)"
    );
    m.def(
        "ortho_projection_2dgs_packed_fwd(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int "
        "image_width, int image_height, float near_plane, float far_plane, float radius_clip) -> (Tensor, Tensor, "
        "Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    );
    m.def(
        "ortho_projection_2dgs_packed_bwd(Tensor means, Tensor quats, Tensor scales, Tensor viewmats, Tensor Ks, int "
        "image_width, int image_height, Tensor batch_ids, Tensor camera_ids, Tensor gaussian_ids, Tensor "
        "ray_transforms, Tensor v_means2d, Tensor v_depths, Tensor v_ray_transforms, Tensor v_normals, bool "
        "viewmats_requires_grad, bool sparse_grad) -> (Tensor, Tensor, Tensor, Tensor)"
    );
    m.def(
        "rasterize_to_pixels_ortho_2dgs_fwd(Tensor means2d, Tensor ray_transforms, Tensor colors, Tensor opacities, "
        "Tensor normals, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int tile_size, Tensor "
        "tile_offsets, Tensor flatten_ids) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)"
    );
    m.def(
        "rasterize_to_pixels_ortho_2dgs_bwd(Tensor means2d, Tensor ray_transforms, Tensor colors, Tensor opacities, "
        "Tensor normals, Tensor densify, Tensor? backgrounds, Tensor? masks, int image_width, int image_height, int "
        "tile_size, Tensor tile_offsets, Tensor flatten_ids, Tensor render_colors, Tensor render_alphas, Tensor "
        "last_ids, Tensor median_ids, Tensor v_render_colors, Tensor v_render_alphas, Tensor v_render_normals, Tensor "
        "v_render_distort, Tensor v_render_median, bool absgrad) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, "
        "Tensor)"
    );
    m.def(
        "rasterize_to_indices_ortho_2dgs(int range_start, int range_end, Tensor transmittances, Tensor means2d, Tensor "
        "ray_transforms, Tensor opacities, int image_width, int image_height, int tile_size, Tensor tile_offsets, "
        "Tensor flatten_ids) -> (Tensor, Tensor)"
    );
}

TORCH_LIBRARY_IMPL(gsplat, CUDA, m)
{

    m.impl("ortho_projection_2dgs_fused_fwd", &ortho_projection_2dgs_fused_fwd);
    m.impl("ortho_projection_2dgs_fused_bwd", &ortho_projection_2dgs_fused_bwd);
    m.impl("ortho_projection_2dgs_packed_fwd", &ortho_projection_2dgs_packed_fwd);
    m.impl("ortho_projection_2dgs_packed_bwd", &ortho_projection_2dgs_packed_bwd);
    m.impl("rasterize_to_pixels_ortho_2dgs_fwd", &rasterize_to_pixels_ortho_2dgs_fwd);
    m.impl("rasterize_to_pixels_ortho_2dgs_bwd", &rasterize_to_pixels_ortho_2dgs_bwd);
    m.impl("rasterize_to_indices_ortho_2dgs", &rasterize_to_indices_ortho_2dgs);
}
#endif
} // namespace gsplat
