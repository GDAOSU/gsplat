#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Projection.h"
#include "Projection2DGS.cuh" // Utils for 2DGS Projection
#include "Utils.cuh"

#include <glm/gtc/matrix_access.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/vec_swizzle.hpp>

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void ortho_projection_2dgs_fused_fwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t
        *__restrict__ p_k, // [B, N, 3]:  Gaussian means. (i.e. source points)
    const scalar_t *__restrict__ quats,  // [B, N, 4]:  Quaternions (No need to
                                         // be normalized): This is the rotation
                                         // component (for 2D)
    const scalar_t *__restrict__ scales, // [B, N, 3]:  Scales. [B, N, 3] scales
                                         // for x, y, z
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]:  World-to-Camera
                                           // coordinate mat [R t] [0 1]
    const scalar_t
        *__restrict__ Ks, // [B, C, 3, 3]:  Projective transformation matrix
                          // [f_x 0  c_x]
                          // [0  f_y c_y]
                          // [0   0   1]  : f_x, f_y are focal lengths, c_x, c_y
                          // is coords for camera center on screen space
    const uint32_t image_width,  // Image width  pixels
    const uint32_t image_height, // Image height pixels
    const scalar_t
        near_plane, // Near clipping plane (for finite range used in z sorting)
    const scalar_t
        far_plane, // Far clipping plane (for finite range used in z sorting)
    const scalar_t radius_clip, // Radius clipping threshold (through away small
                                // primitives)
    // outputs
    int32_t
        *__restrict__ radii, // [B, C, N, 2]   The maximum radius of the
                             // projected Gaussians in pixel unit. Int32 tensor.
    scalar_t *__restrict__ means2d, // [B, C, N, 2] 2D means of the projected
                                    // Gaussians.
    scalar_t *__restrict__ depths,  // [B, C, N] The z-depth of the projected
                                    // Gaussians.
    scalar_t
        *__restrict__ ray_transforms, // [B, C, N, 3, 3] Transformation matrices
                                      // that transform xy-planes in pixel
                                      // spaces into splat coordinates (WH)^T in
                                      // equation (9) in paper
    scalar_t *__restrict__ normals // [B, C, N, 3] The normals in camera spaces.
) {

    /**
     * ===============================================
     * Initialize execution and threading variables:
     * idx: global thread index
     * bid: batch id
     * cid: camera id
     * gid: gaussian id

     * THIS KERNEL LAUNCHES PER PRIMITIVE PER CAMERA i.e. C*N THREADS IN TOTAL
     * ===============================================
    */

    // parallelize over B * C * N.
    uint32_t idx =
        cg::this_grid().thread_rank(); // get the thread index from grid
    if (idx >= B * C * N) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N;       // gaussian id

    /**
     * ===============================================
     * Load data and put together camera rotation / translation
     * ===============================================
     */

    // shift pointers to the current camera and gaussian
    p_k +=
        bid * N * 3 +
        gid *
            3; // find the mean of the primitive this thread is responsible for
    viewmats += bid * C * 16 + cid * 16; // step 4x4 camera matrix
    Ks += bid * C * 9 + cid * 9;         // step 3x3 intrinsic matrix

    // linear coefficient of the 2x3 affine camera. Explicit Transpose
    // glm is column-major but input is row-major
    mat3 A33 = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    // offset of affine camera, make the last component 1
    vec3 b31 = vec3(viewmats[3], viewmats[7], viewmats[11]);

    // transform Gaussian center to camera space
    vec3 p_k_camera = A33 * glm::make_vec3(p_k) + b31;

    // return this thread for overly small primitives
    if (p_k_camera.z <= near_plane || p_k_camera.z >= far_plane) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    mat3 K = mat3(
        Ks[0],
        0.0,
        0.0, // 1st column
        0.0,
        Ks[4],
        0.0, // 2nd column
        Ks[2],
        Ks[5],
        1.0 // 3rd column
    );
    const vec2 p_k_image = glm::xy(K * vec3(p_k_camera.x, p_k_camera.y, 1.0));

    quats += bid * N * 4 + gid * 4;
    scales += bid * N * 3 + gid * 3;

    // The third column is the surface normal
    mat3 RS33 = quat_to_rotmat(glm::make_vec4(quats)) *
                mat3(scales[0], 0.0, 0.0, 0.0, scales[1], 0.0, 0.0, 0.0, 1.0);

    mat2 ARS = mat2(
        sum(glm::row(A33, 0) * RS33[0]),
        sum(glm::row(A33, 1) * RS33[0]), // 1st column
        sum(glm::row(A33, 0) * RS33[1]),
        sum(glm::row(A33, 1) * RS33[1]) // 2nd column
    );

    mat3 M_u2i = mat3(
        ARS[0][0],
        ARS[0][1],
        0.0, // 1st column
        ARS[1][0],
        ARS[1][1],
        0.0, // 2nd column
        p_k_camera.x,
        p_k_camera.y,
        1.0 // M_u2i is the inverse of the affine transformation matrix
    );
    M_u2i =
        K * M_u2i; // M_u2i is the inverse of the affine transformation matrix

    mat3 M_i2u = glm::inverse(M_u2i);

    /**
     * ===============================================
     * Compute AABB
     * ===============================================
     */
    vec3 tmp_p = M_u2i * vec3(1, 1, 1);
    // ==============================================
    const float radius_x = ceil(3.33f * max(1e-2, abs(tmp_p.x - p_k_image.x)));
    const float radius_y = ceil(3.33f * max(1e-2, abs(tmp_p.y - p_k_image.y)));

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // CULLING STEP:
    // mask out gaussians outside the image region
    if (p_k_image.x + radius_x <= 0 || p_k_image.x - radius_x >= image_width ||
        p_k_image.y + radius_y <= 0 || p_k_image.y - radius_y >= image_height) {
        radii[idx * 2] = 0;
        radii[idx * 2 + 1] = 0;
        return;
    }

    // normals dual visible
    vec3 normal = RS33[2]; // the normal is in world space

    // write to outputs
    radii[idx * 2] = (int32_t)radius_x;
    radii[idx * 2 + 1] = (int32_t)radius_y;
    means2d[idx * 2] = p_k_image.x;
    means2d[idx * 2 + 1] = p_k_image.y;
    depths[idx] = p_k_camera.z;

    // row major storing
    ray_transforms[idx * 9] = M_i2u[0][0];
    ray_transforms[idx * 9 + 1] = M_i2u[1][0];
    ray_transforms[idx * 9 + 2] = M_i2u[2][0]; // 1st row
    ray_transforms[idx * 9 + 3] = M_i2u[0][1];
    ray_transforms[idx * 9 + 4] = M_i2u[1][1];
    ray_transforms[idx * 9 + 5] = M_i2u[2][1]; // 2nd row
    ray_transforms[idx * 9 + 6] = M_i2u[0][2];
    ray_transforms[idx * 9 + 7] = M_i2u[1][2];
    ray_transforms[idx * 9 + 8] = M_i2u[2][2]; // 3rd row

    // primitive normals
    normals[idx * 3] = normal.x;
    normals[idx * 3 + 1] = normal.y;
    normals[idx * 3 + 2] = normal.z;
}

void launch_ortho_projection_2dgs_fused_fwd_kernel(
    // inputs
    const at::Tensor means,    // [..., N, 3]
    const at::Tensor quats,    // [..., N, 4]
    const at::Tensor scales,   // [..., N, 3]
    const at::Tensor viewmats, // [..., C, 4, 4]
    const at::Tensor Ks,       // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    // outputs
    at::Tensor radii,          // [..., C, N, 2]
    at::Tensor means2d,        // [..., C, N, 2]
    at::Tensor depths,         // [..., C, N]
    at::Tensor ray_transforms, // [..., C, N, 3, 3]
    at::Tensor normals         // [..., C, N, 3]
) {
    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t B = means.numel() / (N * 3); // number of batches
    uint32_t C = viewmats.size(-3);       // number of cameras

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    ortho_projection_2dgs_fused_fwd_kernel<float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            B,
            C,
            N,
            means.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            near_plane,
            far_plane,
            radius_clip,
            radii.data_ptr<int32_t>(),
            means2d.data_ptr<float>(),
            depths.data_ptr<float>(),
            ray_transforms.data_ptr<float>(),
            normals.data_ptr<float>()
        );
}

template <typename scalar_t>
__global__ void ortho_projection_2dgs_fused_bwd_kernel(
    // fwd inputs
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means,    // [B, N, 3]
    const scalar_t *__restrict__ quats,    // [B, N, 4]
    const scalar_t *__restrict__ scales,   // [B, N, 3]
    const scalar_t *__restrict__ viewmats, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const int32_t *__restrict__ radii,           // [B, C, N, 2]
    const scalar_t *__restrict__ ray_transforms, // [B, C, N, 3, 3]
    // grad outputs
    const scalar_t *__restrict__ v_means2d,        // [B, C, N, 2]
    const scalar_t *__restrict__ v_depths,         // [B, C, N]
    const scalar_t *__restrict__ v_normals,        // [B, C, N, 3]
    const scalar_t *__restrict__ v_ray_transforms, // [B, C, N, 3, 3]
    // grad inputs
    scalar_t *__restrict__ v_means,   // [B, N, 3]
    scalar_t *__restrict__ v_quats,   // [B, N, 4]
    scalar_t *__restrict__ v_scales,  // [B, N, 3]
    scalar_t *__restrict__ v_viewmats // [B, C, 4, 4]
) {
    //TODO(SXS): Need update
    // parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N || radii[idx * 2] <= 0 || radii[idx * 2 + 1] <= 0) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N;       // gaussian id

    // shift pointers to the current camera and gaussian
    means += bid * N * 3 + gid * 3;
    viewmats += bid * C * 16 + cid * 16;
    Ks += bid * C * 9 + cid * 9;

    ray_transforms += idx * 9;

    v_means2d += idx * 2;
    v_depths += idx;
    v_normals += idx * 3;
    v_ray_transforms += idx * 9;

    // transform Gaussian to camera space
    mat3 R = mat3(
        viewmats[0],
        viewmats[4],
        viewmats[8], // 1st column
        viewmats[1],
        viewmats[5],
        viewmats[9], // 2nd column
        viewmats[2],
        viewmats[6],
        viewmats[10] // 3rd column
    );
    vec3 t = vec3(viewmats[3], viewmats[7], viewmats[11]);
    vec3 mean_w = glm::make_vec3(means);
    vec3 mean_c;
    posW2C(R, t, mean_w, mean_c);

    vec4 quat = glm::make_vec4(quats + bid * N * 4 + gid * 4);
    vec2 scale = glm::make_vec2(scales + bid * N * 3 + gid * 3);

    mat3 P = mat3(Ks[0], 0.0, Ks[2], 0.0, Ks[4], Ks[5], 0.0, 0.0, 1.0);

    mat3 _v_ray_transforms = mat3(
        v_ray_transforms[0],
        v_ray_transforms[1],
        v_ray_transforms[2],
        v_ray_transforms[3],
        v_ray_transforms[4],
        v_ray_transforms[5],
        v_ray_transforms[6],
        v_ray_transforms[7],
        v_ray_transforms[8]
    );

    _v_ray_transforms[2][2] += v_depths[0];

    vec3 v_normal = glm::make_vec3(v_normals);

    vec3 v_mean(0.f);
    vec2 v_scale(0.f);
    vec4 v_quat(0.f);
    mat3 v_R(0.f);
    vec3 v_t(0.f);
    compute_ray_transforms_aabb_vjp(
        ray_transforms,
        v_means2d,
        v_normal,
        R,
        P,
        t,
        mean_w,
        mean_c,
        quat,
        scale,
        _v_ray_transforms,
        v_quat,
        v_scale,
        v_mean,
        v_R,
        v_t
    );

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means += bid * N * 3 + gid * 3;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means + i, v_mean[i]);
            }
        }
    }

    // Directly output gradients w.r.t. the quaternion and scale
    warpSum(v_quat, warp_group_g);
    warpSum(v_scale, warp_group_g);
    if (warp_group_g.thread_rank() == 0) {
        v_quats += bid * N * 4 + gid * 4;
        v_scales += bid * N * 3 + gid * 3;
        gpuAtomicAdd(v_quats, v_quat[0]);
        gpuAtomicAdd(v_quats + 1, v_quat[1]);
        gpuAtomicAdd(v_quats + 2, v_quat[2]);
        gpuAtomicAdd(v_quats + 3, v_quat[3]);
        gpuAtomicAdd(v_scales, v_scale[0]);
        gpuAtomicAdd(v_scales + 1, v_scale[1]);
    }

    if (v_viewmats != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_R, warp_group_c);
        warpSum(v_t, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats += bid * C * 16 + cid * 16;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
#pragma unroll
                for (uint32_t j = 0; j < 3; j++) {
                    gpuAtomicAdd(v_viewmats + i * 4 + j, v_R[j][i]);
                }
                gpuAtomicAdd(v_viewmats + i * 4 + 3, v_t[i]);
            }
        }
    }
}

void launch_ortho_projection_2dgs_fused_bwd_kernel(
    // fwd inputs
    const at::Tensor means,    // [..., N, 3]
    const at::Tensor quats,    // [..., N, 4]
    const at::Tensor scales,   // [..., N, 3]
    const at::Tensor viewmats, // [..., C, 4, 4]
    const at::Tensor Ks,       // [..., C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const at::Tensor radii,          // [..., C, N, 2]
    const at::Tensor ray_transforms, // [..., C, N, 3, 3]
    // grad outputs
    const at::Tensor v_means2d,        // [..., C, N, 2]
    const at::Tensor v_depths,         // [..., C, N]
    const at::Tensor v_normals,        // [..., C, N, 3]
    const at::Tensor v_ray_transforms, // [..., C, N, 3, 3]
    const bool viewmats_requires_grad,
    // outputs
    at::Tensor v_means,   // [..., N, 3]
    at::Tensor v_quats,   // [..., N, 4]
    at::Tensor v_scales,  // [..., N, 3]
    at::Tensor v_viewmats // [..., C, 4, 4]
) {
    uint32_t N = means.size(-2);          // number of gaussians
    uint32_t B = means.numel() / (N * 3); // number of batches
    uint32_t C = viewmats.size(-3);       // number of cameras

    int64_t n_elements = B * C * N;
    dim3 threads(256);
    dim3 grid((n_elements + threads.x - 1) / threads.x);
    int64_t shmem_size = 0; // No shared memory used in this kernel

    if (n_elements == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    ortho_projection_2dgs_fused_bwd_kernel<float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            B,
            C,
            N,
            means.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            radii.data_ptr<int32_t>(),
            ray_transforms.data_ptr<float>(),
            v_means2d.data_ptr<float>(),
            v_depths.data_ptr<float>(),
            v_normals.data_ptr<float>(),
            v_ray_transforms.data_ptr<float>(),
            v_means.data_ptr<float>(),
            v_quats.data_ptr<float>(),
            v_scales.data_ptr<float>(),
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
        );
}

} // namespace gsplat
