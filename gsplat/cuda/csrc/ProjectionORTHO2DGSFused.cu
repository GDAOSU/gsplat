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
        *__restrict__ means, // [B, N, 3]:  Gaussian means. (i.e. source points)
    const scalar_t *__restrict__ quats_ptr,  // [B, N, 4]:  Quaternions (No need
                                             // to be normalized): This is the
                                             // rotation component (for 2D)
    const scalar_t *__restrict__ scales_ptr, // [B, N, 3]:  Scales. [B, N, 3]
                                             // scales for x, y, z
    const scalar_t *__restrict__ viewmats_ptr, // [B, C, 4, 4]:  World-to-Camera
                                               // coordinate mat [R t] [0 1]
    const scalar_t
        *__restrict__ Ks_ptr, // [B, C, 3, 3]:  Projective transformation matrix
                              // [f_x 0  c_x]
                              // [0  f_y c_y]
                              // [0   0   1]  : f_x, f_y are focal lengths, c_x,
                              // c_y is coords for camera center on screen space
    const uint32_t image_width,  // Image width  pixels
    const uint32_t image_height, // Image height pixels
    const scalar_t
        near_plane, // Near clipping plane (for finite range used in z sorting)
    const scalar_t
        far_plane, // Far clipping plane (for finite range used in z sorting)
    const scalar_t radius_clip, // Radius clipping threshold (through away small
                                // primitives)
    // outputs
    int32_t *__restrict__ radii_ptr, // [B, C, N, 2]   The maximum radius of the
                                     // projected Gaussians in pixel unit. Int32
                                     // tensor.
    scalar_t *__restrict__ means2d_ptr, // [B, C, N, 2] 2D means of the
                                        // projected Gaussians.
    scalar_t *__restrict__ depths_ptr, // [B, C, N] The z-depth of the projected
                                       // Gaussians.
    scalar_t *__restrict__ M_i2u_ptr, // [B, C, N, 3, 3] Transformation matrices
                                      // that transform xy-planes in pixel
                                      // spaces into splat coordinates (WH)^T in
                                      // equation (9) in paper
    scalar_t *__restrict__ normals_ptr // [B, C, N, 3] World-space normals,
                                       // flipped to face the affine camera.
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
    const vec3 p_k = glm::make_vec3(
        means + bid * N * 3 + gid * 3
    ); // find the mean of the primitive this thread is responsible for
    viewmats_ptr += bid * C * 16 + cid * 16; // step 4x4 camera matrix
    Ks_ptr += bid * C * 9 + cid * 9;         // step 3x3 intrinsic matrix

    // linear coefficient of the 2x3 affine camera. Explicit Transpose
    // glm is column-major but input is row-major
    const mat3 A33 = mat3(
        viewmats_ptr[0],
        viewmats_ptr[4],
        viewmats_ptr[8], // 1st column
        viewmats_ptr[1],
        viewmats_ptr[5],
        viewmats_ptr[9], // 2nd column
        viewmats_ptr[2],
        viewmats_ptr[6],
        viewmats_ptr[10] // 3rd column
    );
    // offset of affine camera, make the last component 1
    const vec3 b31 = vec3(viewmats_ptr[3], viewmats_ptr[7], viewmats_ptr[11]);

    // transform Gaussian center to camera space
    const vec3 p_camera = A33 * p_k + b31;

    // return this thread for overly small primitives
    if (p_camera.z <= near_plane || p_camera.z >= far_plane) {
        radii_ptr[idx * 2] = 0;
        radii_ptr[idx * 2 + 1] = 0;
        return;
    }

    const scalar_t fx = Ks_ptr[0];
    const scalar_t fy = Ks_ptr[4];
    const scalar_t cx = Ks_ptr[2];
    const scalar_t cy = Ks_ptr[5];

    const vec2 p_image(fx * p_camera.x + cx, fy * p_camera.y + cy);

    quats_ptr += bid * N * 4 + gid * 4;
    scales_ptr += bid * N * 3 + gid * 3;

    // The third column is the surface normal
    const mat3 R33 = quat_to_rotmat(glm::make_vec4(quats_ptr));
    const mat2x3 RS32(R33[0] * scales_ptr[0], R33[1] * scales_ptr[1]);

    const mat2 ARS22 = mat2(
        sum(glm::row(A33, 0) * RS32[0]),
        sum(glm::row(A33, 1) * RS32[0]), // 1st column
        sum(glm::row(A33, 0) * RS32[1]),
        sum(glm::row(A33, 1) * RS32[1]) // 2nd column
    );

    const mat2 KARS = mat2(
        fx * ARS22[0][0],
        fy * ARS22[0][1], // 1st column
        fx * ARS22[1][0],
        fy * ARS22[1][1] // 2nd column
    );

    // M_u2i is the inverse of the affine transformation matrix
    const mat3 M_u2i(
        KARS[0][0],
        KARS[0][1],
        0.0, // 1st column
        KARS[1][0],
        KARS[1][1],
        0.0, // 2nd column
        p_image.x,
        p_image.y,
        1.0
    );

    const scalar_t det_M = glm::determinant(M_u2i);
    // ill-conditioned primitives will have det(M) = 0.0f, we ignore them
    if (det_M == 0.0f) {
        radii_ptr[idx * 2] = 0;
        radii_ptr[idx * 2 + 1] = 0;
        return; // skip if the determinant is zero
    }

    const mat3 M_i2u = glm::inverse(M_u2i);
    /**
     * ===============================================
     * Compute AABB
     * ===============================================
     */
    const float extent_x = max(
        1e-2f,
        sqrtf(KARS[0][0] * KARS[0][0] + KARS[1][0] * KARS[1][0])
    );
    const float extent_y = max(
        1e-2f,
        sqrtf(KARS[0][1] * KARS[0][1] + KARS[1][1] * KARS[1][1])
    );
    const float radius_x = ceil(3.33f * extent_x);
    const float radius_y = ceil(3.33f * extent_y);

    if (radius_x <= radius_clip && radius_y <= radius_clip) {
        radii_ptr[idx * 2] = 0;
        radii_ptr[idx * 2 + 1] = 0;
        return;
    }

    // CULLING STEP:
    // mask out gaussians outside the image region
    if (p_image.x + radius_x <= 0 || p_image.x - radius_x >= image_width ||
        p_image.y + radius_y <= 0 || p_image.y - radius_y >= image_height) {
        radii_ptr[idx * 2] = 0;
        radii_ptr[idx * 2 + 1] = 0;
        return;
    }

    // Keep ortho normals in world space. A general affine camera can include
    // scale/shear, so applying A33 would not preserve true surface normals.
    vec3 normal = R33[2]; // the normal is in world space
    // flip normal if it is pointing away from the camera
    const float multiplier = glm::dot(-normal, glm::row(A33, 2)) > 0 ? 1 : -1;
    normal *= multiplier;


    // write to outputs
    radii_ptr[idx * 2] = (int32_t)radius_x;
    radii_ptr[idx * 2 + 1] = (int32_t)radius_y;
    means2d_ptr[idx * 2] = p_image.x;
    means2d_ptr[idx * 2 + 1] = p_image.y;
    depths_ptr[idx] = p_camera.z;

    // row major storing
    M_i2u_ptr[idx * 9] = M_i2u[0][0];
    M_i2u_ptr[idx * 9 + 1] = M_i2u[1][0];
    M_i2u_ptr[idx * 9 + 2] = M_i2u[2][0]; // 1st row
    M_i2u_ptr[idx * 9 + 3] = M_i2u[0][1];
    M_i2u_ptr[idx * 9 + 4] = M_i2u[1][1];
    M_i2u_ptr[idx * 9 + 5] = M_i2u[2][1]; // 2nd row
    M_i2u_ptr[idx * 9 + 6] = M_i2u[0][2];
    M_i2u_ptr[idx * 9 + 7] = M_i2u[1][2];
    M_i2u_ptr[idx * 9 + 8] = M_i2u[2][2]; // 3rd row

    // primitive normals
    normals_ptr[idx * 3] = normal.x;
    normals_ptr[idx * 3 + 1] = normal.y;
    normals_ptr[idx * 3 + 2] = normal.z;
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
    at::Tensor radii,   // [..., C, N, 2]
    at::Tensor means2d, // [..., C, N, 2]
    at::Tensor depths,  // [..., C, N]
    at::Tensor M_i2u_,  // [..., C, N, 3, 3]
    at::Tensor normals  // [..., C, N, 3]
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
            M_i2u_.data_ptr<float>(),
            normals.data_ptr<float>()
        );
}

template <typename scalar_t>
__global__ void ortho_projection_2dgs_fused_bwd_kernel(
    // fwd inputs
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const scalar_t *__restrict__ means_ptr,    // [B, N, 3]
    const scalar_t *__restrict__ quats_ptr,    // [B, N, 4]
    const scalar_t *__restrict__ scales_ptr,   // [B, N, 3]
    const scalar_t *__restrict__ viewmats_ptr, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks_ptr,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    // fwd outputs
    const int32_t *__restrict__ radii_ptr,  // [B, C, N, 2]
    const scalar_t *__restrict__ M_i2u_ptr, // [B, C, N, 3, 3]
    // grad outputs
    const scalar_t *__restrict__ v_means2d_ptr, // [B, C, N, 2]
    const scalar_t *__restrict__ v_depths_ptr,  // [B, C, N]
    const scalar_t *__restrict__ v_normals_ptr, // [B, C, N, 3]
    const scalar_t *__restrict__ v_M_i2u_ptr,   // [B, C, N, 3, 3]
    // grad inputs
    scalar_t *__restrict__ v_means_ptr,   // [B, N, 3]
    scalar_t *__restrict__ v_quats_ptr,   // [B, N, 4]
    scalar_t *__restrict__ v_scales_ptr,  // [B, N, 3]
    scalar_t *__restrict__ v_viewmats_ptr // [B, C, 4, 4]
) {
    // NOTE: Variables are named after Matrix[row][column], in row-major
    // convention. While in GLM, the default is Matrix[column]x[row]. Please be
    // aware of the difference.

    //  parallelize over C * N.
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= B * C * N || radii_ptr[idx * 2] <= 0 ||
        radii_ptr[idx * 2 + 1] <= 0) {
        return;
    }
    const uint32_t bid = idx / (C * N); // batch id
    const uint32_t cid = (idx / N) % C; // camera id
    const uint32_t gid = idx % N;       // gaussian id

    // shift pointers to the current camera and gaussian
    means_ptr += bid * N * 3 + gid * 3;
    viewmats_ptr += bid * C * 16 + cid * 16;
    Ks_ptr += bid * C * 9 + cid * 9;
    M_i2u_ptr += idx * 9;
    v_means2d_ptr += idx * 2;
    v_depths_ptr += idx;
    v_normals_ptr += idx * 3;
    v_M_i2u_ptr += idx * 9;

    const vec3 p_k = glm::make_vec3(means_ptr);
    const vec4 quat = glm::make_vec4(quats_ptr + bid * N * 4 + gid * 4);
    const vec2 scale = glm::make_vec2(scales_ptr + bid * N * 3 + gid * 3);

    // Reconstruct forward pass intermediate variables
    const mat3 A33 = mat3(
        viewmats_ptr[0],
        viewmats_ptr[4],
        viewmats_ptr[8], // 1st column
        viewmats_ptr[1],
        viewmats_ptr[5],
        viewmats_ptr[9], // 2nd column
        viewmats_ptr[2],
        viewmats_ptr[6],
        viewmats_ptr[10] // 3rd column
    );

    const mat3x2 A23(
        A33[0][0],
        A33[0][1], // 1st column
        A33[1][0],
        A33[1][1], // 2nd column
        A33[2][0],
        A33[2][1] // 3rd column (offset)
    );

    // offset of affine camera, make the last component 1
    const vec3 b31 = vec3(viewmats_ptr[3], viewmats_ptr[7], viewmats_ptr[11]);
    const vec3 p_camera = A33 * p_k + b31;
    const scalar_t fx = Ks_ptr[0];
    const scalar_t fy = Ks_ptr[4];
    const scalar_t cx = Ks_ptr[2];
    const scalar_t cy = Ks_ptr[5];

    const vec2 p_image(fx * p_camera.x + cx, fy * p_camera.y + cy);

    const mat3 R33 = quat_to_rotmat(quat);
    const mat3 S33 = mat3(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, 1.0);
    const mat3 RS33 = R33 * S33;

    // normals dual visible
    const vec3& normal = R33[2]; // the normal is in world space
    const float multiplier = glm::dot(-normal, glm::row(A33, 2)) > 0 ? 1 : -1;

    const mat2 ARS22 = mat2(
        sum(glm::row(A33, 0) * RS33[0]),
        sum(glm::row(A33, 1) * RS33[0]),
        sum(glm::row(A33, 0) * RS33[1]),
        sum(glm::row(A33, 1) * RS33[1])
    );

    const mat2 KARS22 = mat2(
        fx * ARS22[0][0],
        fy * ARS22[0][1], // 1st column
        fx * ARS22[1][0],
        fy * ARS22[1][1] // 2nd column
    );

    const mat3 M_u2i = mat3(
        KARS22[0][0],
        KARS22[0][1],
        0.0, // 1st column
        KARS22[1][0],
        KARS22[1][1],
        0.0, // 2nd column
        p_image.x,
        p_image.y,
        1.0
    );

    // Load M_i2u from cache
    const mat3 M_i2u = mat3(
        M_i2u_ptr[0],
        M_i2u_ptr[3],
        M_i2u_ptr[6], // 1st column
        M_i2u_ptr[1],
        M_i2u_ptr[4],
        M_i2u_ptr[7], // 2nd column
        M_i2u_ptr[2],
        M_i2u_ptr[5],
        M_i2u_ptr[8] // 3rd column
    );

    // Load input gradients
    const mat3 v_M_i2u_val = mat3(
        v_M_i2u_ptr[0],
        v_M_i2u_ptr[3],
        v_M_i2u_ptr[6], // 1st column
        v_M_i2u_ptr[1],
        v_M_i2u_ptr[4],
        v_M_i2u_ptr[7], // 2nd column
        v_M_i2u_ptr[2],
        v_M_i2u_ptr[5],
        v_M_i2u_ptr[8] // 3rd column
    );                 // w.r.t. A33, b31, means, quat, scale

    vec2 v_means2d_val(v_means2d_ptr[0], v_means2d_ptr[1]); // w.r.t. A33, b31, and means
    const scalar_t v_depths_val = v_depths_ptr[0];      // w.r.t, A33, b31, and means
    const vec3 v_normals_val = glm::make_vec3(v_normals_ptr); // w.r.t. quat, A33

    /////////////////////////////
    // Initialize gradient accumulators
    vec4 v_quat(0.f);
    vec2 v_scale(0.f);
    vec3 v_mean(0.f);
    mat3 v_A33(0.f); // part of viewmat
    vec3 v_b31(0.f); // part of viewmat

    ///////////////////////////////////
    // Backward pass

    // STEP 1: M_i2u -> M_u2i
    // FWD: M_i2u = inv(M_u2i)
    // BWD: v_M_u2i = -M_i2u^T * v_M_i2u * M_i2u^T
    mat3 v_M_u2i_val =
        -glm::transpose(M_i2u) * v_M_i2u_val * glm::transpose(M_i2u);

    // STEP 2: M_u2i -> p_k_image(means2d), and accumulate with v_means2d
    // FWD: M_u2i = [  KARS22 | p_k_image ]
    //              [ 0 0 0 |     1     ]
    v_means2d_val.x += v_M_u2i_val[2][0]; // p_image.x
    v_means2d_val.y += v_M_u2i_val[2][1]; // p_image.y

    mat2 v_KARS22 = mat2(
        v_M_u2i_val[0][0],
        v_M_u2i_val[0][1], // 1st column
        v_M_u2i_val[1][0],
        v_M_u2i_val[1][1] // 2nd column
    );

    // STEP 3: p_image ---> p_camera[:2]
    //         depth  ---> p_camera[2]
    // FWD: p_image = K[:2,:2] * p_camera + K[:2,2]
    //      depths = p_camera.z
    // BWD: v_p_camera = K[:2,:2]^T * v_p_image
    vec3 v_p_camera(v_means2d_val.x * fx, v_means2d_val.y * fy, v_depths_val);

    // STEP 4: p_camera -> p_k, A33, b31
    // FWD: p_camera = A33 * p_k + b31
    // BWD: v_p_k (aka. v_mean) = A33^T * v_p_camera
    //      v_A33 = outer(v_p_camera, p_k)
    //      v_b31 = v_p_camera
    v_mean = glm::transpose(A33) * v_p_camera;
    v_A33 += glm::outerProduct(v_p_camera, p_k);
    v_b31 += v_p_camera;

    // STEP 5: KARS22 -> ARS22
    // FWD: KARS22 = K22 * ARS22 = K22 * (A23 * R33 * S32)
    // BWD: v_ARS22 = K22.T * v_KARS22

    mat2 v_ARS22(
        fx * v_KARS22[0][0],
        fy * v_KARS22[0][1], // 1st column
        fx * v_KARS22[1][0],
        fy * v_KARS22[1][1] // 2nd column
    );

    // STEP 6: ARS22 -> quat, scale, A33
    // FWD: ARS22 = A23 * RS32 = A23 * R33 * S32
    // BWD: v_A23 = v_ARS22 * RS32^T
    //      v_RS32 = A23^T * v_ARS22
    //      v_RS33 = v_RS32 * S32^T = v_RS32 * diag(scale)
    //      v_S32 = R33^T * v_RS32
    //         -> v_s0 = R33.c0 * v_RS32.c0
    //         -> v_s1 = R33.c1 * v_RS32.c1
    //
    const mat3x2 v_A23 = v_ARS22 * glm::transpose(mat2x3(RS33[0], RS33[1]));
    const mat2x3 v_RS32 = glm::transpose(A23) * v_ARS22;
    v_scale[0] += sum(R33[0] * v_RS32[0]);
    v_scale[1] += sum(R33[1] * v_RS32[1]);
    const mat3 v_R33(v_RS32[0] * scale[0], v_RS32[1] * scale[1], v_normals_val * multiplier);

    // accumulate v_A33
    #pragma unroll 
    for (int i = 0; i < 2; ++i)
        #pragma unroll
        for (int j = 0; j < 3; ++j)
            v_A33[j][i] += v_A23[j][i];

    quat_to_rotmat_vjp(quat, v_R33, v_quat);

    ////////////////////////////////////////////////

    // #if __CUDA_ARCH__ >= 700
    // write out results with warp-level reduction
    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    auto warp_group_g = cg::labeled_partition(warp, gid);
    if (v_means_ptr != nullptr) {
        warpSum(v_mean, warp_group_g);
        if (warp_group_g.thread_rank() == 0) {
            v_means_ptr += bid * N * 3 + gid * 3;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
                gpuAtomicAdd(v_means_ptr + i, v_mean[i]);
            }
        }
    }

    // Directly output gradients w.r.t. the quaternion and scale
    warpSum(v_quat, warp_group_g);
    warpSum(v_scale, warp_group_g);
    if (warp_group_g.thread_rank() == 0) {
        v_quats_ptr += bid * N * 4 + gid * 4;
        v_scales_ptr += bid * N * 3 + gid * 3;
        gpuAtomicAdd(v_quats_ptr, v_quat[0]);
        gpuAtomicAdd(v_quats_ptr + 1, v_quat[1]);
        gpuAtomicAdd(v_quats_ptr + 2, v_quat[2]);
        gpuAtomicAdd(v_quats_ptr + 3, v_quat[3]);
        gpuAtomicAdd(v_scales_ptr, v_scale[0]);
        gpuAtomicAdd(v_scales_ptr + 1, v_scale[1]);
    }

    if (v_viewmats_ptr != nullptr) {
        auto warp_group_c = cg::labeled_partition(warp, cid);
        warpSum(v_A33, warp_group_c);
        warpSum(v_b31, warp_group_c);
        if (warp_group_c.thread_rank() == 0) {
            v_viewmats_ptr += bid * C * 16 + cid * 16;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
#pragma unroll
                for (uint32_t j = 0; j < 3; j++) {
                    gpuAtomicAdd(v_viewmats_ptr + i * 4 + j, v_A33[j][i]);
                }
                gpuAtomicAdd(v_viewmats_ptr + i * 4 + 3, v_b31[i]);
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
    const at::Tensor radii,  // [..., C, N, 2]
    const at::Tensor M_i2u_, // [..., C, N, 3, 3]
    // grad outputs
    const at::Tensor v_means2d, // [..., C, N, 2]
    const at::Tensor v_depths,  // [..., C, N]
    const at::Tensor v_normals, // [..., C, N, 3]
    const at::Tensor v_M_i2u_,  // [..., C, N, 3, 3]
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
            M_i2u_.data_ptr<float>(),
            v_means2d.data_ptr<float>(),
            v_depths.data_ptr<float>(),
            v_normals.data_ptr<float>(),
            v_M_i2u_.data_ptr<float>(),
            v_means.data_ptr<float>(),
            v_quats.data_ptr<float>(),
            v_scales.data_ptr<float>(),
            viewmats_requires_grad ? v_viewmats.data_ptr<float>() : nullptr
        );
}

} // namespace gsplat
