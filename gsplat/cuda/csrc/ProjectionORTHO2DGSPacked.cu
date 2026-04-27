#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

#include "Common.h"
#include "Projection.h"
#include "Projection2DGS.cuh"
#include "Utils.cuh"

#include <glm/gtc/matrix_access.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/vec_swizzle.hpp>

namespace gsplat {

namespace cg = cooperative_groups;

template <typename scalar_t>
__global__ void ortho_projection_2dgs_packed_fwd_kernel(
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
    const scalar_t near_plane,
    const scalar_t far_plane,
    const scalar_t radius_clip,
    const int32_t *__restrict__ block_accum, // [B * C * blocks_per_row]
    int32_t *__restrict__ block_cnts,        // [B * C * blocks_per_row]
    // outputs
    int32_t *__restrict__ indptr,       // [B * C + 1]
    int64_t *__restrict__ batch_ids,    // [nnz]
    int64_t *__restrict__ camera_ids,   // [nnz]
    int64_t *__restrict__ gaussian_ids, // [nnz]
    int32_t *__restrict__ radii,        // [nnz, 2]
    scalar_t *__restrict__ means2d,     // [nnz, 2]
    scalar_t *__restrict__ depths,      // [nnz]
    scalar_t *__restrict__ M_i2u_ptr,   // [nnz, 3, 3]
    scalar_t *__restrict__ normals      // [nnz, 3]
) {
    int32_t blocks_per_row = gridDim.x;
    int32_t row_idx = blockIdx.y;
    int32_t block_col_idx = blockIdx.x;
    int32_t block_idx = row_idx * blocks_per_row + block_col_idx;
    int32_t col_idx = block_col_idx * blockDim.x + threadIdx.x;
    const int32_t bid = row_idx / C;
    const int32_t cid = row_idx % C;
    const int32_t gid = col_idx;

    bool valid = (bid < B) && (cid < C) && (gid < N);

    vec2 p_image;
    vec3 p_camera;
    mat3 M_i2u;
    vec3 normal;
    float radius_x = 0.0f;
    float radius_y = 0.0f;

    if (valid) {
        means_ptr += bid * N * 3 + gid * 3;
        quats_ptr += bid * N * 4 + gid * 4;
        scales_ptr += bid * N * 3 + gid * 3;
        viewmats_ptr += bid * C * 16 + cid * 16;
        Ks_ptr += bid * C * 9 + cid * 9;

        const vec3 p_k = glm::make_vec3(means_ptr);
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
        const vec3 b31 =
            vec3(viewmats_ptr[3], viewmats_ptr[7], viewmats_ptr[11]);

        p_camera = A33 * p_k + b31;
        if (p_camera.z <= near_plane || p_camera.z >= far_plane) {
            valid = false;
        }

        if (valid) {
            const scalar_t fx = Ks_ptr[0];
            const scalar_t fy = Ks_ptr[4];
            const scalar_t cx = Ks_ptr[2];
            const scalar_t cy = Ks_ptr[5];
            p_image = vec2(fx * p_camera.x + cx, fy * p_camera.y + cy);

            const mat3 R33 = quat_to_rotmat(glm::make_vec4(quats_ptr));
            const mat2x3 RS32(R33[0] * scales_ptr[0], R33[1] * scales_ptr[1]);

            const mat2 ARS22 = mat2(
                sum(glm::row(A33, 0) * RS32[0]),
                sum(glm::row(A33, 1) * RS32[0]),
                sum(glm::row(A33, 0) * RS32[1]),
                sum(glm::row(A33, 1) * RS32[1])
            );

            const mat2 KARS = mat2(
                fx * ARS22[0][0],
                fy * ARS22[0][1],
                fx * ARS22[1][0],
                fy * ARS22[1][1]
            );

            const mat3 M_u2i = mat3(
                KARS[0][0],
                KARS[0][1],
                0.0,
                KARS[1][0],
                KARS[1][1],
                0.0,
                p_image.x,
                p_image.y,
                1.0
            );

            if (glm::determinant(M_u2i) == 0.0f) {
                valid = false;
            }

            if (valid) {
                M_i2u = glm::inverse(M_u2i);

                const float extent_x = max(
                    1e-2f,
                    sqrtf(KARS[0][0] * KARS[0][0] +
                          KARS[1][0] * KARS[1][0])
                );
                const float extent_y = max(
                    1e-2f,
                    sqrtf(KARS[0][1] * KARS[0][1] +
                          KARS[1][1] * KARS[1][1])
                );
                radius_x = ceil(3.33f * extent_x);
                radius_y = ceil(3.33f * extent_y);

                if (radius_x <= radius_clip && radius_y <= radius_clip) {
                    valid = false;
                }

                if (p_image.x + radius_x <= 0 ||
                    p_image.x - radius_x >= image_width ||
                    p_image.y + radius_y <= 0 ||
                    p_image.y - radius_y >= image_height) {
                    valid = false;
                }
            }

            if (valid) {
                normal = R33[2];
                const float multiplier =
                    glm::dot(-normal, glm::row(A33, 2)) > 0 ? 1 : -1;
                normal *= multiplier;
            }
        }
    }

    int32_t thread_data = static_cast<int32_t>(valid);
    if (block_cnts != nullptr) {
        int32_t aggregate;
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockReduce<int32_t, N_THREADS_PACKED> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;
            aggregate = BlockReduce(temp_storage).Sum(thread_data);
        } else {
            aggregate = 0;
        }
        if (threadIdx.x == 0) {
            block_cnts[block_idx] = aggregate;
        }
    } else {
        if (__syncthreads_or(thread_data)) {
            typedef cub::BlockScan<int32_t, N_THREADS_PACKED> BlockScan;
            __shared__ typename BlockScan::TempStorage temp_storage;
            BlockScan(temp_storage).ExclusiveSum(thread_data, thread_data);
        }
        if (valid) {
            if (block_idx > 0) {
                thread_data += block_accum[block_idx - 1];
            }
            batch_ids[thread_data] = bid;
            camera_ids[thread_data] = cid;
            gaussian_ids[thread_data] = gid;
            radii[thread_data * 2] = (int32_t)radius_x;
            radii[thread_data * 2 + 1] = (int32_t)radius_y;
            means2d[thread_data * 2] = p_image.x;
            means2d[thread_data * 2 + 1] = p_image.y;
            depths[thread_data] = p_camera.z;
            M_i2u_ptr[thread_data * 9] = M_i2u[0][0];
            M_i2u_ptr[thread_data * 9 + 1] = M_i2u[1][0];
            M_i2u_ptr[thread_data * 9 + 2] = M_i2u[2][0];
            M_i2u_ptr[thread_data * 9 + 3] = M_i2u[0][1];
            M_i2u_ptr[thread_data * 9 + 4] = M_i2u[1][1];
            M_i2u_ptr[thread_data * 9 + 5] = M_i2u[2][1];
            M_i2u_ptr[thread_data * 9 + 6] = M_i2u[0][2];
            M_i2u_ptr[thread_data * 9 + 7] = M_i2u[1][2];
            M_i2u_ptr[thread_data * 9 + 8] = M_i2u[2][2];
            normals[thread_data * 3] = normal.x;
            normals[thread_data * 3 + 1] = normal.y;
            normals[thread_data * 3 + 2] = normal.z;
        }
        if (threadIdx.x == 0 && block_col_idx == 0) {
            if (row_idx == 0) {
                indptr[0] = 0;
                indptr[B * C] = block_accum[B * C * blocks_per_row - 1];
            } else {
                indptr[row_idx] = block_accum[block_idx - 1];
            }
        }
    }
}

void launch_ortho_projection_2dgs_packed_fwd_kernel(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const float near_plane,
    const float far_plane,
    const float radius_clip,
    const at::optional<at::Tensor> block_accum,
    at::optional<at::Tensor> block_cnts,
    at::optional<at::Tensor> indptr,
    at::optional<at::Tensor> batch_ids,
    at::optional<at::Tensor> camera_ids,
    at::optional<at::Tensor> gaussian_ids,
    at::optional<at::Tensor> radii,
    at::optional<at::Tensor> means2d,
    at::optional<at::Tensor> depths,
    at::optional<at::Tensor> ray_transforms,
    at::optional<at::Tensor> normals
) {
    uint32_t N = means.size(-2);
    uint32_t B = means.numel() / (N * 3);
    uint32_t C = viewmats.size(-3);

    uint32_t nrows = B * C;
    uint32_t ncols = N;
    uint32_t blocks_per_row = (ncols + N_THREADS_PACKED - 1) / N_THREADS_PACKED;

    dim3 threads(N_THREADS_PACKED);
    dim3 grid(blocks_per_row, nrows, 1);
    int64_t shmem_size = 0;

    if (B == 0 || C == 0 || N == 0) {
        return;
    }

    ortho_projection_2dgs_packed_fwd_kernel<float>
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
            block_accum.has_value() ? block_accum.value().data_ptr<int32_t>()
                                    : nullptr,
            block_cnts.has_value() ? block_cnts.value().data_ptr<int32_t>()
                                   : nullptr,
            indptr.has_value() ? indptr.value().data_ptr<int32_t>() : nullptr,
            batch_ids.has_value() ? batch_ids.value().data_ptr<int64_t>()
                                  : nullptr,
            camera_ids.has_value() ? camera_ids.value().data_ptr<int64_t>()
                                   : nullptr,
            gaussian_ids.has_value() ? gaussian_ids.value().data_ptr<int64_t>()
                                     : nullptr,
            radii.has_value() ? radii.value().data_ptr<int32_t>() : nullptr,
            means2d.has_value() ? means2d.value().data_ptr<float>() : nullptr,
            depths.has_value() ? depths.value().data_ptr<float>() : nullptr,
            ray_transforms.has_value()
                ? ray_transforms.value().data_ptr<float>()
                : nullptr,
            normals.has_value() ? normals.value().data_ptr<float>() : nullptr
        );
}

template <typename scalar_t>
__global__ void ortho_projection_2dgs_packed_bwd_kernel(
    const uint32_t B,
    const uint32_t C,
    const uint32_t N,
    const uint32_t nnz,
    const scalar_t *__restrict__ means_ptr,    // [B, N, 3]
    const scalar_t *__restrict__ quats_ptr,    // [B, N, 4]
    const scalar_t *__restrict__ scales_ptr,   // [B, N, 3]
    const scalar_t *__restrict__ viewmats_ptr, // [B, C, 4, 4]
    const scalar_t *__restrict__ Ks_ptr,       // [B, C, 3, 3]
    const uint32_t image_width,
    const uint32_t image_height,
    const int64_t *__restrict__ batch_ids,       // [nnz]
    const int64_t *__restrict__ camera_ids,      // [nnz]
    const int64_t *__restrict__ gaussian_ids,    // [nnz]
    const scalar_t *__restrict__ M_i2u_ptr,      // [nnz, 3, 3]
    const scalar_t *__restrict__ v_means2d_ptr,  // [nnz, 2]
    const scalar_t *__restrict__ v_depths_ptr,   // [nnz]
    const scalar_t *__restrict__ v_M_i2u_ptr,    // [nnz, 3, 3]
    const scalar_t *__restrict__ v_normals_ptr,  // [nnz, 3]
    const bool sparse_grad,
    scalar_t *__restrict__ v_means_ptr,   // [B, N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_quats_ptr,   // [B, N, 4] or [nnz, 4]
    scalar_t *__restrict__ v_scales_ptr,  // [B, N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_viewmats_ptr // [B, C, 4, 4]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= nnz) {
        return;
    }

    const int64_t bid = batch_ids[idx];
    const int64_t cid = camera_ids[idx];
    const int64_t gid = gaussian_ids[idx];

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

    const mat3 A33 = mat3(
        viewmats_ptr[0],
        viewmats_ptr[4],
        viewmats_ptr[8],
        viewmats_ptr[1],
        viewmats_ptr[5],
        viewmats_ptr[9],
        viewmats_ptr[2],
        viewmats_ptr[6],
        viewmats_ptr[10]
    );

    const mat3x2 A23(
        A33[0][0],
        A33[0][1],
        A33[1][0],
        A33[1][1],
        A33[2][0],
        A33[2][1]
    );

    const vec3 b31 =
        vec3(viewmats_ptr[3], viewmats_ptr[7], viewmats_ptr[11]);
    const vec3 p_camera = A33 * p_k + b31;
    const scalar_t fx = Ks_ptr[0];
    const scalar_t fy = Ks_ptr[4];

    const mat3 R33 = quat_to_rotmat(quat);
    const mat3 S33 =
        mat3(scale[0], 0.0, 0.0, 0.0, scale[1], 0.0, 0.0, 0.0, 1.0);
    const mat3 RS33 = R33 * S33;

    const vec3 &normal = R33[2];
    const float multiplier = glm::dot(-normal, glm::row(A33, 2)) > 0 ? 1 : -1;

    const mat2 ARS22 = mat2(
        sum(glm::row(A33, 0) * RS33[0]),
        sum(glm::row(A33, 1) * RS33[0]),
        sum(glm::row(A33, 0) * RS33[1]),
        sum(glm::row(A33, 1) * RS33[1])
    );

    const mat2 KARS22 = mat2(
        fx * ARS22[0][0],
        fy * ARS22[0][1],
        fx * ARS22[1][0],
        fy * ARS22[1][1]
    );

    const mat3 M_i2u = mat3(
        M_i2u_ptr[0],
        M_i2u_ptr[3],
        M_i2u_ptr[6],
        M_i2u_ptr[1],
        M_i2u_ptr[4],
        M_i2u_ptr[7],
        M_i2u_ptr[2],
        M_i2u_ptr[5],
        M_i2u_ptr[8]
    );

    const mat3 v_M_i2u_val = mat3(
        v_M_i2u_ptr[0],
        v_M_i2u_ptr[3],
        v_M_i2u_ptr[6],
        v_M_i2u_ptr[1],
        v_M_i2u_ptr[4],
        v_M_i2u_ptr[7],
        v_M_i2u_ptr[2],
        v_M_i2u_ptr[5],
        v_M_i2u_ptr[8]
    );

    vec2 v_means2d_val(v_means2d_ptr[0], v_means2d_ptr[1]);
    const scalar_t v_depths_val = v_depths_ptr[0];
    const vec3 v_normals_val = glm::make_vec3(v_normals_ptr);

    vec4 v_quat(0.f);
    vec2 v_scale(0.f);
    vec3 v_mean(0.f);
    mat3 v_A33(0.f);
    vec3 v_b31(0.f);

    mat3 v_M_u2i_val =
        -glm::transpose(M_i2u) * v_M_i2u_val * glm::transpose(M_i2u);

    v_means2d_val.x += v_M_u2i_val[2][0];
    v_means2d_val.y += v_M_u2i_val[2][1];

    mat2 v_KARS22 = mat2(
        v_M_u2i_val[0][0],
        v_M_u2i_val[0][1],
        v_M_u2i_val[1][0],
        v_M_u2i_val[1][1]
    );

    vec3 v_p_camera(v_means2d_val.x * fx, v_means2d_val.y * fy, v_depths_val);
    v_mean = glm::transpose(A33) * v_p_camera;
    v_A33 += glm::outerProduct(v_p_camera, p_k);
    v_b31 += v_p_camera;

    mat2 v_ARS22(
        fx * v_KARS22[0][0],
        fy * v_KARS22[0][1],
        fx * v_KARS22[1][0],
        fy * v_KARS22[1][1]
    );

    const mat3x2 v_A23 =
        v_ARS22 * glm::transpose(mat2x3(RS33[0], RS33[1]));
    const mat2x3 v_RS32 = glm::transpose(A23) * v_ARS22;
    v_scale[0] += sum(R33[0] * v_RS32[0]);
    v_scale[1] += sum(R33[1] * v_RS32[1]);
    const mat3 v_R33(
        v_RS32[0] * scale[0],
        v_RS32[1] * scale[1],
        v_normals_val * multiplier
    );

#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            v_A33[j][i] += v_A23[j][i];
        }

    quat_to_rotmat_vjp(quat, v_R33, v_quat);

    auto warp = cg::tiled_partition<32>(cg::this_thread_block());
    if (sparse_grad) {
        if (v_means_ptr != nullptr) {
            v_means_ptr += idx * 3;
#pragma unroll
            for (uint32_t i = 0; i < 3; i++) {
                v_means_ptr[i] = v_mean[i];
            }
        }
        v_quats_ptr += idx * 4;
        v_scales_ptr += idx * 3;
        v_quats_ptr[0] = v_quat[0];
        v_quats_ptr[1] = v_quat[1];
        v_quats_ptr[2] = v_quat[2];
        v_quats_ptr[3] = v_quat[3];
        v_scales_ptr[0] = v_scale[0];
        v_scales_ptr[1] = v_scale[1];
    } else {
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

void launch_ortho_projection_2dgs_packed_bwd_kernel(
    const at::Tensor means,
    const at::Tensor quats,
    const at::Tensor scales,
    const at::Tensor viewmats,
    const at::Tensor Ks,
    const uint32_t image_width,
    const uint32_t image_height,
    const at::Tensor batch_ids,
    const at::Tensor camera_ids,
    const at::Tensor gaussian_ids,
    const at::Tensor ray_transforms,
    const at::Tensor v_means2d,
    const at::Tensor v_depths,
    const at::Tensor v_ray_transforms,
    const at::Tensor v_normals,
    const bool sparse_grad,
    at::Tensor v_means,
    at::Tensor v_quats,
    at::Tensor v_scales,
    at::optional<at::Tensor> v_viewmats
) {
    uint32_t N = means.size(-2);
    uint32_t B = means.numel() / (N * 3);
    uint32_t C = viewmats.size(-3);
    uint32_t nnz = camera_ids.size(0);

    dim3 threads(256);
    dim3 grid((nnz + threads.x - 1) / threads.x);
    int64_t shmem_size = 0;

    if (nnz == 0) {
        return;
    }

    ortho_projection_2dgs_packed_bwd_kernel<float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            B,
            C,
            N,
            nnz,
            means.data_ptr<float>(),
            quats.data_ptr<float>(),
            scales.data_ptr<float>(),
            viewmats.data_ptr<float>(),
            Ks.data_ptr<float>(),
            image_width,
            image_height,
            batch_ids.data_ptr<int64_t>(),
            camera_ids.data_ptr<int64_t>(),
            gaussian_ids.data_ptr<int64_t>(),
            ray_transforms.data_ptr<float>(),
            v_means2d.data_ptr<float>(),
            v_depths.data_ptr<float>(),
            v_ray_transforms.data_ptr<float>(),
            v_normals.data_ptr<float>(),
            sparse_grad,
            v_means.data_ptr<float>(),
            v_quats.data_ptr<float>(),
            v_scales.data_ptr<float>(),
            v_viewmats.has_value() ? v_viewmats.value().data_ptr<float>()
                                   : nullptr
        );
}

} // namespace gsplat
