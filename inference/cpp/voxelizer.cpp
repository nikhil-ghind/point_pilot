#include "voxelizer.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

Voxelizer::Voxelizer(
    float vx, float vy, float vz,
    const float range[6],
    int max_pts,
    int max_v
)
    : voxel_size_x(vx), voxel_size_y(vy), voxel_size_z(vz),
      max_points_per_voxel(max_pts), max_voxels(max_v)
{
    std::memcpy(pc_range, range, 6 * sizeof(float));

    // Compute grid dimensions: round((max - min) / voxel_size)
    grid_x = static_cast<int>(std::round((range[3] - range[0]) / vx));
    grid_y = static_cast<int>(std::round((range[4] - range[1]) / vy));
    grid_z = static_cast<int>(std::round((range[5] - range[2]) / vz));
}

std::vector<VoxelData> Voxelizer::voxelize(
    const Point* points,
    size_t n_points
)
{
    // Map from voxel key → list of point indices
    std::unordered_map<int64_t, std::vector<size_t>> voxel_map;
    voxel_map.reserve(static_cast<size_t>(max_voxels * 1.5));

    for (size_t i = 0; i < n_points; ++i) {
        const Point& p = points[i];

        // Filter out-of-range points
        if (p.x < pc_range[0] || p.x >= pc_range[3] ||
            p.y < pc_range[1] || p.y >= pc_range[4] ||
            p.z < pc_range[2] || p.z >= pc_range[5]) {
            continue;
        }

        // Compute voxel indices
        int ix = static_cast<int>((p.x - pc_range[0]) / voxel_size_x);
        int iy = static_cast<int>((p.y - pc_range[1]) / voxel_size_y);
        int iz = static_cast<int>((p.z - pc_range[2]) / voxel_size_z);

        // Clamp to grid bounds
        ix = std::max(0, std::min(ix, grid_x - 1));
        iy = std::max(0, std::min(iy, grid_y - 1));
        iz = std::max(0, std::min(iz, grid_z - 1));

        int64_t key = coordToKey(ix, iy, iz);
        voxel_map[key].push_back(i);
    }

    // Build output, respecting max_voxels limit
    std::vector<VoxelData> result;
    result.reserve(std::min(static_cast<size_t>(max_voxels), voxel_map.size()));

    int n_voxels = 0;
    for (auto& [key, pt_indices] : voxel_map) {
        if (n_voxels >= max_voxels) break;

        VoxelData vd;
        VoxelCoord raw_coord = keyToCoord(key);
        vd.coord = raw_coord;
        vd.num_points = static_cast<int32_t>(
            std::min(static_cast<size_t>(max_points_per_voxel), pt_indices.size())
        );

        for (int p_idx = 0; p_idx < vd.num_points; ++p_idx) {
            const Point& pt = points[pt_indices[p_idx]];
            std::array<float, 9> feat = {};
            feat[0] = pt.x;
            feat[1] = pt.y;
            feat[2] = pt.z;
            feat[3] = pt.intensity;
            // Features 4-8 will be filled in voxelizeWithFeatures
            vd.points.push_back(feat);
        }

        result.push_back(std::move(vd));
        ++n_voxels;
    }

    return result;
}

std::vector<VoxelData> Voxelizer::voxelizeWithFeatures(
    const Point* points,
    size_t n_points
)
{
    // First, do basic voxelization
    std::vector<VoxelData> voxels = voxelize(points, n_points);

    // Augment each voxel with pillar-center offsets
    for (auto& vd : voxels) {
        int n = vd.num_points;
        if (n == 0) continue;

        // Compute pillar center (mean of valid points xyz)
        float cx = 0.f, cy = 0.f, cz = 0.f;
        for (int i = 0; i < n; ++i) {
            cx += vd.points[i][0];
            cy += vd.points[i][1];
            cz += vd.points[i][2];
        }
        cx /= n;
        cy /= n;
        cz /= n;

        // Set augmented features for each point
        for (int i = 0; i < n; ++i) {
            vd.points[i][4] = cx;               // x_c
            vd.points[i][5] = cy;               // y_c
            vd.points[i][6] = cz;               // z_c
            vd.points[i][7] = vd.points[i][0] - cx;  // dx from center x
            vd.points[i][8] = vd.points[i][1] - cy;  // dy from center y
        }

        // Pad remaining slots with zeros (already zero-initialized)
        // No action needed — points vector contains exactly num_points entries
    }

    return voxels;
}

void Voxelizer::flattenToTensors(
    float* voxels_out,
    int32_t* coords_out,
    int32_t* num_points_out,
    const std::vector<VoxelData>& voxel_data
)
{
    const int P = max_points_per_voxel;
    const int C = 9;

    for (size_t v = 0; v < voxel_data.size(); ++v) {
        const VoxelData& vd = voxel_data[v];

        // Fill voxel features: (V, P, 9) row-major
        float* voxel_row = voxels_out + v * P * C;
        std::memset(voxel_row, 0, P * C * sizeof(float));

        for (int p = 0; p < vd.num_points; ++p) {
            float* feat_ptr = voxel_row + p * C;
            std::memcpy(feat_ptr, vd.points[p].data(), C * sizeof(float));
        }

        // Fill coordinates: [0 (batch), iz, iy, ix]
        int32_t* coord_ptr = coords_out + v * 4;
        coord_ptr[0] = 0;                   // batch index (always 0 for inference)
        coord_ptr[1] = vd.coord.iz;
        coord_ptr[2] = vd.coord.iy;
        coord_ptr[3] = vd.coord.ix;

        // Fill num_points
        num_points_out[v] = vd.num_points;
    }
}
