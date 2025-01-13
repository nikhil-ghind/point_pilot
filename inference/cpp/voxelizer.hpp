#pragma once

#include <vector>
#include <array>
#include <unordered_map>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <stdexcept>

/**
 * C++ port of the Python Voxelizer.
 *
 * Converts raw point clouds into pillar representation for PointPillars inference.
 * Mirrors the NumPy implementation in data/voxelization.py exactly.
 */

struct Point {
    float x, y, z, intensity;
};

struct VoxelCoord {
    int32_t iz, iy, ix;
};

struct VoxelData {
    std::vector<std::array<float, 9>> points;  // up to P points, 9 features each
    VoxelCoord coord;
    int32_t num_points;
};

class Voxelizer {
public:
    // Configuration
    float voxel_size_x, voxel_size_y, voxel_size_z;
    float pc_range[6];  // x_min, y_min, z_min, x_max, y_max, z_max
    int max_points_per_voxel;
    int max_voxels;

    // Grid dimensions
    int grid_x, grid_y, grid_z;

    /**
     * Constructor
     *
     * @param vx, vy, vz  voxel size in meters
     * @param range       point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
     * @param max_pts     max points per pillar
     * @param max_v       max number of non-empty voxels
     */
    Voxelizer(
        float vx, float vy, float vz,
        const float range[6],
        int max_pts = 100,
        int max_v = 12000
    );

    /**
     * Voxelize a point cloud.
     *
     * @param points      Input point cloud (N points)
     * @param n_points    Number of input points
     * @return            Vector of VoxelData structs
     */
    std::vector<VoxelData> voxelize(const Point* points, size_t n_points);

    /**
     * Voxelize and augment with pillar-center features.
     *
     * Produces 9-dimensional features per point:
     *   [x, y, z, intensity, x_c, y_c, z_c, dx_p, dy_p]
     *
     * @param points    Input points
     * @param n_points  Number of points
     * @return          Augmented voxel data
     */
    std::vector<VoxelData> voxelizeWithFeatures(
        const Point* points,
        size_t n_points
    );

    /**
     * Flatten voxels into tensors ready for ONNX Runtime inference.
     *
     * @param voxels        [out] (V, P, 9) float32 buffer
     * @param coords        [out] (V, 4) int32 buffer [0, iz, iy, ix]
     * @param num_points    [out] (V,) int32 buffer
     * @param voxel_data    Input voxel data from voxelizeWithFeatures
     */
    void flattenToTensors(
        float* voxels,           // [V * P * 9]
        int32_t* coords,         // [V * 4]
        int32_t* num_points,     // [V]
        const std::vector<VoxelData>& voxel_data
    );

    // Getters
    int getGridX() const { return grid_x; }
    int getGridY() const { return grid_y; }
    int getGridZ() const { return grid_z; }
    int getMaxVoxels() const { return max_voxels; }
    int getMaxPointsPerVoxel() const { return max_points_per_voxel; }

private:
    /**
     * Compute flat hash key for a voxel coordinate.
     * key = ix * (ny * nz) + iy * nz + iz
     */
    inline int64_t coordToKey(int ix, int iy, int iz) const {
        return static_cast<int64_t>(ix) * (grid_y * grid_z)
             + static_cast<int64_t>(iy) * grid_z
             + static_cast<int64_t>(iz);
    }

    /**
     * Recover grid coordinates from flat key.
     */
    inline VoxelCoord keyToCoord(int64_t key) const {
        VoxelCoord c;
        c.ix = static_cast<int32_t>(key / (grid_y * grid_z));
        c.iy = static_cast<int32_t>((key % (grid_y * grid_z)) / grid_z);
        c.iz = static_cast<int32_t>(key % grid_z);
        return c;
    }
};
