/**
 * PointPillars C++ ONNX Runtime Inference
 *
 * Loads a trained PointPillars ONNX model and runs inference on a
 * KITTI-format binary point cloud file.
 *
 * Usage:
 *   ./pointpillars_infer pointpillars.onnx pointcloud.bin [score_threshold]
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. -DONNXRUNTIME_ROOT=/path/to/onnxruntime
 *   make -j4
 */

#include "voxelizer.hpp"

#include <onnxruntime/core/session/onnxruntime_cxx_api.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <chrono>


// ─── Utility ─────────────────────────────────────────────────────────────────

/**
 * Read a KITTI velodyne binary .bin file.
 * Returns a vector of Point structs (x, y, z, intensity).
 */
std::vector<Point> readPointCloud(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open point cloud file: " + path);
    }

    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t n_floats = file_size / sizeof(float);
    size_t n_points = n_floats / 4;

    std::vector<float> raw(n_floats);
    file.read(reinterpret_cast<char*>(raw.data()), file_size);
    file.close();

    std::vector<Point> points(n_points);
    for (size_t i = 0; i < n_points; ++i) {
        points[i].x         = raw[i * 4 + 0];
        points[i].y         = raw[i * 4 + 1];
        points[i].z         = raw[i * 4 + 2];
        points[i].intensity = raw[i * 4 + 3];
    }

    std::cout << "Loaded " << n_points << " points from " << path << "\n";
    return points;
}


// ─── Detection box ───────────────────────────────────────────────────────────

struct DetBox {
    float x, y, z, l, w, h, yaw;
    float score;
    int class_id;
};

const char* CLASS_NAMES[] = {"Car", "Pedestrian", "Cyclist"};


// ─── NMS ─────────────────────────────────────────────────────────────────────

/**
 * 2D axis-aligned IoU between two boxes (ignores yaw for speed).
 */
static float iou2d(const DetBox& a, const DetBox& b) {
    float ax1 = a.x - a.l / 2, ay1 = a.y - a.w / 2;
    float ax2 = a.x + a.l / 2, ay2 = a.y + a.w / 2;
    float bx1 = b.x - b.l / 2, by1 = b.y - b.w / 2;
    float bx2 = b.x + b.l / 2, by2 = b.y + b.w / 2;

    float ix1 = std::max(ax1, bx1);
    float iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2);
    float iy2 = std::min(ay2, by2);

    float iw = std::max(0.f, ix2 - ix1);
    float ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;

    float aa = (ax2 - ax1) * (ay2 - ay1);
    float ba = (bx2 - bx1) * (by2 - by1);
    float uni = aa + ba - inter;

    return inter / std::max(uni, 1e-6f);
}

/**
 * Greedy NMS over detection boxes.
 */
std::vector<DetBox> nms(
    std::vector<DetBox> boxes,
    float iou_threshold = 0.5f
) {
    // Sort by descending score
    std::sort(boxes.begin(), boxes.end(),
              [](const DetBox& a, const DetBox& b) { return a.score > b.score; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<DetBox> kept;

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        kept.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (boxes[i].class_id != boxes[j].class_id) continue;
            if (iou2d(boxes[i], boxes[j]) > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    return kept;
}


// ─── Decode predictions ──────────────────────────────────────────────────────

/**
 * Decode raw model outputs into DetBox structs.
 *
 * @param cls_data  (HWA * num_classes) float32 classification logits
 * @param box_data  (HWA * 7) float32 box encodings
 * @param H, W      feature map dimensions
 * @param A         anchors per location
 * @param num_classes
 * @param pc_range  [x_min, y_min, z_min, x_max, y_max, z_max]
 * @param anchor_l, anchor_w, anchor_h  anchor dimensions (Car class)
 * @param anchor_z  anchor center Z
 * @param score_threshold
 */
std::vector<DetBox> decodePredictions(
    const float* cls_data,
    const float* box_data,
    int H, int W, int A, int num_classes,
    const float pc_range[6],
    float anchor_l, float anchor_w, float anchor_h, float anchor_z,
    float score_threshold = 0.4f
) {
    int HWA = H * W * A;
    std::vector<DetBox> boxes;

    float x_stride = (pc_range[3] - pc_range[0]) / W;
    float y_stride = (pc_range[4] - pc_range[1]) / H;
    float x_offset = pc_range[0] + x_stride / 2.f;
    float y_offset = pc_range[1] + y_stride / 2.f;

    // Two rotations per location: 0 and pi/2
    float anchor_rotations[2] = {0.f, 1.5707963f};
    float d_a = std::sqrt(anchor_l * anchor_l + anchor_w * anchor_w);

    for (int hwa = 0; hwa < HWA; ++hwa) {
        // Decompose flat index: hwa = (iy * W + ix) * A + ia
        int ia  = hwa % A;
        int hw  = hwa / A;
        int ix  = hw % W;
        int iy  = hw / W;

        float ax = x_offset + ix * x_stride;
        float ay = y_offset + iy * y_stride;
        float ar = anchor_rotations[ia % 2];

        // Find best class
        int best_class = 0;
        float best_score_logit = cls_data[hwa * num_classes];
        for (int c = 1; c < num_classes; ++c) {
            if (cls_data[hwa * num_classes + c] > best_score_logit) {
                best_score_logit = cls_data[hwa * num_classes + c];
                best_class = c;
            }
        }

        // Sigmoid to get probability
        float score = 1.f / (1.f + std::exp(-best_score_logit));
        if (score < score_threshold) continue;

        // Decode box encoding
        const float* enc = box_data + hwa * 7;
        float x = enc[0] * d_a + ax;
        float y = enc[1] * d_a + ay;
        float z = enc[2] * anchor_h + anchor_z;
        float l = std::exp(enc[3]) * anchor_l;
        float w = std::exp(enc[4]) * anchor_w;
        float h = std::exp(enc[5]) * anchor_h;
        float r = enc[6] + ar;

        DetBox det;
        det.x = x; det.y = y; det.z = z;
        det.l = l; det.w = w; det.h = h;
        det.yaw = r;
        det.score = score;
        det.class_id = best_class;
        boxes.push_back(det);
    }

    return boxes;
}


// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <model.onnx> <pointcloud.bin> [score_threshold=0.4]\n";
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string pc_path    = argv[2];
    float score_threshold        = (argc >= 4) ? std::atof(argv[3]) : 0.4f;

    // ── 1. Load point cloud ──────────────────────────────────────────────────
    std::vector<Point> points;
    try {
        points = readPointCloud(pc_path);
    } catch (const std::exception& e) {
        std::cerr << "Error loading point cloud: " << e.what() << "\n";
        return 1;
    }

    // ── 2. Voxelization ──────────────────────────────────────────────────────
    float pc_range[6] = {0.f, -39.68f, -3.f, 69.12f, 39.68f, 1.f};
    Voxelizer voxelizer(0.16f, 0.16f, 4.0f, pc_range, 100, 12000);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<VoxelData> voxel_data = voxelizer.voxelizeWithFeatures(
        points.data(), points.size()
    );
    auto t1 = std::chrono::high_resolution_clock::now();

    int V = static_cast<int>(voxel_data.size());
    int P = voxelizer.getMaxPointsPerVoxel();
    int C = 9;

    std::cout << "Voxelization: " << V << " pillars in "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()
              << " ms\n";

    if (V == 0) {
        std::cout << "No valid voxels found. Exiting.\n";
        return 0;
    }

    // Allocate flat tensor buffers
    std::vector<float>   voxels_buf(V * P * C, 0.f);
    std::vector<int32_t> coords_buf(V * 4, 0);
    std::vector<int32_t> npts_buf(V, 0);

    voxelizer.flattenToTensors(
        voxels_buf.data(),
        coords_buf.data(),
        npts_buf.data(),
        voxel_data
    );

    // ── 3. ONNX Runtime inference ────────────────────────────────────────────
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "PointPillars");
    Ort::SessionOptions session_opts;
    session_opts.SetIntraOpNumThreads(4);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    std::cout << "Loading ONNX model: " << model_path << "\n";
    Ort::Session session(env, model_path.c_str(), session_opts);
    Ort::AllocatorWithDefaultOptions allocator;

    // Get input/output names
    size_t num_inputs  = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();
    std::cout << "Model inputs: " << num_inputs << ", outputs: " << num_outputs << "\n";

    // Input names
    std::vector<std::string> input_names_str(num_inputs);
    std::vector<const char*> input_names(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name = session.GetInputNameAllocated(i, allocator);
        input_names_str[i] = name.get();
        input_names[i] = input_names_str[i].c_str();
    }

    std::vector<std::string> output_names_str(num_outputs);
    std::vector<const char*> output_names(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name = session.GetOutputNameAllocated(i, allocator);
        output_names_str[i] = name.get();
        output_names[i] = output_names_str[i].c_str();
    }

    // Build input tensors
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // voxels: (V, P, C) float32
    std::vector<int64_t> voxels_shape = {(int64_t)V, (int64_t)P, (int64_t)C};
    auto voxels_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        voxels_buf.data(), voxels_buf.size(),
        voxels_shape.data(), voxels_shape.size()
    );

    // coords: (V, 4) int64
    std::vector<int64_t> coords_i64(coords_buf.begin(), coords_buf.end());
    std::vector<int64_t> coords_shape = {(int64_t)V, 4};
    auto coords_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        coords_i64.data(), coords_i64.size(),
        coords_shape.data(), coords_shape.size()
    );

    // num_points: (V,) int64
    std::vector<int64_t> npts_i64(npts_buf.begin(), npts_buf.end());
    std::vector<int64_t> npts_shape = {(int64_t)V};
    auto npts_tensor = Ort::Value::CreateTensor<int64_t>(
        memory_info,
        npts_i64.data(), npts_i64.size(),
        npts_shape.data(), npts_shape.size()
    );

    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(std::move(voxels_tensor));
    input_tensors.push_back(std::move(coords_tensor));
    input_tensors.push_back(std::move(npts_tensor));

    // Run inference
    std::cout << "Running inference...\n";
    auto t2 = std::chrono::high_resolution_clock::now();

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr},
        input_names.data(), input_tensors.data(), num_inputs,
        output_names.data(), num_outputs
    );

    auto t3 = std::chrono::high_resolution_clock::now();
    std::cout << "Inference time: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(t3 - t2).count()
              << " ms\n";

    // ── 4. Decode predictions ────────────────────────────────────────────────
    // cls_preds: (1, A*C, H, W)
    // box_preds: (1, A*7, H, W)
    auto cls_shape = output_tensors[0].GetTensorTypeAndShapeInfo().GetShape();
    auto box_shape = output_tensors[1].GetTensorTypeAndShapeInfo().GetShape();

    int H_out = static_cast<int>(cls_shape[2]);
    int W_out = static_cast<int>(cls_shape[3]);
    int A = 2;         // anchors per location
    int num_classes = 3;

    const float* cls_data = output_tensors[0].GetTensorData<float>();
    const float* box_data = output_tensors[1].GetTensorData<float>();

    // Transpose from (1, A*C, H, W) to (H, W, A, C) equivalent flat HWA*C
    int HWA = H_out * W_out * A;
    int C_cls = num_classes;
    int C_box = 7;

    // Reorder: cls_data layout is channel-first (1, A*C, H, W)
    // → We need (HWA, C) for our decode function
    std::vector<float> cls_hwac(HWA * C_cls);
    std::vector<float> box_hwac(HWA * C_box);

    for (int h = 0; h < H_out; ++h) {
        for (int w = 0; w < W_out; ++w) {
            for (int a = 0; a < A; ++a) {
                int hwa_idx = (h * W_out + w) * A + a;
                for (int c = 0; c < C_cls; ++c) {
                    // source: channel (a*C_cls + c), spatial (h*W+w)
                    int src = (a * C_cls + c) * H_out * W_out + h * W_out + w;
                    cls_hwac[hwa_idx * C_cls + c] = cls_data[src];
                }
                for (int c = 0; c < C_box; ++c) {
                    int src = (a * C_box + c) * H_out * W_out + h * W_out + w;
                    box_hwac[hwa_idx * C_box + c] = box_data[src];
                }
            }
        }
    }

    // Decode and NMS
    auto raw_boxes = decodePredictions(
        cls_hwac.data(), box_hwac.data(),
        H_out, W_out, A, num_classes,
        pc_range,
        3.9f, 1.6f, 1.56f, -1.78f,  // Car anchor defaults
        score_threshold
    );

    auto final_boxes = nms(raw_boxes, 0.5f);

    // ── 5. Print results ─────────────────────────────────────────────────────
    std::cout << "\n=== Detection Results (" << final_boxes.size() << " detections) ===\n";
    for (size_t i = 0; i < final_boxes.size(); ++i) {
        const DetBox& b = final_boxes[i];
        const char* cls_name = (b.class_id < 3) ? CLASS_NAMES[b.class_id] : "Unknown";
        std::printf(
            "[%zu] %-12s score=%.3f  x=%.2f y=%.2f z=%.2f  "
            "l=%.2f w=%.2f h=%.2f  yaw=%.2f\n",
            i, cls_name, b.score,
            b.x, b.y, b.z,
            b.l, b.w, b.h, b.yaw
        );
    }

    return 0;
}
