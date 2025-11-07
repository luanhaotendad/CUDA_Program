# CUDA Particle System with SFML 3.x

一个使用 **CUDA 13** 做并行计算、**SFML 3.x** 做可视化的粒子系统示例。  
核心特性：
- GPU 端更新粒子位置与速度（`updateParticles`）
- 为每个粒子计算最近邻（`compute_distance` + `thrust::sort_by_key`）
- 采用**半对称、带安全项**的碰撞响应（`resolveCollisions1`），使用原子操作避免写冲突
- 支持自定义每粒子参与碰撞的近邻个数 `K`、穿透校正比例、速度阈值等

---

## 目录结构（示例）

```
.
├── include/
│   ├── CHECK.cuh
│   ├── particle_gpu.cuh
│   └── （你的其他头文件，例如 particle_common.h 等）
├── src/
│   ├── main.cpp
│   └── particle_GPU.cu        # ← 你当前的 CUDA 实现放在这里
├── CMakeLists.txt
└── README.md
```

> 说明：代码里依赖 `NN_CAP`（近邻表容量）常量，请在你的头文件里定义（例如 `#define NN_CAP 1024` 或合适大小）。  
> 若 `count > NN_CAP`，构建近邻时会截断到 `NN_CAP` 之内。

---

## 依赖

- **CUDA 13.0+**（NVCC 13）  
- **CMake ≥ 3.21**（建议 ≥ 3.24）
- **SFML 3.x**（已安装到系统，如 `/usr/local/lib/cmake/SFML`）
- **GCC/G++ 13**（或兼容版本）

---

## 构建与运行

```bash
mkdir -p build && cd build

cmake ..   -DCMAKE_BUILD_TYPE=Release   -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc   -DCUDAToolkit_ROOT=/usr/local/cuda-13.0   -DCMAKE_CUDA_ARCHITECTURES=120   -DSFML_DIR=/usr/local/lib/cmake/SFML   -DSFML_STATIC_LIBRARIES=TRUE

cmake --build . -j"$(nproc)"

./MyApp
```

> 如果 CMake 报找不到 SFML：请确认 `grep` 或 `pkg-config` 显示的是 **3.0.0**，并正确设置了 `-DSFML_DIR=.../cmake/SFML`。  
> 如果 NVCC 编译期报架构错误：检查 `-DCMAKE_CUDA_ARCHITECTURES=120` 与你的驱动/卡是否匹配。

---

## 核心实现概览

### 1) 动力学更新（`updateParticles`）
- 每个线程负责一个粒子：积分位置、边界反弹（用 `fminf/fmaxf` 保证类型一致）
- 写回全局内存

### 2) 近邻构建（`compute_distance` + `build_neighbors_for_idx`）
- 对指定粒子 `idx`，并行计算它与所有粒子的距离（自己置 `INFINITY`）  
- 距离数组 `partticles_squre_distance[]` 与指针数组 `partticles_distance_idx[]` 同步维护  
- 用 `thrust::sort_by_key` 对 **前 n 项**排序（`n = min(count, NN_CAP)`），最近的在前

> 说明：这里的邻居表存的是 **`Particle*` 指针**（设备指针），因此**必须保证构建邻居和后续使用的数组基址一致**（本项目中统一使用 `d_particles`）。

### 3) 碰撞响应（`resolveCollisions1`）
- 只处理一次每对 (i, j)：要求 `jidx > i`，避免重复
- “质量”采用 `m = radius^2` 的近似
- 安全项：
  - `eps`：防除零
  - `v_slop`：低速抖动阈值
  - `pen_slop`：穿透容忍槽
  - `pos_pct`：穿透位置校正比例
  - `j_max`：冲量上限保险丝
- 使用 `atomicAdd` 对两粒子的 `vx/vy` 和 `x/y` 做对称修正，避免并发写入冲突

---

## 性能建议

- `K` 建议 16~48，越大越耗时  
- 近邻构建不必每帧全量，可轮转或隔帧更新  
- 可使用 `thrust::nth_element` 仅选前 K  
- 大规模优化可用**空间哈希/均匀网格**

---

## GPU 监控

```bash
nvtop
nvidia-smi
watch -n1 nvidia-smi
```

---

## 许可证

MIT / Apache-2.0 / 自定义
