#include"../include/CHECK.cuh"
#include"../include/particle_gpu.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include<cmath>

__global__ void resolveCollisions(Particle* all_particles, int count, int K, float restitution)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle* p = &all_particles[i];
    int kmax = min(K, count - 1);  // 最多 count-1 个（不含自己）
    kmax = min(kmax, NN_CAP - 1);

    for (int k = 0; k < kmax; ++k) {
        Particle* q = p->partticles_distance_idx[k];
        if (!q) continue;  // 自己或无效项会是 nullptr

        float dx = q->x - p->x;
        float dy = q->y - p->y;
        float dist2 = dx * dx + dy * dy;
        if (dist2 <= 0.0f) continue;

        float R = p->radius + q->radius;
        if (dist2 > R * R) continue;

        float dist = sqrtf(dist2);
        float nx = dx / dist;
        float ny = dy / dist;

        float rvn = (p->vx - q->vx) * nx + (p->vy - q->vy) * ny;
        if (rvn > 0.0f) continue;

        float jimp = -(1.0f + restitution) * rvn * 0.5f;
        float dvx = jimp * nx;
        float dvy = jimp * ny;

        atomicAdd(&p->vx,  dvx);
        atomicAdd(&p->vy,  dvy);
        atomicAdd(&q->vx, -dvx);
        atomicAdd(&q->vy, -dvy);

        float penetration = R - dist;
        if (penetration > 0.0f) {
            float percent = 0.2f;
            float cx = percent * penetration * nx * 0.5f;
            float cy = percent * penetration * ny * 0.5f;
            atomicAdd(&p->x, -cx);
            atomicAdd(&p->y, -cy);
            atomicAdd(&q->x,  cx);
            atomicAdd(&q->y,  cy);
        }
    }
}

// __global__ void update_near_particles1(Particle *d_particle,int count,int idx,Particle*All_particles)
//     {
        
//         dim3 block(256);
//         dim3 grid((count + 256 - 1) / 256);
//         compute_distance<<<grid,block>>>(All_particles,d_particle->x,d_particle->y,idx,count);
//          for (int i = 0; i < count - 1; i++) {
//              // 每轮循环将最大元素"浮"到末尾
//              for (int j = 0; j < count - i - 1; j++)
//              {
//                  if (d_particle->partticles_squre_distance[j] > d_particle->partticles_squre_distance[j + 1]) {
//                      // 交换元素
//                      double temp=d_particle->partticles_squre_distance[j];
//                      Particle *temp_idx=d_particle->partticles_distance_idx[j];
//                      d_particle->partticles_squre_distance[j]=d_particle->partticles_squre_distance[j+1];               
//                      d_particle->partticles_distance_idx[j]=d_particle->partticles_distance_idx[j+1];
//                      d_particle->partticles_squre_distance[j+1]=temp;
//                      d_particle->partticles_distance_idx[j+1]=temp_idx;//把索引以及距离值交换
//                  }
//              }
//          }
        

//         // //使用thrust库简化代码
//         // thrust::device_ptr<double> dist (d_particle->partticles_squre_distance);
//         // thrust::device_ptr<Particle> idx1(d_particle->partticles_distance_idx);
//         // thrust::sort_by_key(dist, dist + 999, idx1);

//     }
__global__ void compute_distance(Particle* all_particles, int m_index, int count)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= count) return;
    if (j >= NN_CAP) return;               // ← 新增：防越界

    Particle* self  = &all_particles[m_index];
    Particle* other = &all_particles[j];

    if (j == m_index) {
        self->partticles_squre_distance[j] = INFINITY;
        self->partticles_distance_idx[j]   = nullptr;
        return;
    }

    double dx = double(self->x) - double(other->x);
    double dy = double(self->y) - double(other->y);
    double d2 = dx*dx + dy*dy;

    self->partticles_squre_distance[j] = d2;
    self->partticles_distance_idx[j]   = other;
}


// Host 包装：距离 → Thrust 排序（注意类型全用 double / Particle**）
void build_neighbors_for_idx(Particle* d_all, int count, int idx)
{
    int n = (count < NN_CAP) ? count : NN_CAP;  // ← 截断长度

    dim3 block(256), grid((count + block.x - 1) / block.x);
    compute_distance<<<grid, block>>>(d_all, idx, count);
    CUDACheck(cudaPeekAtLastError());

    Particle* self = d_all + idx;
    auto key_begin = thrust::device_pointer_cast(self->partticles_squre_distance);
    auto val_begin = thrust::device_pointer_cast(self->partticles_distance_idx);

    thrust::sort_by_key(key_begin, key_begin + n, val_begin);  // ← 用 n 而不是 count
}



__global__ void updateParticles(Particle* d_particles, int count,
                                float deltaTime, unsigned int winWidth, unsigned int winHeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    Particle p = d_particles[idx]; // 寄存器临时

    p.x += p.vx * deltaTime;
    p.y += p.vy * deltaTime;

    if (p.x - p.radius < 0.0f || p.x + p.radius > float(winWidth)) {
        p.vx *= -0.8f;
        // 用 fminf/fmaxf，类型一致
        p.x = fmaxf(p.radius, fminf(p.x, float(winWidth) - p.radius));
    }
    if (p.y - p.radius < 0.0f || p.y + p.radius > float(winHeight)) {
        p.vy *= -0.8f;
        p.y = fmaxf(p.radius, fminf(p.y, float(winHeight) - p.radius));
    }

    d_particles[idx] = p; // 写回全局
}


void ParticleGPU::update (int count, float deltaTime, unsigned int winW, unsigned int winH)
{
    if (count <= 0) return;

    dim3 block(256);
    dim3 grid((count + block.x - 1) / block.x);

    // 1) 先更新动力学
    updateParticles<<<grid, block>>>(d_particles, count, deltaTime, winW, winH);
    CUDACheck(cudaPeekAtLastError());

    // 2) 然后（按需）为若干粒子构建邻居
    //    - 如果 N 不大，也可以 for (int i=0;i<count;++i) 全量
    //    - 如果 N 很大，建议每帧只更新一部分（轮转）
    for (int i = 0; i < count; ++i) {
        build_neighbors_for_idx(d_particles, count, i);
    }
    resolveCollisions<<<grid, block>>>(d_particles, count, 16, 0.2f);//实现粒子碰撞问题
    // 如需在 CPU 端读取结果，再决定是否同步
    CUDACheck(cudaDeviceSynchronize());

}

void ParticleGPU::syncToCPU (Particle* h_particles, int count) 
{
    cudaMemcpy(h_particles, d_particles, count * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaGetLastError();
}