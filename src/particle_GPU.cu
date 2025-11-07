#include"../include/CHECK.cuh"
#include"../include/particle_gpu.cuh"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include<cmath>
__global__ void resolveCollisions1(Particle* P, int count, int K, float restitution) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;

    Particle* p = &P[i];

    // 安全参数
    const float eps     = 1e-8f;   // 防除零
    const float v_slop  = 0.1f;    // 低速抖动阈值：|rvn|<v_slop 则临时 e=0
    const float pen_slop= 0.01f;   // 穿透容忍
    const float pos_pct = 0.2f;    // 位置校正比例 0.2~0.8
    const float j_max   = 1e5f;    // 冲量上限（保险丝，可按你场景调）

    float m1 = p->radius * p->radius;

    int kmax = min(K, NN_CAP);
    for (int k = 0; k < kmax; ++k) {
        Particle* q = p->partticles_distance_idx[k];
        if (!q) continue;

        int jidx = int(q - P);
        if (jidx <= i || jidx < 0 || jidx >= count) continue; // 只处理一次每对

        float dx = q->x - p->x;
        float dy = q->y - p->y;
        float R  = p->radius + q->radius;

        float dist2 = dx*dx + dy*dy;
        if (dist2 > R*R) continue;     // 未接触
        if (dist2 < eps) continue;     // 同心，跳过或先做微扰/分离

        float invDist = rsqrtf(dist2);
        float nx = dx * invDist;
        float ny = dy * invDist;

        float rvx = p->vx - q->vx;
        float rvy = p->vy - q->vy;
        float rvn = rvx*nx + rvy*ny;
        //if (rvn >= 0.0f) continue;     // 正在分离

        float m2 = q->radius * q->radius;

        // 低速碰撞不反弹，抑制颤动
        float e = (fabsf(rvn) < v_slop) ? 0.0f : fminf(fmaxf(restitution, 0.0f), 1.0f);

        // 冲量（等价于完全弹性向量公式）
        float invMassSum = 1.0f / (1.0f/m1 + 1.0f/m2);
        float jimp = -(1.0f + e) * rvn * invMassSum;
        jimp = fminf(jimp, j_max);     // 保险丝

        float dvx1 = (jimp * nx) / m1;
        float dvy1 = (jimp * ny) / m1;
        float dvx2 = -(jimp * nx) / m2;
        float dvy2 = -(jimp * ny) / m2;

        atomicAdd(&p->vx, dvx1);
        atomicAdd(&p->vy, dvy1);
        atomicAdd(&q->vx, dvx2);
        atomicAdd(&q->vy, dvy2);

        // --- Split impulse 位置校正（不改速度，避免加能） ---
        float dist = 1.0f / invDist;
        float penetration = R - dist;
        float corr = pos_pct * fmaxf(penetration - pen_slop, 0.0f);
        if (corr > 0.0f) {
            float invm1 = 1.0f / m1, invm2 = 1.0f / m2;
            float denom = invm1 + invm2;
            float cx1 = (corr * invm1 / denom) * nx;
            float cy1 = (corr * invm1 / denom) * ny;
            float cx2 = (corr * invm2 / denom) * nx;
            float cy2 = (corr * invm2 / denom) * ny;

            atomicAdd(&p->x, -cx1);
            atomicAdd(&p->y, -cy1);
            atomicAdd(&q->x,  cx2);
            atomicAdd(&q->y,  cy2);
        }
    }
}

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
        // if (dist2 <= pow(p->radius+q->radius,2.0f)) continue;
        float R = p->radius + q->radius;
        if (dist2 > R * R) continue;

        float dist = sqrtf(dist2);
         float nx = dx / dist;
         float ny = dy / dist;

        //  float rvn = (p->vx - q->vx) * nx + (p->vy - q->vy) * ny;
        //  if (rvn > 0.0f) continue;

        // float jimp = -(1.0f + restitution) * rvn * 0.5f;
        // float dvx = jimp * nx;
        // float dvy = jimp * ny;

        // atomicAdd(&p->vx,  dvx);
        // atomicAdd(&p->vy,  dvy);
        // atomicAdd(&q->vx, -dvx);
        // atomicAdd(&q->vy, -dvy);
        float A=2*pow(q->radius,2.0f)*((p->vx-q->vx)*(p->x-q->x)+(p->vy-q->vy)*(p->y-q->y))/((p->radius*p->radius+q->radius*q->radius)*((p->x-q->x)*(p->x-q->x)+(p->y-q->y)*(p->y-q->y)));
        float B=2*pow(p->radius,2.0f)*((q->vx-p->vx)*(q->x-p->x)+(q->vy-p->vy)*(q->y-p->y))/((q->radius*p->radius+p->radius*p->radius)*((q->x-p->x)*(q->x-p->x)+(q->y-p->y)*(q->y-p->y)));
        atomicAdd(&p->vx, -A*(p->x-q->x));
        atomicAdd(&p->vy, -A*(p->y-q->y));
        atomicAdd(&q->vx, B*(p->x-q->x));
        atomicAdd(&q->vy, B*(p->y-q->y));
        // float penetration = R - dist;
        // if (penetration > 0.0f) {
        //     float percent = 0.2f;
        //     float cx = percent * penetration * nx * 0.5f;
        //     float cy = percent * penetration * ny * 0.5f;
        //     atomicAdd(&p->x, -cx);
        //     atomicAdd(&p->y, -cy);
        //     atomicAdd(&q->x,  cx);
        //     atomicAdd(&q->y,  cy);
        // }
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
    resolveCollisions1<<<grid, block>>>(d_particles, count, 48, 0.2f);//实现粒子碰撞问题
    // 如需在 CPU 端读取结果，再决定是否同步
    CUDACheck(cudaDeviceSynchronize());

}

void ParticleGPU::syncToCPU (Particle* h_particles, int count) 
{
    cudaMemcpy(h_particles, d_particles, count * sizeof(Particle), cudaMemcpyDeviceToHost);
    cudaGetLastError();
}