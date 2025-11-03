#include "particle_common.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#define CUDACheck(expr) do { \
    cudaError_t err = (expr); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA 错误: " << cudaGetErrorString(err) \
                  << " (行号: " << __LINE__ << ", 文件: " << __FILE__ << ")" << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while (0)
//__global__ void update_near_particles1(Particle *d_particle,int count,int idx);
// {
//     dim3 block(256);
//     dim3 grid((count + 256 - 1) / 256);
//     compute_distance<<<block,grid>>>(d_particle,d_particle->x,d_particle->y,idx,count);
//     // for (int i = 0; i < count - 1; i++) {
//     //     // 每轮循环将最大元素"浮"到末尾
//     //     for (int j = 0; j < count - i - 1; j++) {
//     //         if (d_particle->partticles_squre_distance[j] > d_particle->partticles_squre_distance[j + 1]) {
//     //             // 交换元素
//     //             int temp=d_particle->partticles_squre_distance[j];
//     //             Particle *temp_idx=d_particle->partticles_distance_idx[j];
//     //             d_particle->partticles_squre_distance[j]=d_particle->partticles_squre_distance[j+1];               
//     //             d_particle->partticles_distance_idx[j]=d_particle->partticles_distance_idx[j+1];
//     //             d_particle->partticles_squre_distance[j+1]=temp;
//     //             d_particle->partticles_distance_idx[j+1]=temp_idx;//把索引以及距离值交换
//     //         }
//     //     }
//     // }
    

//     //使用thrust库简化代码
//     thrust::device_ptr<float> dist (d_particle->partticles_squre_distance);
//     thrust::device_ptr<Particle> idx1(d_particle->partticles_distance_idx);
//     thrust::sort_by_key(dist, dist + 999, idx1);

// }

// GPU 核函数：更新粒子位置（力场计算逻辑）
//__global__ void updateParticles(Particle* d_particles, int count, float deltaTime, unsigned int winWidth, unsigned int winHeight); 
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;//线程的全局索引
//     if (idx >= count) return;

//     //Particle& p = d_particles[idx];
//     // 示例：简单运动+边界反弹（后续替换为力场逻辑)
//     extern __shared__ Particle share_particles[];
//     int tid = threadIdx.x;
//     if(tid>=blockDim.x)
//     {
//         return;
//     }

//     share_particles[tid]=d_particles[idx];//在每一个线程块内的共享内存内是储存了同一线程块下不同线程控制的粒子对象

//     share_particles[tid].x += share_particles[tid].vx * deltaTime;
//     share_particles[tid].y += share_particles[tid].vy * deltaTime;

//     if (share_particles[tid].x - share_particles[tid].radius < 0 || share_particles[tid].x + share_particles[tid].radius > winWidth) {
//         share_particles[tid].vx *= -0.8f;
//         share_particles[tid].x = max(share_particles[tid].radius, min(share_particles[tid].x, (float)winWidth - share_particles[tid].radius));
//     }
//     if (share_particles[tid].y - share_particles[tid].radius < 0 || share_particles[tid].y + share_particles[tid].radius > winHeight) {
//         share_particles[tid].vy *= -0.8f;
//         share_particles[tid].y = max(share_particles[tid].radius, min(share_particles[tid].y, (float)winHeight - share_particles[tid].radius));
//     }
//     dim3 block(256);
//     dim3 grid((count + 256 - 1) / 256);
//     update_near_particles1<<<grid,block>>>(&share_particles[tid],count,idx);
//     int i=0;
//     while(0)
//     {
//         if(i=idx) break;//排除和自己碰撞
//         //从最近的粒子筛选
//         if(share_particles[tid].partticles_squre_distance[i]<=2.0f)//粒子之间碰到了
//         {
//             //满足在直角坐标系下的动量定理
//             //两粒子的碰撞法向量n
//             float L=sqrtf(powf(share_particles[tid].x - share_particles[tid].partticles_distance_idx[i]->x,2.0f)+powf(share_particles[tid].y - share_particles[tid].partticles_distance_idx[i]->y,2.0f));//两粒子间的圆心距离
//             float n_x=(share_particles[tid].partticles_distance_idx[i]->x - share_particles[tid].x)/L;
//             float n_y=(share_particles[tid].partticles_distance_idx[i]->y - share_particles[tid].y)/L;
//             float s=(share_particles[tid].vx - share_particles[tid].partticles_distance_idx[i]->vx)*n_x+(share_particles[tid].vy - share_particles[tid].partticles_distance_idx[i]->vy)*n_y;
//             float v_x=share_particles[tid].vx-s*n_x;
//             float v_y=share_particles[tid].vy-s*n_y;
//             d_particles[idx].vx=v_x;
//             d_particles[idx].vy=v_y;
//             i++;
//         }
//         else
//         {
//             break;
//         }
//     }
// }
//__global__ void compute_distance(Particle* d_particles,int m_x,int m_y,int m_index,int count);//在初始所有粒子之后进行粒子距离运算
// {  
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;//线程的全局索引
//     if (idx >= count) return;
//     //Particle& p = d_particles[idx];//保存在本地内存，采用将全局内存加载到共享内存上
//     extern __shared__ Particle share_particles[];
//     int tid = threadIdx.x;
//     if(tid>=blockDim.x) return;
//     share_particles[tid]=d_particles[idx];//全局内存到共享内存上
//     if(m_index==idx)
//     {
//         return;//除去自己以外的任何粒子
//     }

//     else{
//         double square_distance=(m_x-share_particles[tid].x)*(m_x-share_particles[tid].x)+(m_y-share_particles[tid].y)*(m_y-share_particles[tid].y);
//         share_particles[tid].partticles_squre_distance[idx]=square_distance;//获取其余粒子对某个粒子的距离放在该列表
//         share_particles[tid].partticles_distance_idx[idx]=&d_particles[idx];//获取其余粒子的索引
//     }
// }
// __global__ void update_near_particles(Particle* d_particles, int count)
// {
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;//线程的全局索引
//     if (idx >= count) return;
//     //Particle& p = d_particles[idx];//保存在本地内存，采用将全局内存加载到共享内存上

//     extern __shared__ Particle share_particles[];
//     int tid = threadIdx.x;
//     if(tid<blockDim.x)
//     {
//         share_particles[tid]=d_particles[idx];
//     }

//     int computer_counts=count;
//     dim3 block(256);
//     dim3 grid((computer_counts + 256 - 1) / 256);
//     compute_distance<<<block,grid>>>(&share_particles[tid],share_particles[tid].x,share_particles[tid].y,idx,count);
//     for (int i = 0; i < count - 1; i++) {
//         // 每轮循环将最大元素"浮"到末尾
//         for (int j = 0; j < count - i - 1; j++) {
//             if (share_particles[tid].partticles_squre_distance[j] > share_particles[tid].partticles_squre_distance[j + 1]) {
//                 // 交换元素
//                 int temp=share_particles[tid].partticles_squre_distance[j];
//                 Particle *temp_idx=share_particles[tid].partticles_distance_idx[j];
//                 share_particles[tid].partticles_squre_distance[j]=share_particles[tid].partticles_squre_distance[j+1];            
//                 share_particles[tid].partticles_distance_idx[j]=share_particles[tid].partticles_distance_idx[j+1];
//                 share_particles[tid].partticles_squre_distance[j+1]=temp;
//                 share_particles[tid].partticles_distance_idx[j+1]=temp_idx;//把索引以及距离值交换
//             }
//         }
//     }
// }

