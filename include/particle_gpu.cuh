#include"CHECK.cuh"
// 封装 CUDA 操作的类（供 CPU 调用）
class ParticleGPU {
private:
    Particle* d_particles = nullptr; // GPU 内存指针
public:
    // 分配 GPU 内存
    void init(int count, const Particle* h_particles) 
    {
        cudaMalloc(&d_particles, count * sizeof(Particle));
        cudaMemcpy(d_particles, h_particles, count * sizeof(Particle), cudaMemcpyHostToDevice);
    }

    // 调用 GPU 核函数更新粒子
    void update(int count, float deltaTime, unsigned int winW, unsigned int winH);
    // {
    //     if (count <= 0) return;  // 避免无效计算
        
    //     dim3 block(256);
    //     dim3 grid((count + block.x - 1) / block.x);  // 计算网格大小
        
    //     // 启动第一个内核：更新粒子力学状态
    //     updateParticles<<<grid, block>>>(d_particles, count, deltaTime, winW, winH);
    //     CUDACheck(cudaGetLastError());  // 检查内核启动错误
    //     CUDACheck(cudaDeviceSynchronize());  // 同步设备，确保内核执行完成（可选，根据需求）
        
    //     // 启动第二个内核：计算邻近粒子（修正块大小参数和函数名）
    //     // update_near_particles<<<grid, block>>>(d_particles, count);  // 第二个参数必须是 blockSize
    //     // CUDACheck(cudaGetLastError());  // 检查内核启动错误
    //     // CUDACheck(cudaDeviceSynchronize());  // 同步设备
    // }

        // 将 GPU 数据同步回 CPU
    void syncToCPU(Particle* h_particles, int count); 
    // {
    //     cudaMemcpy(h_particles, d_particles, count * sizeof(Particle), cudaMemcpyDeviceToHost);
    //     cudaGetLastError();
    // }

    // 释放 GPU 内存
    ~ParticleGPU() {
        if (d_particles) cudaFree(d_particles);
    }
};