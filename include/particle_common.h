// 仅包含 CPU/GPU 通用的数据结构（无 SFML、无 CUDA 关键字）
#ifndef PARTICLE_COMMON_H
#define PARTICLE_COMMON_H
#ifndef NN_CAP//邻居数量=count-1
#define NN_CAP 999
#endif
#include<iostream>
#include<string>
using namespace std;
// 用 class 替代 struct，存储粒子数据
class Particle {
public:
    float x, y;      // 位置
    float vx, vy;
    float radius;    // 半径

    // 改这里：用索引 + 距离，避免指针排序的麻烦
       // 
    double   partticles_squre_distance[NN_CAP];  // 距离平方
    Particle* partticles_distance_idx[NN_CAP];   // 指向邻居粒子的指针

    Particle(float x_=0, float y_=0, float vx_=0, float vy_=0, float r_=1.0f)
        : x(x_), y(y_), vx(vx_), vy(vy_), radius(r_) {}
};

#endif // PARTICLE_COMMON_H