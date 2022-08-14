#include <iostream>
#include <stdio.h>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <time.h>

#include <cuda.h>
#include <cuda_gl_interop.h>
#include <curand.h>
#include <curand_kernel.h>

#include "particle_renderer.hpp"
#include "nbody.cuh"

#define DIM 512
#define G 6.674e-11


void CHECK_CUDA(cudaError_t err) {
    if (err != cudaSuccess) {
        std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
    }
}

void setDevice() {
    cudaDeviceProp prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice(&dev, &prop);

    cudaGLSetGLDevice(dev);

    std::cout << "Using device " << dev << std::endl;
}

void launchInitKernel(unsigned int numBodies, float3* positions) {
    dim3 numBlocks(12);
    dim3 threadsPerBlock(numBodies, numBodies);

    cudaError_t cudaStatus;
    plane<<<numBlocks, threadsPerBlock>>>(positions);
    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error launching initialization kernel: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

void launchGravityKernel(unsigned int numBodies, float3* positions, float3* velocities) {
    dim3 numBlocks(12);
    dim3 threadsPerBlock(numBodies, numBodies);

    cudaError_t cudaStatus;

    float mass = 100000.0; //kg
    float dt = 5.0; //seconds

    gravityKernel<<<numBlocks, threadsPerBlock>>>(positions, velocities, mass, dt);
    cudaStatus = cudaGetLastError();

    if (cudaStatus != cudaSuccess) {
        std::cerr << "Error launching initialization kernel: " << cudaGetErrorString(cudaStatus) << std::endl;
    }
}

__global__ void plane(float3* positions) {
    //unsigned int id = threadIdx.x + threadIdx.y * blockDim.x;
    int tid = threadIdx.x;
    int col_offset = blockDim.x * blockDim.y * blockIdx.x;
    int row_offset = gridDim.x * blockIdx.y * blockDim.x * blockDim.y + blockDim.x * threadIdx.y;
    int id = tid + col_offset + row_offset;
 
    positions[id].x = (threadIdx.x) * 50.0;
    positions[id].y = (blockIdx.x * 50.0) - 1000.0 ;
    positions[id].z = (threadIdx.y) * 50.0;
}

__global__ void gravityKernel(float3* positions, float3* d_velocity, float mass, float dt) {
    int tid = threadIdx.x;
    int col_offset = blockDim.x * blockDim.y * blockIdx.x;
    int row_offset = gridDim.x * blockIdx.y * blockDim.x * blockDim.y + blockDim.x * threadIdx.y;
    int i = tid + col_offset + row_offset;
    const float3 d0_i = positions[i];
    float3 a = {0, 0, 0};

    for (int j = 0; j < blockDim.x * blockDim.y * gridDim.x; j++) {
        if (j == i) continue;

        const float3 d0_j = positions[j];
        float3 r_ij;
        r_ij.x = d0_i.x - d0_j.x;
        r_ij.y = d0_i.y - d0_j.y;
        r_ij.z = d0_i.z - d0_j.z;

        float r_squared = (r_ij.x * r_ij.x) + (r_ij.y * r_ij.y) + (r_ij.z * r_ij.z);

        float F_coef = -G * mass / r_squared;

        a.x += F_coef * r_ij.x * rsqrt(r_squared);
        a.y += F_coef * r_ij.y * rsqrt(r_squared);
        a.z += F_coef * r_ij.z * rsqrt(r_squared);

        const float3 v0_i = d_velocity[i];
        d_velocity[i].x = v0_i.x + (a.x * dt);
        d_velocity[i].y = v0_i.y + (a.y * dt);
        d_velocity[i].z = v0_i.z + (a.z * dt);

        positions[i].x = (d0_i.x + v0_i.x * dt + a.x * dt * dt / 2.0);
        positions[i].y = (d0_i.y + v0_i.y * dt + a.y * dt * dt / 2.0);
        positions[i].z = (d0_i.z + v0_i.z * dt + a.z * dt * dt / 2.0);
    }   
}


