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

cudaError_t launchKernel(unsigned int numBodies, float3* positions)
 {
    int numBlocks = 1;
    dim3 threadsPerBlock(numBodies, numBodies);

    double time = glfwGetTime();

    cudaError_t cudaStatus;
    testKernel<<<numBlocks, threadsPerBlock>>>(positions, time);
    cudaStatus = cudaGetLastError();

    return cudaStatus;
}

__global__ void testKernel(float3* positions, double time) {
    unsigned int id = threadIdx.x + threadIdx.y * blockDim.x;
 
    positions[id].x = threadIdx.x;
    positions[id].y = threadIdx.y;
    positions[id].z = 0.0;
}


