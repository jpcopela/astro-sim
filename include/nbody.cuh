#ifndef __NBODY_H__
#define __NBODY_H__

#include <cuda_runtime.h>

#define G 6.674e-11

typedef struct vec3d {
    float x;
    float y;
    float z;
};

void CHECK_CUDA(cudaError_t err);

void setDevice();

void createBuffers(GLuint& particles_vertex_buffer);

cudaError_t launchKernel(unsigned int numBodies, float3* positions);

__global__ void testKernel(float3* positions, double time);

#endif