#ifndef __NBODY_H__
#define __NBODY_H__

#include <cuda_runtime.h>

#define G 6.674e-11

typedef struct vec3d {
    float x;
    float y;
    float z;
};

cudaError_t setupBodies(vec3d* d, const float* m, const vec3d* v0, const vec3d* d0, const float dt, unsigned int size);

__global__ void gravityKernel(vec3d* v, vec3d* d, const float *m, const vec3d* v0, const vec3d* d0, float dt);

cudaError_t gravityIter(vec3d* d, vec3d* dev_v, vec3d* dev_d, const float* dev_m, const vec3d* dev_v0, const vec3d* dev_d0, float dt, unsigned int size);

cudaError_t updateBodies(vec3d* d, vec3d* dev_v, vec3d* dev_d, float* dev_m, vec3d* dev_v0, vec3d* dev_d0, float dt, unsigned int size, cudaError_t cudaStatus);

void setDevice();

#endif