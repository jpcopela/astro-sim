#include <iostream>
#include <glad/glad.h>

#include <cuda.h>
#include <cuda_gl_interop.h>

#include "particle_renderer.hpp"
#include "nbody.cuh"

#define DIM 512

cudaGraphicsResource *resource;

void setDevice() {
    cudaDeviceProp prop;
    int dev;

    memset(&prop, 0, sizeof(cudaDeviceProp));
    prop.major = 1;
    prop.minor = 0;
    cudaChooseDevice(&dev, &prop);

    cudaGLSetGLDevice(dev);
}
