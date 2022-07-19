#include <iostream>

#include <glad/glad.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include "test.cuh"

int Greetings::hello(int num) {
    std::cout << "Hello World!" << std::endl;
    return num;
}