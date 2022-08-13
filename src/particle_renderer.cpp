#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

#include <glad/glad.h>
#include "camera.hpp"

#include "obj_renderer.hpp"
#include "particle_renderer.hpp"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>


#include "nbody.cuh"


//must set public variable texturePath before calling!
void Particles::initializeParticles() {
    if (texturePath != NULL) {
        createParticleBuffers();
        loadTexture();

        size_t size;
        CHECK_CUDA(cudaGraphicsMapResources(1, &resources[0], 0));
        CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&buffers[0], &size, resources[0]));

        launchInitKernel(numBodies, buffers[0]);

        CHECK_CUDA(cudaGraphicsUnmapResources(1, &resources[0], NULL));
    }
    else {
        std::cout << "No texture path provided" << std::endl;
    }
}

void Particles::update() {
    
    size_t size;
    
    CHECK_CUDA(cudaGraphicsMapResources(2, resources, 0));
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&buffers[0], &size, resources[0]));
    CHECK_CUDA(cudaGraphicsResourceGetMappedPointer((void**)&buffers[1], &size, resources[1]));

    launchGravityKernel(numBodies, buffers[0], buffers[1]);

    CHECK_CUDA(cudaGraphicsUnmapResources(2, resources, NULL));

    glBindBuffer(GL_ARRAY_BUFFER, particles_vertex_buffer);
}

void Particles::createParticleBuffers() {
	glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);

    glGenBuffers(1, &particles_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, numBodies * numBodies * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
	glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    );

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&resources[0], particles_vertex_buffer, cudaGraphicsMapFlagsNone));

    glGenBuffers(1, &velocity_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, velocity_buffer);
    glBufferData(GL_ARRAY_BUFFER, numBodies * numBodies * 3 * sizeof(float), 0, GL_DYNAMIC_DRAW);

     glEnableVertexAttribArray(1);
	glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    ); 

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    
    CHECK_CUDA(cudaGraphicsGLRegisterBuffer(&resources[1], velocity_buffer, cudaGraphicsMapFlagsNone));

    glBindVertexArray(0);
}

void Particles::destroy() {
	glDeleteVertexArrays(1, &vertexArrayID);
	glDeleteBuffers(1, &particles_vertex_buffer);
	//glDeleteBuffers(1, &particles_position_buffer);
}

void Particles::display() {
    glBindVertexArray(vertexArrayID);

    glBindBuffer(GL_ARRAY_BUFFER, particles_vertex_buffer);
	glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    ); 

     glBindBuffer(GL_ARRAY_BUFFER, velocity_buffer);
	glVertexAttribPointer(
        1,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    ); 
    
    glDrawArrays(GL_POINTS, 0, numBodies * numBodies);
}

//loads the texture used to display particles
void Particles::loadTexture() {   
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
	
	glEnable(GL_POINT_SPRITE);
	glEnable(GL_PROGRAM_POINT_SIZE);
	
	int width, height, nrChannels;
    unsigned char *data = stbi_load(texturePath, &width, &height, &nrChannels, 0);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	if (data) {
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);
	}
	else {
		std::cout << "Failed to load texture" << std::endl;
	}
	stbi_image_free(data);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_ONE);
}