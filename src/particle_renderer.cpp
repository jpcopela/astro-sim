#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>

#include <glad/glad.h>
#include "camera.hpp"

#include "obj_renderer.hpp"
#include "particle_renderer.hpp"

#include "nbody.cuh"


//must set public variable texturePath before calling!
void Particles::initializeParticles(unsigned int numBodies, bool randomDist) {
    if (texturePath != NULL) {
        createParticleBuffers();
        loadTexture();

        if (randomDist) {
            
        }
    }
    else {
        std::cout << "No texture path provided" << std::endl;
    }
}

void Particles::update() {

}

void Particles::createParticleBuffers() {
	glGenVertexArrays(1, &vertexArrayID);
    glBindVertexArray(vertexArrayID);

    glGenBuffers(1, &particles_vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, particles_vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, sizeof(p_vertex_buffer_data), p_vertex_buffer_data, GL_STREAM_DRAW);
    
    glEnableVertexAttribArray(0);
	glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    );

	
}

void Particles::destroy() {
	glDeleteVertexArrays(1, &vertexArrayID);
	glDeleteBuffers(1, &particles_vertex_buffer);
	//glDeleteBuffers(1, &particles_position_buffer);
}

void Particles::display() {
    glBindBuffer(GL_ARRAY_BUFFER, particles_vertex_buffer);
	glVertexAttribPointer(
        0,
        3,
        GL_FLOAT,
        GL_FALSE,
        0,
        (void*)0
    ); 

	glBindVertexArray(vertexArrayID);

    glDrawArrays(GL_POINTS, 0, 5);

	glBindVertexArray(0);
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