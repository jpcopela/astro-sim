#include <glad/glad.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <glm/glm.hpp>



static const GLfloat p_vertex_buffer_data[] = {
     0.0f, 0.5f,  0.5f, 
     0.0f, 0.5f, -0.5f, 
     0.0f, -0.5f, -0.5f, 
     0.0f, -0.5f,  0.5f, 
     1.0f, 1.5f, 0.0f
};

static const struct Particle {
    glm::vec4 particleColor;

    glm::vec3 pos, velocity;
    float mass;
};



class Particles {
    public:
        Particles() {};
        
        static const unsigned int numParticles = 5;
        const char * texturePath;
        
        void initializeParticles(unsigned int numBodies);
        void display();
        void update();
        void destroy();

    private:  

        cudaGraphicsResource_t resource;

        unsigned int numBodies = 16;

        GLuint vertexArrayID;
        GLuint particles_vertex_buffer;
        

        unsigned int texture;

        void createParticleBuffers();
        
        void loadTexture();
};