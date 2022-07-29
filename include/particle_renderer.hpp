#include <glad/glad.h>
#include <glm/glm.hpp>

static const GLfloat p_vertex_buffer_data[] = {
     0.0f, 0.5f,  0.5f, 
     0.0f, 0.5f, -0.5f, 
     0.0f, -0.5f, -0.5f, 
     0.0f, -0.5f,  0.5f, 
     1.0f, 1.5f, 0.0f
};

static const unsigned int p_indices[] = {  
    0, 1, 3,   
    1, 2, 3    
};   

static const GLfloat particles_positions[] = {
    0.6, 0.0, 0.0, 1.2, 0.0, 0.0, 1.8, 0.0, 0.0
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
        
        void initializeParticles(unsigned int numBodies, bool randomDist);
        void display();
        void update();
        void destroy();
        
        

    private:
        Particle particles[numParticles]; 
        GLuint particles_vertex_buffer;
        GLuint particles_position_buffer;

        GLuint particlesEBO;

        GLuint vertexArrayID;

        unsigned int texture;

        void createParticleBuffers();
        
        void loadTexture();
};