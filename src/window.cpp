#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include <camera.hpp>
#include <shaders.hpp>
#include <objloader.hpp>
#include <obj_renderer.hpp>
#include <particle_renderer.hpp>

#include "nbody.cuh"

#include <glm/gtc/matrix_transform.hpp>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

//variables
const unsigned int width = 1920;
const unsigned int height = 1080;

GLuint matrixID;
glm::mat4 mvp;

int main() {
    
    
    setDevice();

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(width, height, "astro-sim", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }    

    

    //initialize class members
    Camera camera;
    Shaders shaders;
    Particles particles;
    
    //Compile vertex and fragment shaders  
    shaders.vertex_file_path = "../shaders/vertex.vert";
    shaders.fragment_file_path = "../shaders/fragment.frag";

    GLuint programID = shaders.loadShaders();

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    
    
    particles.texturePath = "../textures/star_texture.jpg";
    particles.initializeParticles(100);
    glEnable(GL_PROGRAM_POINT_SIZE); 

    // render loop
    while (!glfwWindowShouldClose(window))
    {   
        processInput(window);

        camera.updateCamera(window);
        mvp = camera.getMvp();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glUseProgram(programID);

        glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);

        particles.update();
        particles.display();
        
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //delete various things
    particles.destroy();
	glDeleteProgram(programID);

    //end program
	glfwTerminate();
    return 0;
}

//check if window should close
void processInput(GLFWwindow *window) {
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);     
}

//check if window has been resized
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}