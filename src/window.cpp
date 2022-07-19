#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

#include "test.cuh"

#include <controls.hpp>
#include <shaders.hpp>
#include <objloader.hpp>
#include <rendering.hpp>


#include <glm/gtc/matrix_transform.hpp>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window);

//variables
const unsigned int width = 1920;
const unsigned int height = 1080;
GLuint matrixID;
glm::mat4 mvp;

int main() {
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);


#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
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
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }    


    // Enable depth test
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_CULL_FACE);
    // Accept fragment if it closer to the camera than the former one
    glDepthFunc(GL_LESS);

    GLuint vertexArrayID;
	glGenVertexArrays(1, &vertexArrayID);
	glBindVertexArray(vertexArrayID);

    //Read our .obj file
    Object object;
    object.filePath = "../models/cube.obj";
    object.loadObject();
    
    std::vector< glm::vec3 > vertices = object.outVertices;
    std::vector< glm::vec2 > uvs = object.outUvs;

    // Create and compile our GLSL program from the shaders
    Shaders shaders;
    shaders.vertex_file_path = "../shaders/vertex.vert";
    shaders.fragment_file_path = "../shaders/fragment.frag";

    GLuint programID = shaders.loadShaders();

    glClearColor(0.07f, 0.13f, 0.17f, 1.0f);

    int major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);

    std::cout << "OpenGL version: " << major << "." << minor << std::endl;

    Greetings greetings;
    greetings.hello(3);

    Camera camera; 
    Renderer renderer;
    renderer.createBuffers(vertices, uvs);

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);
        
        camera.updateCamera(window);
        mvp = camera.getMvp();
        

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(programID);
        glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);

    
        renderer.draw(vertices); // Draw the triangle !


        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    // Cleanup VBO and shader
    renderer.deleteBuffers();
	glDeleteProgram(programID);
	glDeleteVertexArrays(1, &vertexArrayID);

	// Close OpenGL window and terminate GLFW
	glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window) {
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}
