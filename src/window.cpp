#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <iostream>

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
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
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

    // Read our .obj file
    std::vector< glm::vec3 > vertices;
    std::vector< glm::vec2 > uvs;
    std::vector< glm::vec3 > normals;
    bool res = loadObject("../models/cube.obj", vertices, uvs, normals);

    GLuint vertexBuffer;
	glGenBuffers(1, &vertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, vertexBuffer);
	glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);

	GLuint uvBuffer;
	glGenBuffers(1, &uvBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, uvBuffer);
	glBufferData(GL_ARRAY_BUFFER, uvs.size() * sizeof(glm::vec2), &uvs[0], GL_STATIC_DRAW);

    // Create and compile our GLSL program from the shaders
    GLuint programID = LoadShaders( "../shaders/vertex.vert", "../shaders/fragment.frag" );
    glClearColor(0.07f, 0.13f, 0.17f, 1.0f);

    int major, minor;
    glGetIntegerv(GL_MAJOR_VERSION, &major);
    glGetIntegerv(GL_MINOR_VERSION, &minor);

    std::cout << major << "." << minor << std::endl;

    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        processInput(window);

        computeMatricesFromInputs(window, 3.14f, 0.0f, 45.0f, 0.01f, 0.005f);
        glm::mat4 projection = getProjectionMatrix();
        glm::mat4 view = getViewMatrix();
        glm::mat4 model = glm::mat4(1.0f);
        mvp = projection * view * model;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(programID);
        glUniformMatrix4fv(matrixID, 1, GL_FALSE, &mvp[0][0]);

        draw(vertexBuffer, uvBuffer, vertices);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    // Cleanup VBO and shader
	glDeleteBuffers(1, &vertexBuffer);
	glDeleteBuffers(1, &uvBuffer);
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
