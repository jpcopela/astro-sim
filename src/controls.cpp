#include "controls.hpp"
#include "glm/glm.hpp"

void computeMatricesFromInputs(GLFWwindow* window, float horizontalAngle, float verticalAngleß, float fov, float speed, float mouseSpeed);

glm::mat4 getProjectionMatrix();

glm::mat4 getViewMatrix();