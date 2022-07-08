#include "objloader.hpp"

bool loadObject(const char * path,
    std::vector <glm::vec3> & outVertices,
    std::vector <glm::vec2> & outUvs,
    std::vector <glm::vec3> & outNormals);