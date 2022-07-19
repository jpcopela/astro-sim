#include <glad/glad.h>

#include <vector>
#include <glm/glm.hpp>

class Renderer {
	public:
		Renderer() {};
		void draw(std::vector< glm::vec3 > vertices);
		void createBuffers(std::vector< glm::vec3 > vertices, std::vector<glm::vec2> uvs);
		void deleteBuffers();

	private:
		GLuint vertexBuffer;
		GLuint uvBuffer;
};