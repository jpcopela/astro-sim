#version 330 core
layout(location = 0) in vec3 vertexPositionMS;
layout(location = 1) in vec3 offset;

uniform mat4 MVP;



void main() {
    gl_PointSize = 10.0;
    gl_Position = MVP * vec4(offset + vertexPositionMS, 1);
}
