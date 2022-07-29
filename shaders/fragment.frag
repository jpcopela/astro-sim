#version 450 core

layout (location = 0) out vec4 color;

uniform sampler2D starImage;

void main() {
  //Don't ask me why I have to divide by 3 here, it has something
  //to do with the texture image I use and I don't know what it is
  color = texture(starImage, gl_PointCoord / 3.0) * vec4(1.0, 1.0, 1.0, 1.0);
}