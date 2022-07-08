cmake_minimum_required (VERSION 2.8.3)
project("space-sim")

find_package(OpenGL REQUIRED)

include_directories(
    # my headers
    "./include/"
    )


set( GLFW_SHARED_LIB
    "${PROJECT_SOURCE_DIR}/libs/libglfw.3.dylib")

set ( ALL_LIBS 
    ${OpenGL}
    ${GLFW_SHARED_LIB}
    "-ldl"
    )

add_executable(sim 
    "src/glad.c"
    "src/window.cpp"
    )

target_link_libraries(sim ${ALL_LIBS})
install(TARGETS sim DESTINATION "${PROJECT_SOURCE_DIR}/bin/")
