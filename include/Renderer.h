#pragma once
#include <epoxy/gl.h>
#include "OrbitCamera.h"

class Renderer {
public:
    Renderer()  = default;
    ~Renderer();

    // Call once while the GL context is current (from on_realize)
    void init();

    // Update stored viewport dimensions (from on_resize)
    void set_viewport(int width, int height);

    // Issue draw calls (from on_render); GL context must be current
    void draw();

    // Exposed so GTK input handlers can manipulate the camera directly
    OrbitCamera camera;

private:
    // Shader helpers
    GLuint compile_shader(GLenum type, const char *src);
    GLuint link_program(GLuint vert, GLuint frag);

    GLuint m_program = 0;
    GLuint m_uMVP    = 0;   // uniform location

    // Axis gizmo (3 coloured lines)
    GLuint m_axis_vao = 0;
    GLuint m_axis_vbo = 0;

    // Wireframe unit cube (12 edges)
    GLuint m_cube_vao  = 0;
    GLuint m_cube_vbo  = 0;

    int m_width  = 1280;
    int m_height = 720;
};
