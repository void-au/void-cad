#pragma once
#include <epoxy/gl.h>
#include "OrbitCamera.h"

class Renderer {
public:
    enum class ProjectionMode {
        Perspective,
        Isometric,
    };

    Renderer()  = default;
    ~Renderer();

    // Call once while the GL context is current.
    void init();

    // Update stored viewport dimensions.
    void set_viewport(int width, int height);

    // Issue draw calls; GL context must be current.
    void draw();

    void set_wireframe(bool enabled) { m_wireframe = enabled; }
    bool wireframe() const { return m_wireframe; }

    void set_projection_mode(ProjectionMode mode) { m_projection_mode = mode; }
    ProjectionMode projection_mode() const { return m_projection_mode; }
    bool is_isometric() const { return m_projection_mode == ProjectionMode::Isometric; }

    // Exposed so input handlers can manipulate the camera directly.
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

    // Solid unit cube (12 triangles)
    GLuint m_solid_cube_vao = 0;
    GLuint m_solid_cube_vbo = 0;

    int m_width  = 1280;
    int m_height = 720;
    bool m_wireframe = false;
    ProjectionMode m_projection_mode = ProjectionMode::Perspective;
};
