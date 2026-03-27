#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <cstdio>
// OrbitCamera is included transitively via Renderer.h

// ---------------------------------------------------------------------------
// GLSL shaders
// ---------------------------------------------------------------------------

static const char *VERT_SRC = R"glsl(
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
uniform mat4 uMVP;
out vec3 vColor;
void main()
{
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
}
)glsl";

static const char *FRAG_SRC = R"glsl(
#version 330 core
in  vec3 vColor;
out vec4 FragColor;
void main()
{
    FragColor = vec4(vColor, 1.0);
}
)glsl";

// ---------------------------------------------------------------------------
// Geometry data  (interleaved: x y z  r g b)
// ---------------------------------------------------------------------------

// XYZ axis gizmo — 3 lines (6 vertices)
// X = red, Y = green, Z = blue
static const float AXIS_VERTS[] = {
    // X axis
    0.0f, 0.0f, 0.0f,   1.0f, 0.2f, 0.2f,
    1.0f, 0.0f, 0.0f,   1.0f, 0.2f, 0.2f,
    // Y axis
    0.0f, 0.0f, 0.0f,   0.2f, 1.0f, 0.2f,
    0.0f, 1.0f, 0.0f,   0.2f, 1.0f, 0.2f,
    // Z axis (cyan-blue for visibility on dark bg)
    0.0f, 0.0f, 0.0f,   0.2f, 0.6f, 1.0f,
    0.0f, 0.0f, 1.0f,   0.2f, 0.6f, 1.0f,
};

// Unit wireframe cube centred at origin — 12 edges (24 vertices for GL_LINES)
static const float g = 0.70f;  // grey value
static const float CUBE_VERTS[] = {
    // bottom face
    -0.5f,-0.5f,-0.5f, g,g,g,    0.5f,-0.5f,-0.5f, g,g,g,
     0.5f,-0.5f,-0.5f, g,g,g,    0.5f, 0.5f,-0.5f, g,g,g,
     0.5f, 0.5f,-0.5f, g,g,g,   -0.5f, 0.5f,-0.5f, g,g,g,
    -0.5f, 0.5f,-0.5f, g,g,g,   -0.5f,-0.5f,-0.5f, g,g,g,
    // top face
    -0.5f,-0.5f, 0.5f, g,g,g,    0.5f,-0.5f, 0.5f, g,g,g,
     0.5f,-0.5f, 0.5f, g,g,g,    0.5f, 0.5f, 0.5f, g,g,g,
     0.5f, 0.5f, 0.5f, g,g,g,   -0.5f, 0.5f, 0.5f, g,g,g,
    -0.5f, 0.5f, 0.5f, g,g,g,   -0.5f,-0.5f, 0.5f, g,g,g,
    // vertical edges
    -0.5f,-0.5f,-0.5f, g,g,g,   -0.5f,-0.5f, 0.5f, g,g,g,
     0.5f,-0.5f,-0.5f, g,g,g,    0.5f,-0.5f, 0.5f, g,g,g,
     0.5f, 0.5f,-0.5f, g,g,g,    0.5f, 0.5f, 0.5f, g,g,g,
    -0.5f, 0.5f,-0.5f, g,g,g,   -0.5f, 0.5f, 0.5f, g,g,g,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

GLuint Renderer::compile_shader(GLenum type, const char *src)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &src, nullptr);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Shader compile error: %s\n", log);
    }
    return shader;
}

GLuint Renderer::link_program(GLuint vert, GLuint frag)
{
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vert);
    glAttachShader(prog, frag);
    glLinkProgram(prog);

    GLint ok = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512];
        glGetProgramInfoLog(prog, sizeof(log), nullptr, log);
        std::fprintf(stderr, "Program link error: %s\n", log);
    }
    glDeleteShader(vert);
    glDeleteShader(frag);
    return prog;
}

static void upload_line_vao(GLuint &vao, GLuint &vbo,
                            const float *data, GLsizeiptr size)
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, size, data, GL_STATIC_DRAW);

    const GLsizei stride = 6 * sizeof(float);
    // location 0: position
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    // location 1: colour
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
}

// ---------------------------------------------------------------------------
// Public interface
// ---------------------------------------------------------------------------

void Renderer::init()
{
    GLuint vert = compile_shader(GL_VERTEX_SHADER,   VERT_SRC);
    GLuint frag = compile_shader(GL_FRAGMENT_SHADER, FRAG_SRC);
    m_program   = link_program(vert, frag);
    m_uMVP      = glGetUniformLocation(m_program, "uMVP");

    upload_line_vao(m_axis_vao, m_axis_vbo,
                    AXIS_VERTS, sizeof(AXIS_VERTS));
    upload_line_vao(m_cube_vao, m_cube_vbo,
                    CUBE_VERTS, sizeof(CUBE_VERTS));

    glClearColor(0.15f, 0.15f, 0.15f, 1.0f);
    glLineWidth(1.5f);
}

void Renderer::set_viewport(int width, int height)
{
    m_width  = width;
    m_height = height;
}

void Renderer::draw()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view  = camera.view_matrix();
    float aspect    = (m_height > 0) ? static_cast<float>(m_width) / m_height : 1.0f;
    glm::mat4 proj  = glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    glm::mat4 mvp   = proj * view * model;

    glUseProgram(m_program);
    glUniformMatrix4fv(m_uMVP, 1, GL_FALSE, glm::value_ptr(mvp));

    // Draw axis gizmo
    glBindVertexArray(m_axis_vao);
    glDrawArrays(GL_LINES, 0, 6);

    // Draw wireframe cube
    glBindVertexArray(m_cube_vao);
    glDrawArrays(GL_LINES, 0, 24);

    glBindVertexArray(0);
}

Renderer::~Renderer()
{
    glDeleteVertexArrays(1, &m_axis_vao);
    glDeleteBuffers(1, &m_axis_vbo);
    glDeleteVertexArrays(1, &m_cube_vao);
    glDeleteBuffers(1, &m_cube_vbo);
    glDeleteProgram(m_program);
}
