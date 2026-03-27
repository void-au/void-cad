#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <cmath>
#include <string>
// OrbitCamera is included transitively via Renderer.h

// ---------------------------------------------------------------------------
// GLSL shaders — body is API-agnostic; the #version line is prepended at
// runtime so the same source works with both desktop GL 3.3+ and GLES 3.2+.
// ---------------------------------------------------------------------------

static const char *VERT_BODY = R"glsl(
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

static const char *FRAG_BODY = R"glsl(
precision mediump float;
in  vec3 vColor;
uniform vec4 uColorMul;
uniform vec4 uColorAdd;
out vec4 FragColor;
void main()
{
    FragColor = vec4(vColor, 1.0) * uColorMul + uColorAdd;
}
)glsl";

// Return the appropriate #version header for the current GL context.
static std::string glsl_version_header()
{
    const char *ver = reinterpret_cast<const char *>(glGetString(GL_VERSION));
    // GLES version strings start with "OpenGL ES"
    if (ver && std::string(ver).find("OpenGL ES") != std::string::npos)
        return "#version 320 es\n";
    return "#version 330 core\n";
}

// ---------------------------------------------------------------------------
// Geometry data  (interleaved: x y z  r g b)
// ---------------------------------------------------------------------------

// XYZ axis gizmo — 3 lines (6 vertices)
static const float AXIS_VERTS[] = {
    // X axis
    0.0f, 0.0f, 0.0f,   1.0f, 1.0f, 1.0f,
    1.0f, 0.0f, 0.0f,   1.0f, 1.0f, 1.0f,
    // Y axis
    0.0f, 0.0f, 0.0f,   0.85f, 0.85f, 0.85f,
    0.0f, 1.0f, 0.0f,   0.85f, 0.85f, 0.85f,
    // Z axis
    0.0f, 0.0f, 0.0f,   0.70f, 0.70f, 0.70f,
    0.0f, 0.0f, 1.0f,   0.70f, 0.70f, 0.70f,
};

// Unit wireframe cube centred at origin — 12 edges (24 vertices for GL_LINES)
static const float edge = 1.0f;
static const float face = 0.13333334f; // #222222
static const float CUBE_VERTS[] = {
    // bottom face
    -0.5f,-0.5f,-0.5f, edge,edge,edge,    0.5f,-0.5f,-0.5f, edge,edge,edge,
     0.5f,-0.5f,-0.5f, edge,edge,edge,    0.5f, 0.5f,-0.5f, edge,edge,edge,
     0.5f, 0.5f,-0.5f, edge,edge,edge,   -0.5f, 0.5f,-0.5f, edge,edge,edge,
    -0.5f, 0.5f,-0.5f, edge,edge,edge,   -0.5f,-0.5f,-0.5f, edge,edge,edge,
    // top face
    -0.5f,-0.5f, 0.5f, edge,edge,edge,    0.5f,-0.5f, 0.5f, edge,edge,edge,
     0.5f,-0.5f, 0.5f, edge,edge,edge,    0.5f, 0.5f, 0.5f, edge,edge,edge,
     0.5f, 0.5f, 0.5f, edge,edge,edge,   -0.5f, 0.5f, 0.5f, edge,edge,edge,
    -0.5f, 0.5f, 0.5f, edge,edge,edge,   -0.5f,-0.5f, 0.5f, edge,edge,edge,
    // vertical edges
    -0.5f,-0.5f,-0.5f, edge,edge,edge,   -0.5f,-0.5f, 0.5f, edge,edge,edge,
     0.5f,-0.5f,-0.5f, edge,edge,edge,    0.5f,-0.5f, 0.5f, edge,edge,edge,
     0.5f, 0.5f,-0.5f, edge,edge,edge,    0.5f, 0.5f, 0.5f, edge,edge,edge,
    -0.5f, 0.5f,-0.5f, edge,edge,edge,   -0.5f, 0.5f, 0.5f, edge,edge,edge,
};

static const float SOLID_CUBE_VERTS[] = {
    // front
    -0.5f,-0.5f, 0.5f, face,face,face,    0.5f,-0.5f, 0.5f, face,face,face,    0.5f, 0.5f, 0.5f, face,face,face,
    -0.5f,-0.5f, 0.5f, face,face,face,    0.5f, 0.5f, 0.5f, face,face,face,   -0.5f, 0.5f, 0.5f, face,face,face,
    // back
    -0.5f,-0.5f,-0.5f, face,face,face,   -0.5f, 0.5f,-0.5f, face,face,face,    0.5f, 0.5f,-0.5f, face,face,face,
    -0.5f,-0.5f,-0.5f, face,face,face,    0.5f, 0.5f,-0.5f, face,face,face,    0.5f,-0.5f,-0.5f, face,face,face,
    // left
    -0.5f,-0.5f,-0.5f, face,face,face,   -0.5f,-0.5f, 0.5f, face,face,face,   -0.5f, 0.5f, 0.5f, face,face,face,
    -0.5f,-0.5f,-0.5f, face,face,face,   -0.5f, 0.5f, 0.5f, face,face,face,   -0.5f, 0.5f,-0.5f, face,face,face,
    // right
     0.5f,-0.5f,-0.5f, face,face,face,    0.5f, 0.5f,-0.5f, face,face,face,    0.5f, 0.5f, 0.5f, face,face,face,
     0.5f,-0.5f,-0.5f, face,face,face,    0.5f, 0.5f, 0.5f, face,face,face,    0.5f,-0.5f, 0.5f, face,face,face,
    // top
    -0.5f, 0.5f,-0.5f, face,face,face,   -0.5f, 0.5f, 0.5f, face,face,face,    0.5f, 0.5f, 0.5f, face,face,face,
    -0.5f, 0.5f,-0.5f, face,face,face,    0.5f, 0.5f, 0.5f, face,face,face,    0.5f, 0.5f,-0.5f, face,face,face,
    // bottom
    -0.5f,-0.5f,-0.5f, face,face,face,    0.5f,-0.5f,-0.5f, face,face,face,    0.5f,-0.5f, 0.5f, face,face,face,
    -0.5f,-0.5f,-0.5f, face,face,face,    0.5f,-0.5f, 0.5f, face,face,face,   -0.5f,-0.5f, 0.5f, face,face,face,
};

static constexpr int kPivotCircleSegments = 48;
static constexpr int kSketchCircleSegments = 64;

static std::vector<float> make_pivot_marker_verts()
{
    std::vector<float> verts;
    verts.reserve(static_cast<std::size_t>(kPivotCircleSegments) * 12);

    for (int i = 0; i < kPivotCircleSegments; ++i) {
        const float a0 = (static_cast<float>(i) / static_cast<float>(kPivotCircleSegments)) * glm::two_pi<float>();
        const float a1 = (static_cast<float>(i + 1) / static_cast<float>(kPivotCircleSegments)) * glm::two_pi<float>();

        verts.push_back(std::cos(a0));
        verts.push_back(std::sin(a0));
        verts.push_back(0.0f);
        verts.push_back(1.0f);
        verts.push_back(1.0f);
        verts.push_back(1.0f);

        verts.push_back(std::cos(a1));
        verts.push_back(std::sin(a1));
        verts.push_back(0.0f);
        verts.push_back(1.0f);
        verts.push_back(1.0f);
        verts.push_back(1.0f);
    }

    return verts;
}

struct CubeEdge {
    glm::vec3 a;
    glm::vec3 b;
};

static const CubeEdge CUBE_EDGES[] = {
    {{-0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f,-0.5f}},
    {{ 0.5f,-0.5f,-0.5f}, { 0.5f, 0.5f,-0.5f}},
    {{ 0.5f, 0.5f,-0.5f}, {-0.5f, 0.5f,-0.5f}},
    {{-0.5f, 0.5f,-0.5f}, {-0.5f,-0.5f,-0.5f}},
    {{-0.5f,-0.5f, 0.5f}, { 0.5f,-0.5f, 0.5f}},
    {{ 0.5f,-0.5f, 0.5f}, { 0.5f, 0.5f, 0.5f}},
    {{ 0.5f, 0.5f, 0.5f}, {-0.5f, 0.5f, 0.5f}},
    {{-0.5f, 0.5f, 0.5f}, {-0.5f,-0.5f, 0.5f}},
    {{-0.5f,-0.5f,-0.5f}, {-0.5f,-0.5f, 0.5f}},
    {{ 0.5f,-0.5f,-0.5f}, { 0.5f,-0.5f, 0.5f}},
    {{ 0.5f, 0.5f,-0.5f}, { 0.5f, 0.5f, 0.5f}},
    {{-0.5f, 0.5f,-0.5f}, {-0.5f, 0.5f, 0.5f}},
};

static float depth_from_ndc(float z)
{
    return (z * 0.5f) + 0.5f;
}

static bool ray_box_intersection(const glm::vec3 &origin,
                                 const glm::vec3 &dir,
                                 float &hit_t,
                                 glm::vec3 &hit_normal)
{
    const glm::vec3 box_min(-0.5f, -0.5f, -0.5f);
    const glm::vec3 box_max( 0.5f,  0.5f,  0.5f);

    float t_min = -std::numeric_limits<float>::infinity();
    float t_max =  std::numeric_limits<float>::infinity();
    glm::vec3 normal(0.0f);

    for (int axis = 0; axis < 3; ++axis) {
        const float o = origin[axis];
        const float d = dir[axis];

        if (std::fabs(d) < 1e-6f) {
            if (o < box_min[axis] || o > box_max[axis]) {
                return false;
            }
            continue;
        }

        float t1 = (box_min[axis] - o) / d;
        float t2 = (box_max[axis] - o) / d;

        glm::vec3 near_normal(0.0f);
        near_normal[axis] = -1.0f;

        if (t1 > t2) {
            std::swap(t1, t2);
            near_normal[axis] = 1.0f;
        }

        if (t1 > t_min) {
            t_min = t1;
            normal = near_normal;
        }
        t_max = std::min(t_max, t2);

        if (t_min > t_max) {
            return false;
        }
    }

    float t_hit = t_min;
    if (t_hit < 0.0f) {
        t_hit = t_max;
    }
    if (t_hit < 0.0f) {
        return false;
    }

    hit_t = t_hit;
    hit_normal = normal;
    return true;
}

static int face_index_from_normal(const glm::vec3 &normal)
{
    if (normal.z > 0.5f)  return 0; // front
    if (normal.z < -0.5f) return 1; // back
    if (normal.x < -0.5f) return 2; // left
    if (normal.x > 0.5f)  return 3; // right
    if (normal.y > 0.5f)  return 4; // top
    if (normal.y < -0.5f) return 5; // bottom
    return -1;
}

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

static void init_dynamic_line_vao(GLuint &vao, GLuint &vbo)
{
    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_DYNAMIC_DRAW);

    const GLsizei stride = 6 * sizeof(float);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
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
    std::string hdr  = glsl_version_header();
    std::string vert_src = hdr + VERT_BODY;
    std::string frag_src = hdr + FRAG_BODY;

    GLuint vert = compile_shader(GL_VERTEX_SHADER,   vert_src.c_str());
    GLuint frag = compile_shader(GL_FRAGMENT_SHADER, frag_src.c_str());
    m_program   = link_program(vert, frag);
    m_uMVP      = glGetUniformLocation(m_program, "uMVP");
    m_uColorMul = glGetUniformLocation(m_program, "uColorMul");
    m_uColorAdd = glGetUniformLocation(m_program, "uColorAdd");

    upload_line_vao(m_axis_vao, m_axis_vbo,
                    AXIS_VERTS, sizeof(AXIS_VERTS));
    upload_line_vao(m_cube_vao, m_cube_vbo,
                    CUBE_VERTS, sizeof(CUBE_VERTS));
    upload_line_vao(m_solid_cube_vao, m_solid_cube_vbo,
                    SOLID_CUBE_VERTS, sizeof(SOLID_CUBE_VERTS));
    const std::vector<float> pivot_marker_verts = make_pivot_marker_verts();
    upload_line_vao(m_pivot_marker_vao, m_pivot_marker_vbo,
                    pivot_marker_verts.data(),
                    static_cast<GLsizeiptr>(pivot_marker_verts.size() * sizeof(float)));
    init_dynamic_line_vao(m_overlay_vao, m_overlay_vbo);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
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
    glm::mat4 proj  = projection_matrix();
    glm::mat4 mvp   = proj * view * model;
    const float scene_dim = m_sketch_active ? 0.45f : 1.0f;

    glUseProgram(m_program);
    glUniformMatrix4fv(m_uMVP, 1, GL_FALSE, glm::value_ptr(mvp));
    glUniform4f(m_uColorMul, scene_dim, scene_dim, scene_dim, 1.0f);
    glUniform4f(m_uColorAdd, 0.0f, 0.0f, 0.0f, 0.0f);

    if (m_wireframe) {
        glBindVertexArray(m_cube_vao);
        glDrawArrays(GL_LINES, 0, 24);
    } else {
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(1.0f, 1.0f);
        glBindVertexArray(m_solid_cube_vao);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glDisable(GL_POLYGON_OFFSET_FILL);

        glBindVertexArray(m_cube_vao);
        glDrawArrays(GL_LINES, 0, 24);
    }

    auto draw_face_overlay = [&](const PrimitiveHit &hit, bool selected) {
        if (!hit.valid() || hit.type != PrimitiveType::Face) {
            return;
        }

        const glm::vec4 color = selected
            ? glm::vec4(1.0f, 1.0f, 1.0f, 0.62f)
            : glm::vec4(0.82f, 0.82f, 0.82f, 0.38f);

        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glEnable(GL_POLYGON_OFFSET_FILL);
        glPolygonOffset(-1.0f, -1.0f);

        glUniform4f(m_uColorMul, 0.0f, 0.0f, 0.0f, 0.0f);
        glUniform4f(m_uColorAdd, color.r, color.g, color.b, color.a);
        glBindVertexArray(m_solid_cube_vao);
        glDrawArrays(GL_TRIANGLES, hit.index * 6, 6);

        glDisable(GL_POLYGON_OFFSET_FILL);
        glDisable(GL_BLEND);
    };

    auto draw_edge_overlay = [&](const PrimitiveHit &hit, bool selected) {
        if (!hit.valid() || hit.type != PrimitiveType::Edge) {
            return;
        }

        const glm::vec4 color = selected
            ? glm::vec4(1.0f, 1.0f, 1.0f, 1.0f)
            : glm::vec4(0.82f, 0.82f, 0.82f, 1.0f);

        glUniform4f(m_uColorMul, 0.0f, 0.0f, 0.0f, 0.0f);
        glUniform4f(m_uColorAdd, color.r, color.g, color.b, color.a);
        glLineWidth(selected ? 4.0f : 3.0f);
        glBindVertexArray(m_cube_vao);
        glDrawArrays(GL_LINES, hit.index * 2, 2);
    };

    if (!m_sketch_active) {
        draw_face_overlay(m_selected_hit, true);
        if (!(m_selected_hit.valid() && m_hovered_hit.valid() &&
              m_selected_hit.type == m_hovered_hit.type &&
              m_selected_hit.index == m_hovered_hit.index)) {
            draw_face_overlay(m_hovered_hit, false);
        }

        draw_edge_overlay(m_selected_hit, true);
        if (!(m_selected_hit.valid() && m_hovered_hit.valid() &&
              m_selected_hit.type == m_hovered_hit.type &&
              m_selected_hit.index == m_hovered_hit.index)) {
            draw_edge_overlay(m_hovered_hit, false);
        }
    }

    glUniform4f(m_uColorMul, 1.0f, 1.0f, 1.0f, 1.0f);
    glUniform4f(m_uColorAdd, 0.0f, 0.0f, 0.0f, 0.0f);
    glLineWidth(1.5f);

    if (m_sketch_active) {
        std::vector<float> grid_verts;
        std::vector<float> sketch_verts;
        grid_verts.reserve(8192);
        sketch_verts.reserve(16384);
        const glm::vec3 grid_plane_offset = m_sketch_plane.normal * 0.0015f;
        const glm::vec3 sketch_plane_offset = m_sketch_plane.normal * 0.0045f;

        auto push_line = [](std::vector<float> &verts,
                            const glm::vec3 &a,
                            const glm::vec3 &b,
                            const glm::vec3 &color) {
            verts.push_back(a.x); verts.push_back(a.y); verts.push_back(a.z);
            verts.push_back(color.r); verts.push_back(color.g); verts.push_back(color.b);
            verts.push_back(b.x); verts.push_back(b.y); verts.push_back(b.z);
            verts.push_back(color.r); verts.push_back(color.g); verts.push_back(color.b);
        };

        auto push_circle = [&](const glm::vec2 &center, float radius, const glm::vec3 &color) {
            if (radius <= 1e-4f) {
                return;
            }
            for (int i = 0; i < kSketchCircleSegments; ++i) {
                const float a0 = (static_cast<float>(i) / static_cast<float>(kSketchCircleSegments)) * glm::two_pi<float>();
                const float a1 = (static_cast<float>(i + 1) / static_cast<float>(kSketchCircleSegments)) * glm::two_pi<float>();
                const glm::vec2 p0 = center + glm::vec2(std::cos(a0), std::sin(a0)) * radius;
                const glm::vec2 p1 = center + glm::vec2(std::cos(a1), std::sin(a1)) * radius;
                push_line(sketch_verts,
                          sketch_world_from_local(p0) + sketch_plane_offset,
                          sketch_world_from_local(p1) + sketch_plane_offset,
                          color);
            }
        };

        auto push_constraint_glyph = [&](const SketchLine &line) {
            if (line.constraint == SketchConstraintType::None) {
                return;
            }

            const glm::vec2 mid = (line.start + line.end) * 0.5f;
            const float s = 0.06f;
            const glm::vec3 glyph_color(0.80f, 0.80f, 0.80f);
            if (line.constraint == SketchConstraintType::Horizontal) {
                push_line(sketch_verts,
                          sketch_world_from_local({mid.x - s, mid.y}) + sketch_plane_offset,
                          sketch_world_from_local({mid.x + s, mid.y}) + sketch_plane_offset,
                          glyph_color);
                push_line(sketch_verts,
                          sketch_world_from_local({mid.x - s, mid.y - (s * 0.35f)}) + sketch_plane_offset,
                          sketch_world_from_local({mid.x - s, mid.y + (s * 0.35f)}) + sketch_plane_offset,
                          glyph_color);
                push_line(sketch_verts,
                          sketch_world_from_local({mid.x + s, mid.y - (s * 0.35f)}) + sketch_plane_offset,
                          sketch_world_from_local({mid.x + s, mid.y + (s * 0.35f)}) + sketch_plane_offset,
                          glyph_color);
            } else if (line.constraint == SketchConstraintType::Vertical) {
                push_line(sketch_verts,
                          sketch_world_from_local({mid.x, mid.y - s}) + sketch_plane_offset,
                          sketch_world_from_local({mid.x, mid.y + s}) + sketch_plane_offset,
                          glyph_color);
                push_line(sketch_verts,
                          sketch_world_from_local({mid.x - (s * 0.35f), mid.y - s}) + sketch_plane_offset,
                          sketch_world_from_local({mid.x + (s * 0.35f), mid.y - s}) + sketch_plane_offset,
                          glyph_color);
                push_line(sketch_verts,
                          sketch_world_from_local({mid.x - (s * 0.35f), mid.y + s}) + sketch_plane_offset,
                          sketch_world_from_local({mid.x + (s * 0.35f), mid.y + s}) + sketch_plane_offset,
                          glyph_color);
            }
        };

        const float extent = 3.0f;
        const float spacing = m_sketch_grid_spacing;
        const glm::vec3 grid_color(0.30f, 0.30f, 0.30f);
        const glm::vec3 axis_color(0.65f, 0.65f, 0.65f);
        const int steps = static_cast<int>(extent / spacing);

        for (int i = -steps; i <= steps; ++i) {
            const float u = static_cast<float>(i) * spacing;
            const bool axis = (i == 0);
            const glm::vec3 color = axis ? axis_color : grid_color;

            push_line(grid_verts,
                      sketch_world_from_local({u, -extent}) + grid_plane_offset,
                      sketch_world_from_local({u,  extent}) + grid_plane_offset,
                      color);
            push_line(grid_verts,
                      sketch_world_from_local({-extent, u}) + grid_plane_offset,
                      sketch_world_from_local({ extent, u}) + grid_plane_offset,
                      color);
        }

        for (const SketchLine &line : m_sketch_lines) {
            push_line(sketch_verts,
                      sketch_world_from_local(line.start) + sketch_plane_offset,
                      sketch_world_from_local(line.end) + sketch_plane_offset,
                      glm::vec3(1.0f, 1.0f, 1.0f));
            push_constraint_glyph(line);
        }

        for (const SketchRectangle &rect : m_sketch_rectangles) {
            const glm::vec2 min_corner = rect.min_corner;
            const glm::vec2 max_corner = rect.max_corner;
            const glm::vec2 a(min_corner.x, min_corner.y);
            const glm::vec2 b(max_corner.x, min_corner.y);
            const glm::vec2 c(max_corner.x, max_corner.y);
            const glm::vec2 d(min_corner.x, max_corner.y);
            const glm::vec3 color(1.0f, 1.0f, 1.0f);
            push_line(sketch_verts, sketch_world_from_local(a) + sketch_plane_offset, sketch_world_from_local(b) + sketch_plane_offset, color);
            push_line(sketch_verts, sketch_world_from_local(b) + sketch_plane_offset, sketch_world_from_local(c) + sketch_plane_offset, color);
            push_line(sketch_verts, sketch_world_from_local(c) + sketch_plane_offset, sketch_world_from_local(d) + sketch_plane_offset, color);
            push_line(sketch_verts, sketch_world_from_local(d) + sketch_plane_offset, sketch_world_from_local(a) + sketch_plane_offset, color);
        }

        for (const SketchCircle &circle : m_sketch_circles) {
            push_circle(circle.center, circle.radius, glm::vec3(1.0f, 1.0f, 1.0f));
            push_line(sketch_verts,
                      sketch_world_from_local(circle.center) + sketch_plane_offset,
                      sketch_world_from_local(circle.center + glm::vec2(circle.radius, 0.0f)) + sketch_plane_offset,
                      glm::vec3(0.72f, 0.72f, 0.72f));
        }

        if (m_sketch_entity_in_progress && m_sketch_cursor_valid) {
            if (m_sketch_tool == SketchTool::Line) {
                push_line(sketch_verts,
                          sketch_world_from_local(m_sketch_anchor) + sketch_plane_offset,
                          m_sketch_preview_world + sketch_plane_offset,
                          glm::vec3(0.75f, 0.75f, 0.75f));
                if (m_sketch_preview_constraint != SketchConstraintType::None) {
                    push_constraint_glyph({m_sketch_anchor, m_sketch_preview_local, m_sketch_preview_constraint});
                }
            } else if (m_sketch_tool == SketchTool::Rectangle) {
                const glm::vec2 min_corner(std::min(m_sketch_anchor.x, m_sketch_preview_local.x),
                                           std::min(m_sketch_anchor.y, m_sketch_preview_local.y));
                const glm::vec2 max_corner(std::max(m_sketch_anchor.x, m_sketch_preview_local.x),
                                           std::max(m_sketch_anchor.y, m_sketch_preview_local.y));
                const glm::vec2 a(min_corner.x, min_corner.y);
                const glm::vec2 b(max_corner.x, min_corner.y);
                const glm::vec2 c(max_corner.x, max_corner.y);
                const glm::vec2 d(min_corner.x, max_corner.y);
                const glm::vec3 color(0.75f, 0.75f, 0.75f);
                push_line(sketch_verts, sketch_world_from_local(a) + sketch_plane_offset, sketch_world_from_local(b) + sketch_plane_offset, color);
                push_line(sketch_verts, sketch_world_from_local(b) + sketch_plane_offset, sketch_world_from_local(c) + sketch_plane_offset, color);
                push_line(sketch_verts, sketch_world_from_local(c) + sketch_plane_offset, sketch_world_from_local(d) + sketch_plane_offset, color);
                push_line(sketch_verts, sketch_world_from_local(d) + sketch_plane_offset, sketch_world_from_local(a) + sketch_plane_offset, color);
            } else if (m_sketch_tool == SketchTool::Circle) {
                const float radius = glm::length(m_sketch_preview_local - m_sketch_anchor);
                push_circle(m_sketch_anchor, radius, glm::vec3(0.75f, 0.75f, 0.75f));
                push_line(sketch_verts,
                          sketch_world_from_local(m_sketch_anchor) + sketch_plane_offset,
                          m_sketch_preview_world + sketch_plane_offset,
                          glm::vec3(0.75f, 0.75f, 0.75f));
            }
        }

        if (m_sketch_cursor_valid) {
            const float cursor_size = std::clamp(camera.distance() * 0.018f, 0.025f, 0.085f);
            const glm::vec2 c = m_sketch_cursor_local;
            const glm::vec3 cursor_color = m_sketch_cursor_snapped
                ? glm::vec3(1.0f, 1.0f, 1.0f)
                : glm::vec3(0.78f, 0.78f, 0.78f);

            push_line(sketch_verts,
                      sketch_world_from_local({c.x - cursor_size, c.y}) + sketch_plane_offset,
                      sketch_world_from_local({c.x + cursor_size, c.y}) + sketch_plane_offset,
                      cursor_color);
            push_line(sketch_verts,
                      sketch_world_from_local({c.x, c.y - cursor_size}) + sketch_plane_offset,
                      sketch_world_from_local({c.x, c.y + cursor_size}) + sketch_plane_offset,
                      cursor_color);
        }

        draw_overlay_lines(grid_verts, 1.0f);
        draw_overlay_lines(sketch_verts, 2.0f);
    }

    if (pivot_marker_visible()) {
        const float marker_scale = std::clamp(camera.distance() * 0.0125f, 0.02f, 0.10f);
        const glm::vec3 normal = glm::normalize(m_pivot_marker_normal);
        const glm::vec3 up_hint = (std::fabs(normal.y) > 0.9f)
            ? glm::vec3(1.0f, 0.0f, 0.0f)
            : glm::vec3(0.0f, 1.0f, 0.0f);
        const glm::vec3 tangent = glm::normalize(glm::cross(up_hint, normal));
        const glm::vec3 bitangent = glm::normalize(glm::cross(normal, tangent));

        glm::mat4 orientation(1.0f);
        orientation[0] = glm::vec4(tangent, 0.0f);
        orientation[1] = glm::vec4(bitangent, 0.0f);
        orientation[2] = glm::vec4(normal, 0.0f);

        const glm::mat4 marker_model = glm::translate(glm::mat4(1.0f), m_pivot_marker_pos)
                                     * orientation
                                     * glm::scale(glm::mat4(1.0f), glm::vec3(marker_scale));
        const glm::mat4 marker_mvp = proj * view * marker_model;

        glDisable(GL_DEPTH_TEST);
        glUniformMatrix4fv(m_uMVP, 1, GL_FALSE, glm::value_ptr(marker_mvp));
        glUniform4f(m_uColorMul, 0.0f, 0.0f, 0.0f, 0.0f);
        glUniform4f(m_uColorAdd, 1.0f, 1.0f, 1.0f, 0.95f);
        glLineWidth(3.0f);
        glBindVertexArray(m_pivot_marker_vao);
        glDrawArrays(GL_LINES, 0, kPivotCircleSegments * 2);
        glEnable(GL_DEPTH_TEST);

        glUniformMatrix4fv(m_uMVP, 1, GL_FALSE, glm::value_ptr(mvp));
        glUniform4f(m_uColorMul, 1.0f, 1.0f, 1.0f, 1.0f);
        glUniform4f(m_uColorAdd, 0.0f, 0.0f, 0.0f, 0.0f);
        glLineWidth(1.5f);
    }

    // Draw axis gizmo after the cube so it stays visible.
    glBindVertexArray(m_axis_vao);
    glDrawArrays(GL_LINES, 0, 6);

    glBindVertexArray(0);
}

void Renderer::draw_overlay_lines(const std::vector<float> &verts, float line_width) const
{
    if (verts.empty()) {
        return;
    }

    glUniform4f(m_uColorMul, 1.0f, 1.0f, 1.0f, 1.0f);
    glUniform4f(m_uColorAdd, 0.0f, 0.0f, 0.0f, 0.0f);
    glBindVertexArray(m_overlay_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_overlay_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
                 verts.data(),
                 GL_DYNAMIC_DRAW);
    glLineWidth(line_width);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(verts.size() / 6));
    glBindVertexArray(0);
    glLineWidth(1.5f);
}

glm::mat4 Renderer::projection_matrix() const
{
    const float aspect = (m_height > 0) ? static_cast<float>(m_width) / m_height : 1.0f;
    if (m_projection_mode == ProjectionMode::Perspective) {
        return glm::perspective(glm::radians(45.0f), aspect, 0.1f, 100.0f);
    }

    const float ortho_height = std::max(0.5f, camera.distance() * 0.60f);
    const float ortho_width = ortho_height * aspect;
    return glm::ortho(-ortho_width, ortho_width,
                      -ortho_height, ortho_height,
                      0.1f, 100.0f);
}

glm::vec2 Renderer::project_to_screen(const glm::vec3 &world, float &ndc_z, bool &ok) const
{
    const glm::mat4 clip_mtx = projection_matrix() * camera.view_matrix();
    const glm::vec4 clip = clip_mtx * glm::vec4(world, 1.0f);

    if (std::fabs(clip.w) < 1e-6f || clip.w <= 0.0f) {
        ok = false;
        ndc_z = 1.0f;
        return glm::vec2(0.0f);
    }

    const glm::vec3 ndc = glm::vec3(clip) / clip.w;
    ndc_z = ndc.z;
    ok = true;

    const float sx = ((ndc.x * 0.5f) + 0.5f) * static_cast<float>(m_width);
    const float sy = (1.0f - ((ndc.y * 0.5f) + 0.5f)) * static_cast<float>(m_height);
    return glm::vec2(sx, sy);
}

Renderer::PrimitiveHit Renderer::pick_face(double mouse_x, double mouse_y) const
{
    PrimitiveHit hit{};
    if (m_width <= 0 || m_height <= 0) {
        return hit;
    }

    const glm::mat4 inv_view_proj = glm::inverse(projection_matrix() * camera.view_matrix());
    const float ndc_x = (2.0f * static_cast<float>(mouse_x) / static_cast<float>(m_width)) - 1.0f;
    const float ndc_y = 1.0f - (2.0f * static_cast<float>(mouse_y) / static_cast<float>(m_height));

    glm::vec4 near_clip(ndc_x, ndc_y, -1.0f, 1.0f);
    glm::vec4 far_clip(ndc_x, ndc_y, 1.0f, 1.0f);
    glm::vec4 near_world4 = inv_view_proj * near_clip;
    glm::vec4 far_world4  = inv_view_proj * far_clip;

    if (std::fabs(near_world4.w) < 1e-6f || std::fabs(far_world4.w) < 1e-6f) {
        return hit;
    }

    const glm::vec3 near_world = glm::vec3(near_world4) / near_world4.w;
    const glm::vec3 far_world  = glm::vec3(far_world4) / far_world4.w;
    const glm::vec3 ray_origin = camera.eye();
    const glm::vec3 ray_dir = glm::normalize(far_world - near_world);

    float t = 0.0f;
    glm::vec3 hit_normal(0.0f);
    if (!ray_box_intersection(ray_origin, ray_dir, t, hit_normal)) {
        return hit;
    }

    const glm::vec3 world_hit = ray_origin + (ray_dir * t);
    const int face_idx = face_index_from_normal(hit_normal);
    if (face_idx < 0) {
        return hit;
    }

    float ndc_z = 1.0f;
    bool ok = false;
    (void)project_to_screen(world_hit, ndc_z, ok);
    if (!ok) {
        return hit;
    }

    hit.type = PrimitiveType::Face;
    hit.index = face_idx;
    hit.world_pos = world_hit;
    hit.normal = hit_normal;
    hit.depth = depth_from_ndc(ndc_z);
    return hit;
}

Renderer::PrimitiveHit Renderer::pick_edge(double mouse_x, double mouse_y) const
{
    PrimitiveHit best{};
    const glm::vec2 mouse(static_cast<float>(mouse_x), static_cast<float>(mouse_y));
    constexpr float kEdgeThresholdPx = 10.0f;

    float best_depth = std::numeric_limits<float>::infinity();
    float best_distance = std::numeric_limits<float>::infinity();

    for (int i = 0; i < 12; ++i) {
        float z_a = 1.0f;
        float z_b = 1.0f;
        bool ok_a = false;
        bool ok_b = false;

        const glm::vec2 a = project_to_screen(CUBE_EDGES[i].a, z_a, ok_a);
        const glm::vec2 b = project_to_screen(CUBE_EDGES[i].b, z_b, ok_b);
        if (!ok_a || !ok_b) {
            continue;
        }

        const glm::vec2 ab = b - a;
        const float ab_len_sq = glm::dot(ab, ab);
        if (ab_len_sq < 1e-6f) {
            continue;
        }

        float t = glm::dot(mouse - a, ab) / ab_len_sq;
        t = glm::clamp(t, 0.0f, 1.0f);

        const glm::vec2 closest = a + (ab * t);
        const float dist = glm::length(mouse - closest);
        if (dist > kEdgeThresholdPx) {
            continue;
        }

        const float ndc_z = glm::mix(z_a, z_b, t);
        const float depth = depth_from_ndc(ndc_z);

        if (depth < best_depth || (std::fabs(depth - best_depth) < 1e-4f && dist < best_distance)) {
            best.type = PrimitiveType::Edge;
            best.index = i;
            best.world_pos = glm::mix(CUBE_EDGES[i].a, CUBE_EDGES[i].b, t);
            best.depth = depth;
            best_depth = depth;
            best_distance = dist;
        }
    }

    return best;
}

void Renderer::update_hovered(double mouse_x, double mouse_y)
{
    const PrimitiveHit face_hit = pick_face(mouse_x, mouse_y);
    const PrimitiveHit edge_hit = pick_edge(mouse_x, mouse_y);

    if (face_hit.valid() && edge_hit.valid()) {
        m_hovered_hit = (edge_hit.depth <= face_hit.depth) ? edge_hit : face_hit;
    } else if (edge_hit.valid()) {
        m_hovered_hit = edge_hit;
    } else if (face_hit.valid()) {
        m_hovered_hit = face_hit;
    } else {
        m_hovered_hit = PrimitiveHit{};
    }
}

void Renderer::clear_hovered()
{
    m_hovered_hit = PrimitiveHit{};
}

void Renderer::select_hovered()
{
    m_selected_hit = m_hovered_hit;
}

void Renderer::clear_selection()
{
    m_selected_hit = PrimitiveHit{};
}

bool Renderer::set_pivot_from_click(double mouse_x, double mouse_y)
{
    const PrimitiveHit face_hit = pick_face(mouse_x, mouse_y);
    const PrimitiveHit edge_hit = pick_edge(mouse_x, mouse_y);

    PrimitiveHit pivot_hit{};

    if (m_selected_hit.valid() && m_selected_hit.type == PrimitiveType::Face &&
        face_hit.valid() && face_hit.index == m_selected_hit.index) {
        pivot_hit = face_hit;
    } else if (m_selected_hit.valid() && m_selected_hit.type == PrimitiveType::Edge &&
               edge_hit.valid() && edge_hit.index == m_selected_hit.index) {
        pivot_hit = edge_hit;
    } else if (face_hit.valid()) {
        pivot_hit = face_hit;
    } else if (edge_hit.valid()) {
        pivot_hit = edge_hit;
    }

    if (!pivot_hit.valid()) {
        return false;
    }

    camera.set_target(pivot_hit.world_pos);
    m_pivot_marker_pos = pivot_hit.world_pos;
    m_pivot_marker_normal = face_hit.valid()
        ? face_hit.normal
        : glm::normalize(camera.eye() - pivot_hit.world_pos);
    m_pivot_marker_until = std::chrono::steady_clock::now() + std::chrono::milliseconds(850);
    return true;
}

void Renderer::configure_sketch_plane(const glm::vec3 &origin, const glm::vec3 &normal)
{
    m_sketch_plane.origin = origin;
    m_sketch_plane.normal = glm::normalize(normal);

    const glm::vec3 up_hint = (std::fabs(m_sketch_plane.normal.y) > 0.9f)
        ? glm::vec3(1.0f, 0.0f, 0.0f)
        : glm::vec3(0.0f, 1.0f, 0.0f);
    m_sketch_plane.x_axis = glm::normalize(glm::cross(up_hint, m_sketch_plane.normal));
    m_sketch_plane.y_axis = glm::normalize(glm::cross(m_sketch_plane.normal, m_sketch_plane.x_axis));
}

glm::vec3 Renderer::cube_face_center(int face_index) const
{
    switch (face_index) {
    case 0: return glm::vec3(0.0f, 0.0f, 0.5f);
    case 1: return glm::vec3(0.0f, 0.0f, -0.5f);
    case 2: return glm::vec3(-0.5f, 0.0f, 0.0f);
    case 3: return glm::vec3(0.5f, 0.0f, 0.0f);
    case 4: return glm::vec3(0.0f, 0.5f, 0.0f);
    case 5: return glm::vec3(0.0f, -0.5f, 0.0f);
    default: return glm::vec3(0.0f);
    }
}

glm::vec3 Renderer::cube_face_normal(int face_index) const
{
    switch (face_index) {
    case 0: return glm::vec3(0.0f, 0.0f, 1.0f);
    case 1: return glm::vec3(0.0f, 0.0f, -1.0f);
    case 2: return glm::vec3(-1.0f, 0.0f, 0.0f);
    case 3: return glm::vec3(1.0f, 0.0f, 0.0f);
    case 4: return glm::vec3(0.0f, 1.0f, 0.0f);
    case 5: return glm::vec3(0.0f, -1.0f, 0.0f);
    default: return glm::vec3(0.0f, 0.0f, 1.0f);
    }
}

glm::vec3 Renderer::sketch_world_from_local(const glm::vec2 &local) const
{
    return m_sketch_plane.origin
         + (m_sketch_plane.x_axis * local.x)
         + (m_sketch_plane.y_axis * local.y);
}

glm::vec2 Renderer::sketch_local_from_world(const glm::vec3 &world) const
{
    const glm::vec3 offset = world - m_sketch_plane.origin;
    return glm::vec2(glm::dot(offset, m_sketch_plane.x_axis),
                     glm::dot(offset, m_sketch_plane.y_axis));
}

bool Renderer::raycast_to_sketch_plane(double mouse_x, double mouse_y, glm::vec3 &world, glm::vec2 &local) const
{
    if (!m_sketch_active || m_width <= 0 || m_height <= 0) {
        return false;
    }

    const glm::mat4 inv_view_proj = glm::inverse(projection_matrix() * camera.view_matrix());
    const float ndc_x = (2.0f * static_cast<float>(mouse_x) / static_cast<float>(m_width)) - 1.0f;
    const float ndc_y = 1.0f - (2.0f * static_cast<float>(mouse_y) / static_cast<float>(m_height));

    glm::vec4 near_clip(ndc_x, ndc_y, -1.0f, 1.0f);
    glm::vec4 far_clip(ndc_x, ndc_y, 1.0f, 1.0f);
    glm::vec4 near_world4 = inv_view_proj * near_clip;
    glm::vec4 far_world4  = inv_view_proj * far_clip;
    if (std::fabs(near_world4.w) < 1e-6f || std::fabs(far_world4.w) < 1e-6f) {
        return false;
    }

    const glm::vec3 near_world = glm::vec3(near_world4) / near_world4.w;
    const glm::vec3 far_world = glm::vec3(far_world4) / far_world4.w;
    const glm::vec3 ray_origin = camera.eye();
    const glm::vec3 ray_dir = glm::normalize(far_world - near_world);
    const float denom = glm::dot(ray_dir, m_sketch_plane.normal);
    if (std::fabs(denom) < 1e-6f) {
        return false;
    }

    const float t = glm::dot(m_sketch_plane.origin - ray_origin, m_sketch_plane.normal) / denom;
    if (t < 0.0f) {
        return false;
    }

    world = ray_origin + (ray_dir * t);
    local = sketch_local_from_world(world);
    return true;
}

glm::vec2 Renderer::snapped_sketch_local(const glm::vec2 &local, bool &snapped_to_vertex) const
{
    glm::vec2 snapped = local;
    snapped_to_vertex = false;

    if (m_sketch_snap_to_grid) {
        snapped.x = std::round(snapped.x / m_sketch_grid_spacing) * m_sketch_grid_spacing;
        snapped.y = std::round(snapped.y / m_sketch_grid_spacing) * m_sketch_grid_spacing;
    }

    float best_dist_sq = std::numeric_limits<float>::infinity();
    auto consider_vertex = [&](const glm::vec2 &v) {
        const float d = glm::dot(v - local, v - local);
        if (d < best_dist_sq) {
            best_dist_sq = d;
            snapped = v;
            snapped_to_vertex = true;
        }
    };

    for (const SketchLine &line : m_sketch_lines) {
        consider_vertex(line.start);
        consider_vertex(line.end);
    }
    for (const SketchRectangle &rect : m_sketch_rectangles) {
        consider_vertex(rect.min_corner);
        consider_vertex(rect.max_corner);
        consider_vertex({rect.min_corner.x, rect.max_corner.y});
        consider_vertex({rect.max_corner.x, rect.min_corner.y});
    }
    for (const SketchCircle &circle : m_sketch_circles) {
        consider_vertex(circle.center + glm::vec2(circle.radius, 0.0f));
        consider_vertex(circle.center + glm::vec2(-circle.radius, 0.0f));
        consider_vertex(circle.center + glm::vec2(0.0f, circle.radius));
        consider_vertex(circle.center + glm::vec2(0.0f, -circle.radius));
        consider_vertex(circle.center);
    }
    // Do not snap an in-progress entity back onto its own anchor point.

    if (best_dist_sq > 0.0064f) {
        snapped_to_vertex = false;
        if (m_sketch_snap_to_grid) {
            return snapped;
        }
        return local;
    }

    return snapped;
}

glm::vec2 Renderer::constrained_line_endpoint(const glm::vec2 &anchor,
                                              const glm::vec2 &candidate,
                                              SketchConstraintType &constraint) const
{
    glm::vec2 result = candidate;
    constraint = SketchConstraintType::None;

    const glm::vec2 delta = candidate - anchor;
    const float abs_dx = std::fabs(delta.x);
    const float abs_dy = std::fabs(delta.y);

    if (abs_dx < 1e-4f && abs_dy < 1e-4f) {
        return result;
    }

    if (abs_dy <= std::max(0.05f, abs_dx * 0.18f)) {
        result.y = anchor.y;
        constraint = SketchConstraintType::Horizontal;
    } else if (abs_dx <= std::max(0.05f, abs_dy * 0.18f)) {
        result.x = anchor.x;
        constraint = SketchConstraintType::Vertical;
    }

    return result;
}

void Renderer::begin_sketch(const PrimitiveHit &selected_hit)
{
    if (selected_hit.valid() && selected_hit.type == PrimitiveType::Face) {
        configure_sketch_plane(cube_face_center(selected_hit.index), cube_face_normal(selected_hit.index));
        camera.set_target(m_sketch_plane.origin);
    } else {
        configure_sketch_plane(glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        camera.set_target(m_sketch_plane.origin);
    }

    m_sketch_active = true;
    m_sketch_cursor_valid = false;
    m_sketch_cursor_snapped = false;
    m_sketch_entity_in_progress = false;
    m_sketch_preview_constraint = SketchConstraintType::None;
    clear_hovered();
}

void Renderer::end_sketch()
{
    m_sketch_active = false;
    m_sketch_cursor_valid = false;
    m_sketch_cursor_snapped = false;
    m_sketch_entity_in_progress = false;
    m_sketch_preview_constraint = SketchConstraintType::None;
}

void Renderer::set_sketch_tool(SketchTool tool)
{
    m_sketch_tool = tool;
    cancel_sketch_preview();
}

const char *Renderer::sketch_tool_name() const
{
    switch (m_sketch_tool) {
    case SketchTool::Line: return "LINE";
    case SketchTool::Rectangle: return "RECTANGLE";
    case SketchTool::Circle: return "CIRCLE";
    }
    return "LINE";
}

std::string Renderer::sketch_constraint_hint() const
{
    if (!m_sketch_active) {
        return "";
    }

    if (m_sketch_tool == SketchTool::Rectangle) {
        return "H/V CONSTRAINED";
    }
    if (m_sketch_tool == SketchTool::Circle) {
        return m_sketch_entity_in_progress ? "RADIUS PREVIEW" : "RADIUS CONSTRAINED";
    }
    if (m_sketch_preview_constraint == SketchConstraintType::Horizontal) {
        return "HORIZONTAL LOCK";
    }
    if (m_sketch_preview_constraint == SketchConstraintType::Vertical) {
        return "VERTICAL LOCK";
    }
    return "FREE ANGLE";
}

bool Renderer::update_sketch_cursor(double mouse_x, double mouse_y)
{
    glm::vec3 world(0.0f);
    glm::vec2 local(0.0f);
    if (!raycast_to_sketch_plane(mouse_x, mouse_y, world, local)) {
        m_sketch_cursor_valid = false;
        m_sketch_cursor_snapped = false;
        return false;
    }

    bool snapped_to_vertex = false;
    glm::vec2 snapped = snapped_sketch_local(local, snapped_to_vertex);
    SketchConstraintType constraint = SketchConstraintType::None;
    if (m_sketch_entity_in_progress && m_sketch_tool == SketchTool::Line) {
        snapped = constrained_line_endpoint(m_sketch_anchor, snapped, constraint);
    }

    m_sketch_cursor_local = snapped;
    m_sketch_cursor_world = sketch_world_from_local(snapped);
    m_sketch_cursor_valid = true;
    m_sketch_cursor_snapped = snapped_to_vertex || (glm::length(snapped - local) > 1e-4f);
    m_sketch_preview_local = snapped;
    m_sketch_preview_world = m_sketch_cursor_world;
    m_sketch_preview_constraint = constraint;
    return true;
}

bool Renderer::handle_sketch_click(double mouse_x, double mouse_y)
{
    if (!update_sketch_cursor(mouse_x, mouse_y) || !m_sketch_cursor_valid) {
        return false;
    }

    if (!m_sketch_entity_in_progress) {
        m_sketch_anchor = m_sketch_cursor_local;
        m_sketch_preview_local = m_sketch_cursor_local;
        m_sketch_preview_world = m_sketch_cursor_world;
        m_sketch_preview_constraint = SketchConstraintType::None;
        m_sketch_entity_in_progress = true;
        return true;
    }

    if (m_sketch_tool == SketchTool::Line) {
        if (glm::length(m_sketch_preview_local - m_sketch_anchor) < 1e-4f) {
            return false;
        }

        m_sketch_lines.push_back({m_sketch_anchor, m_sketch_preview_local, m_sketch_preview_constraint});
        m_sketch_entity_in_progress = false;
        m_sketch_preview_constraint = SketchConstraintType::None;
        return true;
    }

    if (m_sketch_tool == SketchTool::Rectangle) {
        const glm::vec2 min_corner(std::min(m_sketch_anchor.x, m_sketch_preview_local.x),
                                   std::min(m_sketch_anchor.y, m_sketch_preview_local.y));
        const glm::vec2 max_corner(std::max(m_sketch_anchor.x, m_sketch_preview_local.x),
                                   std::max(m_sketch_anchor.y, m_sketch_preview_local.y));
        if (glm::length(max_corner - min_corner) < 1e-4f) {
            return false;
        }
        m_sketch_rectangles.push_back({min_corner, max_corner});
        m_sketch_entity_in_progress = false;
        m_sketch_preview_constraint = SketchConstraintType::None;
        return true;
    }

    const float radius = glm::length(m_sketch_preview_local - m_sketch_anchor);
    if (radius < 1e-4f) {
        return false;
    }
    m_sketch_circles.push_back({m_sketch_anchor, radius});
    m_sketch_entity_in_progress = false;
    m_sketch_preview_constraint = SketchConstraintType::Radius;
    return true;
}

void Renderer::cancel_sketch_preview()
{
    m_sketch_entity_in_progress = false;
    m_sketch_preview_constraint = SketchConstraintType::None;
}

bool Renderer::pivot_marker_visible() const
{
    return std::chrono::steady_clock::now() < m_pivot_marker_until;
}

Renderer::~Renderer()
{
    glDeleteVertexArrays(1, &m_axis_vao);
    glDeleteBuffers(1, &m_axis_vbo);
    glDeleteVertexArrays(1, &m_cube_vao);
    glDeleteBuffers(1, &m_cube_vbo);
    glDeleteVertexArrays(1, &m_solid_cube_vao);
    glDeleteBuffers(1, &m_solid_cube_vbo);
    glDeleteVertexArrays(1, &m_pivot_marker_vao);
    glDeleteBuffers(1, &m_pivot_marker_vbo);
    glDeleteVertexArrays(1, &m_overlay_vao);
    glDeleteBuffers(1, &m_overlay_vbo);
    glDeleteProgram(m_program);
}
