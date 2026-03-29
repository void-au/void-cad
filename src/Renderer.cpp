#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <limits>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
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

static std::string trim_copy(const std::string &s)
{
    const std::size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    const std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

static std::string uppercase_copy(const std::string &s)
{
    std::string out = s;
    for (char &ch : out) {
        if (ch >= 'a' && ch <= 'z') {
            ch = static_cast<char>(ch - 'a' + 'A');
        }
    }
    return out;
}

static bool parse_step_id_ref(const std::string &token, int &out_id)
{
    std::string t = trim_copy(token);
    if (t.size() < 2 || t[0] != '#') {
        return false;
    }
    try {
        out_id = std::stoi(t.substr(1));
        return out_id > 0;
    } catch (...) {
        return false;
    }
}

static bool parse_step_cartesian_coords(const std::string &entity_body, glm::vec3 &out_point)
{
    const std::string body_upper = uppercase_copy(entity_body);
    if (body_upper.find("CARTESIAN_POINT") == std::string::npos) {
        return false;
    }

    const std::size_t coords_open = entity_body.rfind("(");
    const std::size_t coords_close = entity_body.rfind(")");
    if (coords_open == std::string::npos || coords_close == std::string::npos || coords_close <= coords_open) {
        return false;
    }

    std::string coords = entity_body.substr(coords_open + 1, coords_close - coords_open - 1);
    std::vector<float> values;
    std::size_t start = 0;
    while (start < coords.size()) {
        std::size_t comma = coords.find(',', start);
        std::string part = (comma == std::string::npos)
            ? coords.substr(start)
            : coords.substr(start, comma - start);
        part = trim_copy(part);
        if (!part.empty() && part != "*" && part != "$") {
            try {
                values.push_back(std::stof(part));
            } catch (...) {
                return false;
            }
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }

    if (values.size() < 2) {
        return false;
    }
    out_point = glm::vec3(values[0], values[1], (values.size() >= 3 ? values[2] : 0.0f));
    return true;
}

static bool parse_step_vertex_point_ref(const std::string &entity_body, int &out_point_id)
{
    const std::string body_upper = uppercase_copy(entity_body);
    if (body_upper.find("VERTEX_POINT") == std::string::npos) {
        return false;
    }

    const std::size_t hash = entity_body.find('#');
    if (hash == std::string::npos) {
        return false;
    }

    std::size_t end = hash + 1;
    while (end < entity_body.size() && entity_body[end] >= '0' && entity_body[end] <= '9') {
        ++end;
    }
    return parse_step_id_ref(entity_body.substr(hash, end - hash), out_point_id);
}

static bool parse_step_edge_curve_refs(const std::string &entity_body, int &out_v1, int &out_v2)
{
    const std::string body_upper = uppercase_copy(entity_body);
    if (body_upper.find("EDGE_CURVE") == std::string::npos) {
        return false;
    }

    std::vector<int> refs;
    for (std::size_t i = 0; i < entity_body.size(); ++i) {
        if (entity_body[i] != '#') {
            continue;
        }
        std::size_t end = i + 1;
        while (end < entity_body.size() && entity_body[end] >= '0' && entity_body[end] <= '9') {
            ++end;
        }
        int id = -1;
        if (parse_step_id_ref(entity_body.substr(i, end - i), id)) {
            refs.push_back(id);
        }
        i = end;
    }

    if (refs.size() < 2) {
        return false;
    }
    out_v1 = refs[0];
    out_v2 = refs[1];
    return true;
}

struct ParsedStepEntity {
    int id = -1;
    std::string body;
};

static std::vector<ParsedStepEntity> parse_step_entities(const std::string &raw)
{
    std::vector<ParsedStepEntity> entities;
    std::string current;
    current.reserve(512);

    const std::string normalized = raw;
    for (char ch : normalized) {
        current.push_back(ch);
        if (ch != ';') {
            continue;
        }

        std::string stmt = trim_copy(current);
        current.clear();
        if (stmt.empty() || stmt[0] != '#') {
            continue;
        }

        const std::size_t eq = stmt.find('=');
        if (eq == std::string::npos || eq < 2) {
            continue;
        }

        int id = -1;
        if (!parse_step_id_ref(stmt.substr(0, eq), id)) {
            continue;
        }

        std::string body = trim_copy(stmt.substr(eq + 1));
        if (!body.empty() && body.back() == ';') {
            body.pop_back();
            body = trim_copy(body);
        }
        entities.push_back({id, body});
    }

    return entities;
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

    if (m_has_imported_model) {
        glUniform4f(m_uColorMul, scene_dim, scene_dim, scene_dim, 1.0f);
        glUniform4f(m_uColorAdd, 0.0f, 0.0f, 0.0f, 0.0f);
        if (!m_wireframe) {
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0f, 1.0f);
            for (std::size_t i = 0; i < m_imported_body_render_data.size() && i < m_imported_bodies.size(); ++i) {
                if (!m_imported_bodies[i].visible) continue;
                draw_overlay_triangles(m_imported_body_render_data[i].solid_verts);
            }
            glDisable(GL_POLYGON_OFFSET_FILL);
        }
        for (std::size_t i = 0; i < m_imported_body_render_data.size() && i < m_imported_bodies.size(); ++i) {
            if (!m_imported_bodies[i].visible) continue;
            draw_overlay_lines(m_imported_body_render_data[i].line_verts, 1.5f);
        }
    } else if (m_wireframe) {
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

    if (!m_sketch_active && !m_has_imported_model) {
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
        const float spacing = effective_sketch_grid_spacing();
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
            // Compute cursor size in world space based on screen projection
            // Convert a 10-pixel distance at cursor position to world space
            float ndc_z = 0.0f;
            bool ok = false;
            const glm::vec2 cursor_screen = project_to_screen(m_sketch_cursor_world, ndc_z, ok);
            const glm::vec2 offset_screen(cursor_screen.x + 10.0f, cursor_screen.y);
            
            // Unproject the offset point back to world space to get world-space size
            const glm::mat4 view_inv = glm::inverse(camera.view_matrix());
            const glm::mat4 proj_inv = glm::inverse(projection_matrix());
            glm::vec4 offset_ndc(offset_screen.x / m_width * 2.0f - 1.0f,
                                 1.0f - offset_screen.y / m_height * 2.0f,
                                 ndc_z, 1.0f);
            glm::vec4 offset_eye = proj_inv * offset_ndc;
            offset_eye.w = 1.0f;
            glm::vec3 offset_world = glm::vec3(view_inv * offset_eye);
            float cursor_size = glm::length(offset_world - m_sketch_cursor_world);
            cursor_size = std::clamp(cursor_size, 0.025f, 0.25f);
            
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

// Split the top-level comma-separated arguments of a STEP entity body,
// respecting nested parentheses and single-quoted strings.
static std::vector<std::string> split_step_args(const std::string &body)
{
    const std::size_t open = body.find('(');
    if (open == std::string::npos) return {};
    int depth = 0;
    std::size_t close = std::string::npos;
    for (std::size_t i = open; i < body.size(); ++i) {
        if (body[i] == '(') ++depth;
        else if (body[i] == ')') { --depth; if (depth == 0) { close = i; break; } }
    }
    if (close == std::string::npos) return {};
    const std::string inner = body.substr(open + 1, close - open - 1);
    std::vector<std::string> args;
    std::string cur;
    int nest = 0; bool in_str = false;
    for (const char c : inner) {
        if (c == '\'') in_str = !in_str;
        if (!in_str) { if (c == '(') ++nest; else if (c == ')') --nest; }
        if (c == ',' && nest == 0 && !in_str) { args.push_back(trim_copy(cur)); cur.clear(); }
        else cur.push_back(c);
    }
    args.push_back(trim_copy(cur));
    return args;
}

// Parse a DIRECTION entity body → normalised unit vec3.
// Format: DIRECTION('name', (dx, dy, dz))
static bool parse_direction_entity(const std::string &body, glm::vec3 &out)
{
    const auto args = split_step_args(body);
    if (args.size() < 2) return false;
    const auto &tup = args[1];
    const std::size_t op = tup.find('('), cl = tup.rfind(')');
    if (op == std::string::npos || cl == std::string::npos || cl <= op) return false;
    const std::string inner = tup.substr(op + 1, cl - op - 1);
    std::vector<float> vals;
    for (std::size_t s = 0;;) {
        const std::size_t cm = inner.find(',', s);
        const std::string p = trim_copy(cm == std::string::npos ? inner.substr(s) : inner.substr(s, cm - s));
        if (!p.empty()) { try { vals.push_back(std::stof(p)); } catch (...) {} }
        if (cm == std::string::npos) break;
        s = cm + 1;
    }
    if (vals.size() < 3) return false;
    const glm::vec3 v(vals[0], vals[1], vals[2]);
    if (glm::length(v) < 1e-8f) return false;
    out = glm::normalize(v);
    return true;
}

struct CircleGeom {
    glm::vec3 center;
    glm::vec3 x_axis;  // local X in circle plane (angle-zero direction)
    glm::vec3 y_axis;  // local Y in circle plane (= z_axis cross x_axis)
    float radius = 0.0f;
};

// Tessellate a circular arc into line segments.
// same_sense=true → CCW sweep from start_pt to end_pt (STEP convention).
static void tessellate_arc(
    const CircleGeom &cg,
    const glm::vec3  &start_pt,
    const glm::vec3  &end_pt,
    bool              same_sense,
    std::vector<std::pair<glm::vec3, glm::vec3>> &out_segs)
{
    constexpr int kSegsPerRev = 64;
    const bool full_circle = glm::length(start_pt - end_pt)
                             < 1e-4f * std::max(cg.radius, 1e-6f);
    const float theta0 = std::atan2(glm::dot(start_pt - cg.center, cg.y_axis),
                                    glm::dot(start_pt - cg.center, cg.x_axis));
    float sweep;
    if (full_circle) {
        sweep = glm::two_pi<float>();
    } else {
        const float theta1 = std::atan2(glm::dot(end_pt - cg.center, cg.y_axis),
                                        glm::dot(end_pt - cg.center, cg.x_axis));
        if (same_sense) {
            sweep = theta1 - theta0;
            if (sweep <= 0.0f) sweep += glm::two_pi<float>();
        } else {
            sweep = theta1 - theta0;
            if (sweep >= 0.0f) sweep -= glm::two_pi<float>();
        }
    }
    if (std::fabs(sweep) < 1e-6f) return;
    const int n = std::max(4, static_cast<int>(
        std::fabs(sweep) / glm::two_pi<float>() * kSegsPerRev));
    const float dth = sweep / static_cast<float>(n);
    glm::vec3 prev = cg.center
                   + cg.x_axis * (cg.radius * std::cos(theta0))
                   + cg.y_axis * (cg.radius * std::sin(theta0));
    for (int i = 1; i <= n; ++i) {
        const float th = theta0 + dth * static_cast<float>(i);
        const glm::vec3 next = cg.center
                             + cg.x_axis * (cg.radius * std::cos(th))
                             + cg.y_axis * (cg.radius * std::sin(th));
        out_segs.emplace_back(prev, next);
        prev = next;
    }
}

bool Renderer::import_step_file(const std::string &path, std::string &error_message)
{
    error_message.clear();
    m_imported_bodies.clear();
    m_imported_body_render_data.clear();

    std::ifstream in(path);
    if (!in) {
        error_message = "Failed to open file.";
        return false;
    }

    std::string raw((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (raw.empty()) {
        error_message = "STEP file is empty.";
        return false;
    }

    const std::vector<ParsedStepEntity> entities = parse_step_entities(raw);
    if (entities.empty()) {
        error_message = "No STEP entities found.";
        return false;
    }

    // ---------- entity map ----------
    std::unordered_map<int, std::string> entity_map;
    entity_map.reserve(entities.size() * 2);
    for (const auto &e : entities) {
        entity_map[e.id] = e.body;
    }

    auto get_type = [](const std::string &body) -> std::string {
        const std::size_t p = body.find('(');
        if (p == std::string::npos) return "";
        return uppercase_copy(trim_copy(body.substr(0, p)));
    };
    auto get_type_of = [&](int id) -> std::string {
        const auto it = entity_map.find(id);
        return it != entity_map.end() ? get_type(it->second) : "";
    };
    auto get_refs = [](const std::string &s) -> std::vector<int> {
        std::vector<int> refs;
        for (std::size_t i = 0; i < s.size(); ) {
            if (s[i] != '#') { ++i; continue; }
            std::size_t end = i + 1;
            while (end < s.size() && s[end] >= '0' && s[end] <= '9') ++end;
            if (end > i + 1) {
                try { refs.push_back(std::stoi(s.substr(i + 1, end - i - 1))); } catch (...) {}
            }
            i = end;
        }
        return refs;
    };
    auto get_body_refs = [&](int id) -> std::vector<int> {
        const auto it = entity_map.find(id);
        if (it == entity_map.end()) return {};
        return get_refs(it->second);
    };
    // .T. = same sense (true), .F. = reversed
    auto get_orientation = [](const std::string &body) -> bool {
        const std::size_t t = body.rfind(".T.");
        const std::size_t f = body.rfind(".F.");
        if (t == std::string::npos && f == std::string::npos) return true;
        if (t == std::string::npos) return false;
        if (f == std::string::npos) return true;
        return t > f;
    };

    // ---------- pass 1: geometry primitives ----------
    std::unordered_map<int, glm::vec3> points;
    std::unordered_map<int, int>       vertex_to_point;
    std::unordered_map<int, glm::vec3> directions;

    struct AxisPlacement { int origin_id = -1, z_id = -1, x_id = -1; };
    std::unordered_map<int, AxisPlacement> axis_placements;

    struct CircleRaw { int placement_id = -1; float radius = 0.0f; };
    std::unordered_map<int, CircleRaw> circle_raws;

    for (const auto &[id, body] : entity_map) {
        const std::string type = get_type(body);
        if (type == "CARTESIAN_POINT") {
            glm::vec3 p;
            if (parse_step_cartesian_coords(body, p)) points[id] = p;
        } else if (type == "VERTEX_POINT") {
            const auto refs = get_refs(body);
            if (!refs.empty()) vertex_to_point[id] = refs[0];
        } else if (type == "DIRECTION") {
            glm::vec3 d;
            if (parse_direction_entity(body, d)) directions[id] = d;
        } else if (type == "AXIS2_PLACEMENT_3D") {
            const auto args = split_step_args(body);
            if (args.size() >= 3) {
                int oid = -1, zid = -1, xid = -1;
                if (parse_step_id_ref(args[1], oid) && parse_step_id_ref(args[2], zid)) {
                    if (args.size() >= 4 && !args[3].empty() && args[3][0] == '#')
                        parse_step_id_ref(args[3], xid);
                    axis_placements[id] = {oid, zid, xid};
                }
            }
        } else if (type == "CIRCLE") {
            const auto args = split_step_args(body);
            if (args.size() >= 3) {
                int pid = -1;
                if (parse_step_id_ref(args[1], pid)) {
                    try {
                        const float r = std::stof(trim_copy(args[2]));
                        if (r > 0.0f) circle_raws[id] = {pid, r};
                    } catch (...) {}
                }
            }
        }
    }
    if (points.empty()) {
        error_message = "No CARTESIAN_POINT entities found.";
        return false;
    }
    auto vertex_pos = [&](int vid, glm::vec3 &out) -> bool {
        const auto vit = vertex_to_point.find(vid);
        if (vit == vertex_to_point.end()) return false;
        const auto pit = points.find(vit->second);
        if (pit == points.end()) return false;
        out = pit->second;
        return true;
    };

    // ---------- resolve circle geometries ----------
    std::unordered_map<int, CircleGeom> circle_geoms;
    for (const auto &[cid, cr] : circle_raws) {
        const auto ap_it = axis_placements.find(cr.placement_id);
        if (ap_it == axis_placements.end()) continue;
        const auto &ap = ap_it->second;
        const auto oit = points.find(ap.origin_id);
        if (oit == points.end()) continue;
        const auto zit = directions.find(ap.z_id);
        if (zit == directions.end()) continue;
        const glm::vec3 origin = oit->second;
        const glm::vec3 z_axis = zit->second;
        glm::vec3 x_axis;
        if (ap.x_id >= 0) {
            const auto xit = directions.find(ap.x_id);
            if (xit == directions.end()) continue;
            const glm::vec3 candidate = xit->second - z_axis * glm::dot(xit->second, z_axis);
            if (glm::length(candidate) < 1e-8f) continue;
            x_axis = glm::normalize(candidate);
        } else {
            const glm::vec3 up = (std::fabs(z_axis.y) < 0.9f)
                ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
            x_axis = glm::normalize(glm::cross(up, z_axis));
            if (glm::length(x_axis) < 1e-8f) continue;
        }
        const glm::vec3 y_axis = glm::normalize(glm::cross(z_axis, x_axis));
        circle_geoms[cid] = {origin, x_axis, y_axis, cr.radius};
    }

    // ---------- pass 2: EDGE_CURVE map (with curve ref + sense) ----------
    struct EdgeCurveEntry { int vs = -1, ve = -1, curve_id = -1; bool same_sense = true; };
    std::unordered_map<int, EdgeCurveEntry> edge_curves;
    for (const auto &[id, body] : entity_map) {
        if (get_type(body) != "EDGE_CURVE") continue;
        const auto refs = get_refs(body);
        if (refs.size() < 2) continue;
        edge_curves[id] = { refs[0], refs[1],
                            refs.size() >= 3 ? refs[2] : -1,
                            get_orientation(body) };
    }

    // Tessellate one edge: arc if it has a resolved circle, else straight line.
    auto tessellate_edge_entry = [&](const EdgeCurveEntry &ec,
                                     std::vector<std::pair<glm::vec3,glm::vec3>> &out_segs) {
        glm::vec3 pa, pb;
        if (!vertex_pos(ec.vs, pa) || !vertex_pos(ec.ve, pb)) return;
        if (ec.curve_id >= 0) {
            const auto cit = circle_geoms.find(ec.curve_id);
            if (cit != circle_geoms.end()) {
                tessellate_arc(cit->second, pa, pb, ec.same_sense, out_segs);
                return;
            }
        }
        out_segs.emplace_back(pa, pb);
    };

    auto parsed_name_or = [&](const std::string &body, const std::string &fallback) -> std::string {
        const auto args = split_step_args(body);
        if (args.empty()) return fallback;
        std::string name = trim_copy(args[0]);
        if (name.size() >= 2 && name.front() == '\'' && name.back() == '\'') {
            name = name.substr(1, name.size() - 2);
        }
        name = trim_copy(name);
        if (name.empty() || name == "$" || name == "*") return fallback;
        return name;
    };

    // ---------- discover STEP bodies and associated face ids ----------
    struct BodySeed {
        std::string label;
        std::vector<int> face_ids;
    };
    std::vector<BodySeed> body_seeds;
    std::unordered_set<int> assigned_faces;

    auto collect_faces_from = [&](int root_id, std::unordered_set<int> &out_faces) {
        std::vector<int> stack{root_id};
        std::unordered_set<int> visited;
        while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();
            if (!visited.insert(cur).second) continue;
            const std::string type = get_type_of(cur);
            if (type == "ADVANCED_FACE") {
                out_faces.insert(cur);
                continue;
            }
            for (int r : get_body_refs(cur)) stack.push_back(r);
        }
    };

    for (const auto &[id, body] : entity_map) {
        const std::string type = get_type(body);
        if (type != "MANIFOLD_SOLID_BREP" && type != "BREP_WITH_VOIDS") continue;
        const auto refs = get_refs(body);
        if (refs.empty()) continue;

        std::unordered_set<int> body_faces;
        for (int r : refs) collect_faces_from(r, body_faces);
        if (body_faces.empty()) continue;

        BodySeed seed;
        seed.label = parsed_name_or(body, "Body " + std::to_string(body_seeds.size() + 1));
        seed.face_ids.assign(body_faces.begin(), body_faces.end());
        std::sort(seed.face_ids.begin(), seed.face_ids.end());
        body_seeds.push_back(seed);
        assigned_faces.insert(body_faces.begin(), body_faces.end());
    }

    std::vector<int> all_faces;
    all_faces.reserve(entity_map.size() / 4 + 1);
    for (const auto &[id, body] : entity_map) {
        if (get_type(body) == "ADVANCED_FACE") all_faces.push_back(id);
    }
    std::sort(all_faces.begin(), all_faces.end());

    std::vector<int> unassigned_faces;
    for (int f : all_faces) {
        if (assigned_faces.find(f) == assigned_faces.end()) unassigned_faces.push_back(f);
    }
    if (!unassigned_faces.empty()) {
        body_seeds.push_back({"Body " + std::to_string(body_seeds.size() + 1), std::move(unassigned_faces)});
    }
    if (body_seeds.empty() && !all_faces.empty()) {
        body_seeds.push_back({"Body 1", all_faces});
    }

    // ---------- pass 3: ADVANCED_FACE topology -> per-body raw faces + edges ----------
    struct RawFace {
        std::vector<glm::vec3> polygon;
    };
    struct RawBody {
        std::string label;
        std::vector<RawFace> faces;
        std::vector<std::pair<glm::vec3, glm::vec3>> edges;
        std::unordered_set<unsigned long long> edge_dedupe;
        int edge_entity_count = 0;
    };

    auto append_face_geometry = [&](int face_id, RawBody &out_body) {
        const auto face_it = entity_map.find(face_id);
        if (face_it == entity_map.end()) return;
        const std::string &face_body = face_it->second;
        if (get_type(face_body) != "ADVANCED_FACE") return;

        for (int bound_id : get_refs(face_body)) {
            const std::string btype = get_type_of(bound_id);
            if (btype != "FACE_OUTER_BOUND" && btype != "FACE_BOUND") continue;

            int loop_id = -1;
            for (int br : get_body_refs(bound_id)) {
                if (get_type_of(br) == "EDGE_LOOP") { loop_id = br; break; }
            }
            if (loop_id < 0) continue;

            RawFace face;
            constexpr float kJoinEps = 1e-5f;
            for (int oe_id : get_body_refs(loop_id)) {
                if (get_type_of(oe_id) != "ORIENTED_EDGE") continue;
                const auto oe_it = entity_map.find(oe_id);
                if (oe_it == entity_map.end()) continue;
                const bool same = get_orientation(oe_it->second);

                int ec_id = -1;
                for (int r : get_refs(oe_it->second)) {
                    if (get_type_of(r) == "EDGE_CURVE") { ec_id = r; break; }
                }
                if (ec_id < 0) continue;

                const auto ec_it = edge_curves.find(ec_id);
                if (ec_it == edge_curves.end()) continue;
                const int vs = ec_it->second.vs;
                const int ve = ec_it->second.ve;

                // deduplicated wireframe edges (tessellated if curved)
                const auto key = (static_cast<unsigned long long>(static_cast<unsigned int>(std::min(vs,ve))) << 32ULL)
                                | static_cast<unsigned long long>(static_cast<unsigned int>(std::max(vs,ve)));
                std::vector<std::pair<glm::vec3, glm::vec3>> edge_segs;
                tessellate_edge_entry(ec_it->second, edge_segs);
                if (out_body.edge_dedupe.insert(key).second) {
                    out_body.edge_entity_count += 1;
                    out_body.edges.insert(out_body.edges.end(), edge_segs.begin(), edge_segs.end());
                }

                // Build the face loop from the oriented edge polyline,
                // so curved edges contribute intermediate points.
                std::vector<glm::vec3> edge_pts;
                if (!edge_segs.empty()) {
                    if (same) {
                        edge_pts.push_back(edge_segs.front().first);
                        for (const auto &[a, b] : edge_segs) {
                            (void)a;
                            edge_pts.push_back(b);
                        }
                    } else {
                        edge_pts.push_back(edge_segs.back().second);
                        for (auto it = edge_segs.rbegin(); it != edge_segs.rend(); ++it)
                            edge_pts.push_back(it->first);
                    }
                }
                if (edge_pts.empty()) continue;

                if (face.polygon.empty()) {
                    face.polygon.insert(face.polygon.end(), edge_pts.begin(), edge_pts.end());
                } else {
                    std::size_t start_idx = 0;
                    if (glm::length(face.polygon.back() - edge_pts.front()) <= kJoinEps)
                        start_idx = 1;
                    for (std::size_t i = start_idx; i < edge_pts.size(); ++i)
                        face.polygon.push_back(edge_pts[i]);
                }
            }
            if (face.polygon.size() >= 2
                && glm::length(face.polygon.front() - face.polygon.back()) <= kJoinEps)
                face.polygon.pop_back();
            if (face.polygon.size() >= 3) out_body.faces.push_back(std::move(face));
        }
    };

    std::vector<RawBody> raw_bodies;
    raw_bodies.reserve(std::max<std::size_t>(1, body_seeds.size()));
    for (const auto &seed : body_seeds) {
        RawBody body;
        body.label = seed.label;
        for (int face_id : seed.face_ids) {
            append_face_geometry(face_id, body);
        }
        if (!body.faces.empty() || !body.edges.empty()) {
            raw_bodies.push_back(std::move(body));
        }
    }

    // Fallback: no body topology with faces found — collect all edges as one body.
    if (raw_bodies.empty()) {
        RawBody fallback;
        fallback.label = "Body 1";
        for (const auto &[eid, ec] : edge_curves)
        {
            (void)eid;
            tessellate_edge_entry(ec, fallback.edges);
            fallback.edge_entity_count += 1;
        }
        if (!fallback.edges.empty()) raw_bodies.push_back(std::move(fallback));
    }
    if (raw_bodies.empty()) {
        error_message = "No usable geometry found.";
        return false;
    }

    // ---------- bounding box ----------
    glm::vec3 bb_min(std::numeric_limits<float>::infinity());
    glm::vec3 bb_max(-std::numeric_limits<float>::infinity());
    for (const auto &body : raw_bodies) {
        for (const auto &f : body.faces)
            for (const auto &v : f.polygon) { bb_min = glm::min(bb_min, v); bb_max = glm::max(bb_max, v); }
        for (const auto &[a, b] : body.edges) {
            bb_min = glm::min(bb_min, glm::min(a, b));
            bb_max = glm::max(bb_max, glm::max(a, b));
        }
    }
    const glm::vec3 center   = (bb_min + bb_max) * 0.5f;
    const glm::vec3 bb_size  = bb_max - bb_min;
    const float     max_dim  = std::max({bb_size.x, bb_size.y, bb_size.z});
    if (max_dim < 1e-6f) {
        error_message = "Imported model has near-zero size.";
        return false;
    }
    const float scale = 1.6f / max_dim;
    auto xform = [&](const glm::vec3 &v) -> glm::vec3 { return (v - center) * scale; };

    // ---------- build GPU vertex buffers ----------
    const glm::vec3 light_dir  = glm::normalize(glm::vec3(0.55f, 0.75f, 0.40f));
    const glm::vec3 edge_color(0.75f, 0.75f, 0.75f);

    m_imported_bodies.clear();
    m_imported_body_render_data.clear();

    for (const auto &raw_body : raw_bodies) {
        ImportedBodyItem body_item;
        body_item.label = raw_body.label;
        body_item.face_count = static_cast<int>(raw_body.faces.size());
        body_item.edge_count = raw_body.edge_entity_count;
        body_item.visible = true;

        ImportedBodyRenderData body_render;

        for (const auto &face : raw_body.faces) {
            const auto &p = face.polygon;
            const glm::vec3 e1    = p[1] - p[0];
            const glm::vec3 e2    = p[2] - p[0];
            const glm::vec3 cross = glm::cross(e1, e2);
            const glm::vec3 nrm   = glm::length(cross) > 1e-8f ? glm::normalize(cross) : glm::vec3(0.0f, 0.0f, 1.0f);
            const float shade     = 0.18f + 0.65f * std::fabs(glm::dot(nrm, light_dir));
            const glm::vec3 fc(shade * 0.65f, shade * 0.65f, shade * 0.72f);

            for (std::size_t i = 1; i + 1 < p.size(); ++i) {
                for (const glm::vec3 &v : {p[0], p[i], p[i+1]}) {
                    const glm::vec3 t = xform(v);
                    body_render.solid_verts.push_back(t.x); body_render.solid_verts.push_back(t.y); body_render.solid_verts.push_back(t.z);
                    body_render.solid_verts.push_back(fc.r); body_render.solid_verts.push_back(fc.g); body_render.solid_verts.push_back(fc.b);
                }
            }
        }

        for (const auto &[a, b] : raw_body.edges) {
            const glm::vec3 ta = xform(a), tb = xform(b);
            body_render.line_verts.push_back(ta.x); body_render.line_verts.push_back(ta.y); body_render.line_verts.push_back(ta.z);
            body_render.line_verts.push_back(edge_color.r); body_render.line_verts.push_back(edge_color.g); body_render.line_verts.push_back(edge_color.b);
            body_render.line_verts.push_back(tb.x); body_render.line_verts.push_back(tb.y); body_render.line_verts.push_back(tb.z);
            body_render.line_verts.push_back(edge_color.r); body_render.line_verts.push_back(edge_color.g); body_render.line_verts.push_back(edge_color.b);
        }

        if (body_render.line_verts.empty() && body_render.solid_verts.empty()) continue;
        m_imported_bodies.push_back(std::move(body_item));
        m_imported_body_render_data.push_back(std::move(body_render));
    }

    m_has_imported_model = !m_imported_body_render_data.empty();
    clear_hovered();
    clear_selection();
    camera.reset();
    return m_has_imported_model;
}

void Renderer::clear_imported_model()
{
    m_has_imported_model = false;
    m_imported_bodies.clear();
    m_imported_body_render_data.clear();
    clear_hovered();
    clear_selection();
}

bool Renderer::set_imported_body_visible(std::size_t body_index, bool visible)
{
    if (body_index >= m_imported_bodies.size()) return false;
    m_imported_bodies[body_index].visible = visible;
    return true;
}

bool Renderer::toggle_imported_body_visible(std::size_t body_index)
{
    if (body_index >= m_imported_bodies.size()) return false;
    m_imported_bodies[body_index].visible = !m_imported_bodies[body_index].visible;
    return true;
}

void Renderer::draw_overlay_triangles(const std::vector<float> &verts) const
{
    if (verts.empty()) return;

    glUniform4f(m_uColorMul, 1.0f, 1.0f, 1.0f, 1.0f);
    glUniform4f(m_uColorAdd, 0.0f, 0.0f, 0.0f, 0.0f);
    glBindVertexArray(m_overlay_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_overlay_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
                 verts.data(),
                 GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(verts.size() / 6));
    glBindVertexArray(0);
    glLineWidth(1.5f);
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

float Renderer::effective_sketch_grid_spacing() const
{
    if (!m_sketch_active) {
        return m_sketch_grid_spacing;
    }

    // Metric step ladder (meters): 1000, 500, 250, 100, 50, 25, 10, 5, 2.5, 1 mm.
    static const std::array<float, 10> kSteps = {
        1.0f, 0.5f, 0.25f, 0.1f, 0.05f, 0.025f, 0.01f, 0.005f, 0.0025f, 0.001f
    };

    float ndc_z_a = 1.0f;
    float ndc_z_b = 1.0f;
    bool ok_a = false;
    bool ok_b = false;
    const glm::vec2 a = project_to_screen(m_sketch_plane.origin, ndc_z_a, ok_a);
    const glm::vec2 b = project_to_screen(m_sketch_plane.origin + (m_sketch_plane.x_axis * 1.0f), ndc_z_b, ok_b);
    if (!ok_a || !ok_b) {
        return m_sketch_grid_spacing;
    }

    const float pixels_per_meter = glm::length(b - a);
    if (pixels_per_meter < 1e-3f) {
        return m_sketch_grid_spacing;
    }

    // Keep a readable cell size while popping through fixed metric thresholds.
    const float min_px = 24.0f;
    const float max_px = 80.0f;
    const float target_px = 40.0f;

    float best = kSteps.front();
    float best_score = std::numeric_limits<float>::infinity();
    for (float step_m : kSteps) {
        const float px = step_m * pixels_per_meter;
        if (px < min_px || px > max_px) {
            continue;
        }
        const float score = std::fabs(px - target_px);
        if (score < best_score) {
            best_score = score;
            best = step_m;
        }
    }

    if (best_score != std::numeric_limits<float>::infinity()) {
        return best;
    }

    // If no step falls in range, prefer the first step still visible at this zoom.
    float fallback = kSteps.back();
    for (float step_m : kSteps) {
        if ((step_m * pixels_per_meter) >= min_px) {
            fallback = step_m;
            break;
        }
    }
    return fallback;
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

bool Renderer::world_to_screen(const glm::vec3 &world, glm::vec2 &out_screen) const
{
    float ndc_z = 1.0f;
    bool ok = false;
    out_screen = project_to_screen(world, ndc_z, ok);
    return ok;
}

bool Renderer::sketch_preview_rectangle_local(SketchRectangle &out_rect) const
{
    if (!m_sketch_entity_in_progress || m_sketch_tool != SketchTool::Rectangle) return false;
    out_rect.min_corner = glm::min(m_sketch_anchor, m_sketch_preview_local);
    out_rect.max_corner = glm::max(m_sketch_anchor, m_sketch_preview_local);
    return true;
}

bool Renderer::sketch_local_to_screen(const glm::vec2 &local, glm::vec2 &out_screen) const
{
    const glm::vec3 world = sketch_world_from_local(local);
    return world_to_screen(world, out_screen);
}

void Renderer::set_sketch_preview_rectangle_dims(float width_m, float height_m)
{
    if (!m_sketch_entity_in_progress || m_sketch_tool != SketchTool::Rectangle) return;

    // Determine direction from anchor to current preview to preserve quadrant
    glm::vec2 dir = glm::sign(m_sketch_preview_local - m_sketch_anchor);
    if (dir.x == 0.0f) dir.x = 1.0f;
    if (dir.y == 0.0f) dir.y = 1.0f;

    // Set preview local so that anchor + (dir * size) = preview_local
    m_sketch_preview_local = m_sketch_anchor + glm::vec2(dir.x * width_m, dir.y * height_m);
    m_sketch_preview_world = sketch_world_from_local(m_sketch_preview_local);
}

bool Renderer::commit_sketch_preview()
{
    if (!m_sketch_entity_in_progress) {
        return false;
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

    return false;
}

void Renderer::set_sketch_preview_local(const glm::vec2 &local)
{
    m_sketch_preview_local = local;
    m_sketch_entity_in_progress = true;
}

glm::vec2 Renderer::sketch_anchor_local() const
{
    return m_sketch_anchor;
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

glm::vec3 Renderer::selection_center() const
{
    if (!m_selected_hit.valid()) {
        return glm::vec3(0.0f);
    }

    if (m_selected_hit.type == PrimitiveType::Face) {
        if (m_selected_hit.index >= 0) {
            return cube_face_center(m_selected_hit.index);
        }
        return m_selected_hit.world_pos;
    }

    if (m_selected_hit.type == PrimitiveType::Edge) {
        const int idx = m_selected_hit.index;
        const int edge_count = static_cast<int>(sizeof(CUBE_EDGES) / sizeof(CUBE_EDGES[0]));
        if (idx >= 0 && idx < edge_count) {
            return (CUBE_EDGES[idx].a + CUBE_EDGES[idx].b) * 0.5f;
        }
        return m_selected_hit.world_pos;
    }

    return m_selected_hit.world_pos;
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
    const float grid_spacing = effective_sketch_grid_spacing();

    if (m_sketch_snap_to_grid) {
        snapped.x = std::round(snapped.x / grid_spacing) * grid_spacing;
        snapped.y = std::round(snapped.y / grid_spacing) * grid_spacing;
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
