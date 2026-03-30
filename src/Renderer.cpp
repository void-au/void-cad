#include "Renderer.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtc/constants.hpp>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <future>
#include <limits>
#include <cmath>
#include <functional>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include "loaders/StepLoader.h"
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
out vec3 vWorldPos;
void main()
{
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
    vWorldPos = aPos;
}
)glsl";

static const char *FRAG_BODY = R"glsl(
precision mediump float;
in  vec3 vColor;
in  vec3 vWorldPos;
uniform vec4 uColorMul;
uniform vec4 uColorAdd;
uniform int  uLit;
uniform vec3 uLightDir;
uniform float uAmbient;
out vec4 FragColor;
void main()
{
    vec3 base = vColor;
    if (uLit != 0) {
        vec3 n = normalize(cross(dFdx(vWorldPos), dFdy(vWorldPos)));
        n = gl_FrontFacing ? n : -n;
        float ndl = max(dot(n, normalize(-uLightDir)), 0.0);
        float lit = uAmbient + (1.0 - uAmbient) * ndl;
        base *= lit;
    }
    FragColor = vec4(base, 1.0) * uColorMul + uColorAdd;
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

// XYZ axis gizmo — compact neon triad with lit cone arrowheads.
static std::vector<float> make_axis_line_verts()
{
    constexpr float len = 0.2f;
    constexpr float cone_len = 0.025f;

    const glm::vec3 col_x(1.0f, 0.15f, 0.65f); // neon pink
    const glm::vec3 col_y(0.25f, 1.0f, 0.35f); // neon green
    const glm::vec3 col_z(0.20f, 0.95f, 1.0f); // neon cyan

    std::vector<float> verts;
    verts.reserve(static_cast<std::size_t>(3 * 2 * 6));

    auto push_line = [&](const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c) {
        verts.push_back(a.x); verts.push_back(a.y); verts.push_back(a.z);
        verts.push_back(c.r); verts.push_back(c.g); verts.push_back(c.b);
        verts.push_back(b.x); verts.push_back(b.y); verts.push_back(b.z);
        verts.push_back(c.r); verts.push_back(c.g); verts.push_back(c.b);
    };

    auto add_axis_line = [&](const glm::vec3 &axis, const glm::vec3 &color) {
        const glm::vec3 base_center = axis * (len - cone_len);
        push_line(glm::vec3(0.0f), base_center, color);
    };

    add_axis_line(glm::vec3(1.0f, 0.0f, 0.0f), col_x);
    add_axis_line(glm::vec3(0.0f, 1.0f, 0.0f), col_y);
    add_axis_line(glm::vec3(0.0f, 0.0f, 1.0f), col_z);

    return verts;
}

static std::vector<float> make_axis_cone_verts()
{
    constexpr float len = 0.2f;
    constexpr float cone_len = 0.025f;
    constexpr float cone_radius = 0.007f;
    constexpr int cone_segs = 16;

    const glm::vec3 col_x(1.0f, 0.15f, 0.65f); // neon pink
    const glm::vec3 col_y(0.25f, 1.0f, 0.35f); // neon green
    const glm::vec3 col_z(0.20f, 0.95f, 1.0f); // neon cyan

    std::vector<float> verts;
    verts.reserve(static_cast<std::size_t>(3 * cone_segs * 6 * 6));

    auto push_vertex = [&](const glm::vec3 &p, const glm::vec3 &c) {
        verts.push_back(p.x); verts.push_back(p.y); verts.push_back(p.z);
        verts.push_back(c.r); verts.push_back(c.g); verts.push_back(c.b);
    };

    auto push_tri = [&](const glm::vec3 &a, const glm::vec3 &b, const glm::vec3 &c, const glm::vec3 &color) {
        push_vertex(a, color);
        push_vertex(b, color);
        push_vertex(c, color);
    };

    auto add_axis_cone = [&](const glm::vec3 &axis,
                             const glm::vec3 &u,
                             const glm::vec3 &v,
                             const glm::vec3 &color) {
        const glm::vec3 tip = axis * len;
        const glm::vec3 base_center = axis * (len - cone_len);

        for (int i = 0; i < cone_segs; ++i) {
            const float a0 = (static_cast<float>(i) / static_cast<float>(cone_segs)) * glm::two_pi<float>();
            const float a1 = (static_cast<float>(i + 1) / static_cast<float>(cone_segs)) * glm::two_pi<float>();
            const glm::vec3 p0 = base_center + (u * std::cos(a0) + v * std::sin(a0)) * cone_radius;
            const glm::vec3 p1 = base_center + (u * std::cos(a1) + v * std::sin(a1)) * cone_radius;

            // Side wall.
            push_tri(tip, p0, p1, color);
            // Base cap.
            push_tri(base_center, p1, p0, color);
        }
    };

    add_axis_cone(glm::vec3(1.0f, 0.0f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f),
                  glm::vec3(0.0f, 0.0f, 1.0f),
                  col_x);
    add_axis_cone(glm::vec3(0.0f, 1.0f, 0.0f),
                  glm::vec3(1.0f, 0.0f, 0.0f),
                  glm::vec3(0.0f, 0.0f, 1.0f),
                  col_y);
    add_axis_cone(glm::vec3(0.0f, 0.0f, 1.0f),
                  glm::vec3(1.0f, 0.0f, 0.0f),
                  glm::vec3(0.0f, 1.0f, 0.0f),
                  col_z);

    return verts;
}

static const std::vector<float> AXIS_LINE_VERTS = make_axis_line_verts();
static const std::vector<float> AXIS_CONE_VERTS = make_axis_cone_verts();

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

static void append_colored_vertex(std::vector<float> &verts,
                                  const glm::vec3 &position,
                                  const glm::vec3 &color)
{
    verts.push_back(position.x);
    verts.push_back(position.y);
    verts.push_back(position.z);
    verts.push_back(color.r);
    verts.push_back(color.g);
    verts.push_back(color.b);
}

static glm::vec3 transform_position(const glm::mat4 &transform, const glm::vec3 &local)
{
    const glm::vec4 p = transform * glm::vec4(local, 1.0f);
    return glm::vec3(p.x, p.y, p.z);
}

static void append_debug_cylinder_mesh(Renderer::PreparedImportBody &body,
                                       const glm::mat4 &transform,
                                       float radius,
                                       float half_height,
                                       int radial_segments,
                                       const glm::vec3 &line_color,
                                       const glm::vec3 &solid_color)
{
    const int segs = std::max(12, radial_segments);
    const float y0 = -half_height;
    const float y1 = half_height;

    auto local_ring_point = [&](float angle, float y) {
        return glm::vec3(std::cos(angle) * radius, y, std::sin(angle) * radius);
    };

    for (int i = 0; i < segs; ++i) {
        const float a0 = (static_cast<float>(i) / static_cast<float>(segs)) * glm::two_pi<float>();
        const float a1 = (static_cast<float>(i + 1) / static_cast<float>(segs)) * glm::two_pi<float>();

        const glm::vec3 p00 = transform_position(transform, local_ring_point(a0, y0));
        const glm::vec3 p10 = transform_position(transform, local_ring_point(a1, y0));
        const glm::vec3 p01 = transform_position(transform, local_ring_point(a0, y1));
        const glm::vec3 p11 = transform_position(transform, local_ring_point(a1, y1));

        append_colored_vertex(body.solid_verts, p00, solid_color);
        append_colored_vertex(body.solid_verts, p10, solid_color);
        append_colored_vertex(body.solid_verts, p11, solid_color);
        append_colored_vertex(body.solid_verts, p00, solid_color);
        append_colored_vertex(body.solid_verts, p11, solid_color);
        append_colored_vertex(body.solid_verts, p01, solid_color);

        const glm::vec3 c_top = transform_position(transform, glm::vec3(0.0f, y1, 0.0f));
        append_colored_vertex(body.solid_verts, c_top, solid_color);
        append_colored_vertex(body.solid_verts, p11, solid_color);
        append_colored_vertex(body.solid_verts, p01, solid_color);

        const glm::vec3 c_bot = transform_position(transform, glm::vec3(0.0f, y0, 0.0f));
        append_colored_vertex(body.solid_verts, c_bot, solid_color);
        append_colored_vertex(body.solid_verts, p00, solid_color);
        append_colored_vertex(body.solid_verts, p10, solid_color);

        append_colored_vertex(body.line_verts, p00, line_color);
        append_colored_vertex(body.line_verts, p10, line_color);
        append_colored_vertex(body.line_verts, p01, line_color);
        append_colored_vertex(body.line_verts, p11, line_color);
    }

    // A STEP-style closed cylindrical side face typically has a single seam edge.
    const float seam_angle = 0.0f;
    const glm::vec3 b0 = transform_position(transform, local_ring_point(seam_angle, y0));
    const glm::vec3 b1 = transform_position(transform, local_ring_point(seam_angle, y1));
    append_colored_vertex(body.line_verts, b0, line_color);
    append_colored_vertex(body.line_verts, b1, line_color);
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
    for (std::size_t i = 1; i < t.size(); ++i) {
        if (t[i] < '0' || t[i] > '9') {
            return false;
        }
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

static void log_step_header_metadata(const std::string &raw)
{
    const std::size_t header_start = raw.find("HEADER;");
    if (header_start == std::string::npos) {
        return;
    }

    const std::size_t data_start = raw.find("DATA;", header_start);
    if (data_start == std::string::npos || data_start <= header_start) {
        return;
    }

    const std::string header_block = raw.substr(header_start, data_start - header_start);

    std::string statement;
    statement.reserve(256);
    for (char ch : header_block) {
        statement.push_back(ch);
        if (ch != ';') {
            continue;
        }

        std::string trimmed = trim_copy(statement);
        statement.clear();
        if (trimmed.empty()) {
            continue;
        }

        const std::string upper = uppercase_copy(trimmed);
        if (upper.find("FILE_DESCRIPTION") != std::string::npos ||
            upper.find("FILE_NAME") != std::string::npos ||
            upper.find("FILE_SCHEMA") != std::string::npos) {
            std::printf("[STEP][HEADER] %s\n", trimmed.c_str());
        }
    }
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

static void upload_geometry_vao(GLuint &vao,
                                GLuint &vbo,
                                GLsizei &vertex_count,
                                const std::vector<float> &verts,
                                GLenum usage)
{
    if (verts.empty()) {
        vao = 0;
        vbo = 0;
        vertex_count = 0;
        return;
    }

    glGenVertexArrays(1, &vao);
    glGenBuffers(1, &vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(verts.size() * sizeof(float)),
                 verts.data(),
                 usage);

    const GLsizei stride = 6 * sizeof(float);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<void*>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, stride,
                          reinterpret_cast<void*>(3 * sizeof(float)));
    glEnableVertexAttribArray(1);

    glBindVertexArray(0);
    vertex_count = static_cast<GLsizei>(verts.size() / 6);
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
    m_uLit      = glGetUniformLocation(m_program, "uLit");
    m_uLightDir = glGetUniformLocation(m_program, "uLightDir");
    m_uAmbient  = glGetUniformLocation(m_program, "uAmbient");

    upload_line_vao(m_axis_vao, m_axis_vbo,
                    AXIS_LINE_VERTS.data(),
                    static_cast<GLsizeiptr>(AXIS_LINE_VERTS.size() * sizeof(float)));
    upload_line_vao(m_axis_cone_vao, m_axis_cone_vbo,
                    AXIS_CONE_VERTS.data(),
                    static_cast<GLsizeiptr>(AXIS_CONE_VERTS.size() * sizeof(float)));
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

void Renderer::set_light_ambient(float ambient)
{
    m_light_ambient = glm::clamp(ambient, 0.0f, 1.0f);
}

void Renderer::set_light_direction(const glm::vec3 &dir)
{
    const float len2 = glm::dot(dir, dir);
    if (len2 <= 1e-8f) {
        return;
    }
    m_light_dir = glm::normalize(dir);
}

void Renderer::rotate_light(float yaw_degrees, float pitch_degrees)
{
    glm::vec3 dir = m_light_dir;

    if (std::fabs(yaw_degrees) > 1e-4f) {
        const glm::mat4 yaw_rot = glm::rotate(glm::mat4(1.0f), glm::radians(yaw_degrees), glm::vec3(0.0f, 1.0f, 0.0f));
        dir = glm::vec3(yaw_rot * glm::vec4(dir, 0.0f));
    }

    if (std::fabs(pitch_degrees) > 1e-4f) {
        glm::vec3 right = glm::cross(glm::vec3(0.0f, 1.0f, 0.0f), dir);
        const float right_len2 = glm::dot(right, right);
        if (right_len2 <= 1e-8f) {
            right = glm::vec3(1.0f, 0.0f, 0.0f);
        } else {
            right = glm::normalize(right);
        }
        const glm::mat4 pitch_rot = glm::rotate(glm::mat4(1.0f), glm::radians(pitch_degrees), right);
        dir = glm::vec3(pitch_rot * glm::vec4(dir, 0.0f));
    }

    set_light_direction(dir);
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
    glUniform1i(m_uLit, 0);
    glUniform3f(m_uLightDir, m_light_dir.x, m_light_dir.y, m_light_dir.z);
    glUniform1f(m_uAmbient, m_light_ambient);

    if (m_has_imported_model) {
        glUniform4f(m_uColorMul, scene_dim, scene_dim, scene_dim, 1.0f);
        glUniform4f(m_uColorAdd, 0.0f, 0.0f, 0.0f, 0.0f);
        if (!m_wireframe) {
            glUniform1i(m_uLit, m_lighting_enabled ? 1 : 0);
            glEnable(GL_POLYGON_OFFSET_FILL);
            glPolygonOffset(1.0f, 1.0f);
            for (std::size_t i = 0; i < m_imported_body_render_data.size() && i < m_imported_bodies.size(); ++i) {
                if (!m_imported_bodies[i].visible) continue;
                const auto &render_data = m_imported_body_render_data[i];
                if (render_data.solid_vao == 0 || render_data.solid_vertex_count <= 0) continue;
                glBindVertexArray(render_data.solid_vao);
                glDrawArrays(GL_TRIANGLES, 0, render_data.solid_vertex_count);
            }
            glDisable(GL_POLYGON_OFFSET_FILL);
            glUniform1i(m_uLit, 0);
        }
        for (std::size_t i = 0; i < m_imported_body_render_data.size() && i < m_imported_bodies.size(); ++i) {
            if (!m_imported_bodies[i].visible) continue;
            const auto &render_data = m_imported_body_render_data[i];
            if (render_data.line_vao == 0 || render_data.line_vertex_count <= 0) continue;
            glLineWidth(1.5f);
            glBindVertexArray(render_data.line_vao);
            glDrawArrays(GL_LINES, 0, render_data.line_vertex_count);
        }
        glBindVertexArray(0);
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

    // Draw axis gizmo last so it stays visible.
    // Keep axis cones at constant brightness for orientation clarity.
    glUniform1i(m_uLit, 0);
    glBindVertexArray(m_axis_cone_vao);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(AXIS_CONE_VERTS.size() / 6));

    glBindVertexArray(m_axis_vao);
    glDrawArrays(GL_LINES, 0, static_cast<GLsizei>(AXIS_LINE_VERTS.size() / 6));

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
    if (vals.size() < 2) return false;
    const float z = (vals.size() >= 3) ? vals[2] : 0.0f;
    const glm::vec3 v(vals[0], vals[1], z);
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

struct EllipseGeom {
    glm::vec3 center;
    glm::vec3 x_axis;  // major-axis direction
    glm::vec3 y_axis;  // minor-axis direction
    float major_radius = 0.0f;
    float minor_radius = 0.0f;
};

struct BSplineGeom {
    int degree = 0;
    std::vector<glm::vec3> control_points;
    std::vector<float> knots; // expanded knot vector with multiplicities
    std::vector<float> weights; // optional, same size as control_points
};

struct BSplineSurfaceRaw {
    int degree_u = 0;
    int degree_v = 0;
    std::vector<std::vector<int>> control_point_ids;
    std::vector<int> multiplicities_u;
    std::vector<int> multiplicities_v;
    std::vector<float> knots_u;
    std::vector<float> knots_v;
    std::vector<std::vector<float>> weights;
};

struct BSplineSurfaceGeom {
    int degree_u = 0;
    int degree_v = 0;
    std::vector<std::vector<glm::vec3>> control_points;
    std::vector<float> knots_u;
    std::vector<float> knots_v;
    std::vector<std::vector<float>> weights;
};

static int find_bspline_span(float u, int degree, int n, const std::vector<float> &knots)
{
    if (n < degree || knots.size() < static_cast<std::size_t>(n + degree + 2)) {
        return -1;
    }

    if (u <= knots[static_cast<std::size_t>(degree)]) {
        return degree;
    }
    if (u >= knots[static_cast<std::size_t>(n + 1)] - 1e-7f) {
        return n;
    }

    int low = degree;
    int high = n + 1;
    int mid = (low + high) / 2;
    while (u < knots[static_cast<std::size_t>(mid)] ||
           u >= knots[static_cast<std::size_t>(mid + 1)]) {
        if (u < knots[static_cast<std::size_t>(mid)]) {
            high = mid;
        } else {
            low = mid;
        }
        mid = (low + high) / 2;
    }
    return mid;
}

static bool compute_bspline_basis(int span,
                                  float u,
                                  int degree,
                                  const std::vector<float> &knots,
                                  std::vector<float> &out_basis)
{
    if (degree < 0 || span < degree) {
        return false;
    }
    if (knots.size() < static_cast<std::size_t>(span + degree + 2)) {
        return false;
    }

    out_basis.assign(static_cast<std::size_t>(degree + 1), 0.0f);
    out_basis[0] = 1.0f;

    std::vector<float> left(static_cast<std::size_t>(degree + 1), 0.0f);
    std::vector<float> right(static_cast<std::size_t>(degree + 1), 0.0f);

    for (int j = 1; j <= degree; ++j) {
        left[static_cast<std::size_t>(j)] = u - knots[static_cast<std::size_t>(span + 1 - j)];
        right[static_cast<std::size_t>(j)] = knots[static_cast<std::size_t>(span + j)] - u;
        float saved = 0.0f;
        for (int r = 0; r < j; ++r) {
            const float denom = right[static_cast<std::size_t>(r + 1)] + left[static_cast<std::size_t>(j - r)];
            const float term = (std::fabs(denom) > 1e-8f)
                ? (out_basis[static_cast<std::size_t>(r)] / denom)
                : 0.0f;
            out_basis[static_cast<std::size_t>(r)] = saved + right[static_cast<std::size_t>(r + 1)] * term;
            saved = left[static_cast<std::size_t>(j - r)] * term;
        }
        out_basis[static_cast<std::size_t>(j)] = saved;
    }

    return true;
}

static bool evaluate_bspline_surface(const BSplineSurfaceGeom &sg, float u, float v, glm::vec3 &out)
{
    if (sg.control_points.empty() || sg.control_points.front().empty()) {
        return false;
    }

    const int nu = static_cast<int>(sg.control_points.size());
    const int nv = static_cast<int>(sg.control_points.front().size());
    for (const auto &row : sg.control_points) {
        if (static_cast<int>(row.size()) != nv) {
            return false;
        }
    }

    const int pu = sg.degree_u;
    const int pv = sg.degree_v;
    const int n_u = nu - 1;
    const int n_v = nv - 1;
    if (pu < 1 || pv < 1 || n_u < pu || n_v < pv) {
        return false;
    }
    if (sg.knots_u.size() != static_cast<std::size_t>(n_u + pu + 2) ||
        sg.knots_v.size() != static_cast<std::size_t>(n_v + pv + 2)) {
        return false;
    }

    const float u_min = sg.knots_u[static_cast<std::size_t>(pu)];
    const float u_max = sg.knots_u[static_cast<std::size_t>(n_u + 1)];
    const float v_min = sg.knots_v[static_cast<std::size_t>(pv)];
    const float v_max = sg.knots_v[static_cast<std::size_t>(n_v + 1)];
    if (!(u_min < u_max) || !(v_min < v_max)) {
        return false;
    }

    const float uu = std::clamp(u, u_min, u_max);
    const float vv = std::clamp(v, v_min, v_max);

    const int span_u = find_bspline_span(uu, pu, n_u, sg.knots_u);
    const int span_v = find_bspline_span(vv, pv, n_v, sg.knots_v);
    if (span_u < 0 || span_v < 0) {
        return false;
    }

    std::vector<float> basis_u;
    std::vector<float> basis_v;
    if (!compute_bspline_basis(span_u, uu, pu, sg.knots_u, basis_u)) {
        return false;
    }
    if (!compute_bspline_basis(span_v, vv, pv, sg.knots_v, basis_v)) {
        return false;
    }

    const bool rational = !sg.weights.empty()
        && sg.weights.size() == sg.control_points.size()
        && sg.weights.front().size() == sg.control_points.front().size();

    if (!rational) {
        glm::vec3 sum(0.0f);
        for (int l = 0; l <= pv; ++l) {
            const int j = span_v - pv + l;
            if (j < 0 || j > n_v) continue;
            for (int k = 0; k <= pu; ++k) {
                const int i = span_u - pu + k;
                if (i < 0 || i > n_u) continue;
                const float coeff = basis_u[static_cast<std::size_t>(k)] * basis_v[static_cast<std::size_t>(l)];
                sum += sg.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] * coeff;
            }
        }
        out = sum;
        return true;
    }

    glm::vec4 hsum(0.0f);
    for (int l = 0; l <= pv; ++l) {
        const int j = span_v - pv + l;
        if (j < 0 || j > n_v) continue;
        for (int k = 0; k <= pu; ++k) {
            const int i = span_u - pu + k;
            if (i < 0 || i > n_u) continue;
            const float w = std::max(
                sg.weights[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                1e-8f);
            const float coeff = basis_u[static_cast<std::size_t>(k)] * basis_v[static_cast<std::size_t>(l)] * w;
            const glm::vec3 &cp = sg.control_points[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)];
            hsum += glm::vec4(cp * coeff, coeff);
        }
    }

    if (std::fabs(hsum.w) <= 1e-8f) {
        return false;
    }
    out = glm::vec3(hsum) / hsum.w;
    return true;
}

static void tessellate_bspline_surface(const BSplineSurfaceGeom &sg,
                                       std::vector<std::array<glm::vec3, 3>> &out_tris)
{
    out_tris.clear();
    if (sg.control_points.empty() || sg.control_points.front().empty()) {
        return;
    }

    const int nu = static_cast<int>(sg.control_points.size());
    const int nv = static_cast<int>(sg.control_points.front().size());
    const int n_u = nu - 1;
    const int n_v = nv - 1;
    if (n_u < sg.degree_u || n_v < sg.degree_v) {
        return;
    }
    if (sg.knots_u.size() != static_cast<std::size_t>(n_u + sg.degree_u + 2) ||
        sg.knots_v.size() != static_cast<std::size_t>(n_v + sg.degree_v + 2)) {
        return;
    }

    const float u_min = sg.knots_u[static_cast<std::size_t>(sg.degree_u)];
    const float u_max = sg.knots_u[static_cast<std::size_t>(n_u + 1)];
    const float v_min = sg.knots_v[static_cast<std::size_t>(sg.degree_v)];
    const float v_max = sg.knots_v[static_cast<std::size_t>(n_v + 1)];
    if (!(u_min < u_max) || !(v_min < v_max)) {
        return;
    }

    const int samples_u = std::clamp(std::max(nu * 10, sg.degree_u * 12), 16, 220);
    const int samples_v = std::clamp(std::max(nv * 10, sg.degree_v * 10), 8, 160);

    const std::size_t row_stride = static_cast<std::size_t>(samples_v + 1);
    std::vector<glm::vec3> grid;
    grid.reserve(static_cast<std::size_t>(samples_u + 1) * row_stride);

    for (int i = 0; i <= samples_u; ++i) {
        const float tu = static_cast<float>(i) / static_cast<float>(samples_u);
        const float u = u_min + tu * (u_max - u_min);
        for (int j = 0; j <= samples_v; ++j) {
            const float tv = static_cast<float>(j) / static_cast<float>(samples_v);
            const float v = v_min + tv * (v_max - v_min);
            glm::vec3 p;
            if (!evaluate_bspline_surface(sg, u, v, p)) {
                out_tris.clear();
                return;
            }
            grid.push_back(p);
        }
    }

    auto at = [&](int iu, int iv) -> const glm::vec3 & {
        return grid[static_cast<std::size_t>(iu) * row_stride + static_cast<std::size_t>(iv)];
    };

    const float eps2 = 1e-14f;
    for (int i = 0; i < samples_u; ++i) {
        for (int j = 0; j < samples_v; ++j) {
            const glm::vec3 &p00 = at(i, j);
            const glm::vec3 &p10 = at(i + 1, j);
            const glm::vec3 &p11 = at(i + 1, j + 1);
            const glm::vec3 &p01 = at(i, j + 1);

            const glm::vec3 n0 = glm::cross(p10 - p00, p11 - p00);
            if (glm::dot(n0, n0) > eps2) {
                out_tris.push_back({p00, p10, p11});
            }

            const glm::vec3 n1 = glm::cross(p11 - p00, p01 - p00);
            if (glm::dot(n1, n1) > eps2) {
                out_tris.push_back({p00, p11, p01});
            }
        }
    }
}

// Tessellate a circular arc into line segments.
// same_sense=true → CCW sweep from start_pt to end_pt (STEP convention).
static void tessellate_arc(
    const CircleGeom &cg,
    const glm::vec3  &start_pt,
    const glm::vec3  &end_pt,
    bool              same_sense,
    std::vector<std::pair<glm::vec3, glm::vec3>> &out_segs)
{
    constexpr int kSegsPerRev = 72;
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

// Tessellate an ellipse arc into line segments using endpoint parameter angles.
static void tessellate_ellipse_arc(
    const EllipseGeom &eg,
    const glm::vec3   &start_pt,
    const glm::vec3   &end_pt,
    bool               same_sense,
    std::vector<std::pair<glm::vec3, glm::vec3>> &out_segs)
{
    constexpr int kSegsPerRev = 96;
    const bool full_ellipse = glm::length(start_pt - end_pt)
                              < 1e-4f * std::max(eg.major_radius, 1e-6f);

    const auto ellipse_theta = [&](const glm::vec3 &p) {
        const glm::vec3 rel = p - eg.center;
        const float ex = glm::dot(rel, eg.x_axis) / std::max(eg.major_radius, 1e-8f);
        const float ey = glm::dot(rel, eg.y_axis) / std::max(eg.minor_radius, 1e-8f);
        return std::atan2(ey, ex);
    };

    const float theta0 = ellipse_theta(start_pt);
    float sweep = 0.0f;
    if (full_ellipse) {
        sweep = glm::two_pi<float>();
    } else {
        const float theta1 = ellipse_theta(end_pt);
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

    glm::vec3 prev = eg.center
                   + eg.x_axis * (eg.major_radius * std::cos(theta0))
                   + eg.y_axis * (eg.minor_radius * std::sin(theta0));
    for (int i = 1; i <= n; ++i) {
        const float th = theta0 + dth * static_cast<float>(i);
        const glm::vec3 next = eg.center
                             + eg.x_axis * (eg.major_radius * std::cos(th))
                             + eg.y_axis * (eg.minor_radius * std::sin(th));
        out_segs.emplace_back(prev, next);
        prev = next;
    }
}

static bool evaluate_bspline(const BSplineGeom &bg, float u, glm::vec3 &out)
{
    const int p = bg.degree;
    const int n = static_cast<int>(bg.control_points.size()) - 1;
    if (p < 1 || n < p) return false;
    if (bg.knots.size() != static_cast<std::size_t>(n + p + 2)) return false;

    const std::vector<float> &U = bg.knots;
    const float u_min = U[static_cast<std::size_t>(p)];
    const float u_max = U[static_cast<std::size_t>(n + 1)];
    if (!(u_min < u_max)) return false;

    const float uu = std::clamp(u, u_min, u_max);

    int k = p;
    if (uu >= u_max - 1e-7f) {
        k = n;
    } else {
        for (int i = p; i <= n; ++i) {
            if (uu >= U[static_cast<std::size_t>(i)] && uu < U[static_cast<std::size_t>(i + 1)]) {
                k = i;
                break;
            }
        }
    }

    const bool rational = (bg.weights.size() == bg.control_points.size());
    if (!rational) {
        std::vector<glm::vec3> d(static_cast<std::size_t>(p + 1));
        for (int j = 0; j <= p; ++j) {
            d[static_cast<std::size_t>(j)] = bg.control_points[static_cast<std::size_t>(k - p + j)];
        }

        for (int r = 1; r <= p; ++r) {
            for (int j = p; j >= r; --j) {
                const int i = k - p + j;
                const float u0 = U[static_cast<std::size_t>(i)];
                const float u1 = U[static_cast<std::size_t>(i + p + 1 - r)];
                const float denom = u1 - u0;
                const float a = (std::fabs(denom) > 1e-8f) ? ((uu - u0) / denom) : 0.0f;
                d[static_cast<std::size_t>(j)] = (1.0f - a) * d[static_cast<std::size_t>(j - 1)] + a * d[static_cast<std::size_t>(j)];
            }
        }

        out = d[static_cast<std::size_t>(p)];
        return true;
    }

    std::vector<glm::vec4> d(static_cast<std::size_t>(p + 1));
    for (int j = 0; j <= p; ++j) {
        const std::size_t idx = static_cast<std::size_t>(k - p + j);
        const float w = std::max(bg.weights[idx], 1e-8f);
        const glm::vec3 &cp = bg.control_points[idx];
        d[static_cast<std::size_t>(j)] = glm::vec4(cp * w, w);
    }

    for (int r = 1; r <= p; ++r) {
        for (int j = p; j >= r; --j) {
            const int i = k - p + j;
            const float u0 = U[static_cast<std::size_t>(i)];
            const float u1 = U[static_cast<std::size_t>(i + p + 1 - r)];
            const float denom = u1 - u0;
            const float a = (std::fabs(denom) > 1e-8f) ? ((uu - u0) / denom) : 0.0f;
            d[static_cast<std::size_t>(j)] = (1.0f - a) * d[static_cast<std::size_t>(j - 1)] + a * d[static_cast<std::size_t>(j)];
        }
    }

    const glm::vec4 hp = d[static_cast<std::size_t>(p)];
    if (std::fabs(hp.w) <= 1e-8f) return false;
    out = glm::vec3(hp) / hp.w;
    return true;
}

static void tessellate_bspline_edge(
    const BSplineGeom &bg,
    const glm::vec3 &start_pt,
    const glm::vec3 &end_pt,
    bool same_sense,
    std::vector<std::pair<glm::vec3, glm::vec3>> &out_segs)
{
    const int n = static_cast<int>(bg.control_points.size()) - 1;
    if (n < 1 || bg.degree < 1) return;
    if (bg.knots.size() != static_cast<std::size_t>(n + bg.degree + 2)) return;

    const float u_min = bg.knots[static_cast<std::size_t>(bg.degree)];
    const float u_max = bg.knots[static_cast<std::size_t>(n + 1)];
    if (!(u_min < u_max)) return;

    const int samples = std::clamp(static_cast<int>(bg.control_points.size()) * 6, 24, 220);
    std::vector<glm::vec3> pts;
    pts.reserve(static_cast<std::size_t>(samples + 1));
    for (int i = 0; i <= samples; ++i) {
        const float t = static_cast<float>(i) / static_cast<float>(samples);
        const float u = u_min + t * (u_max - u_min);
        glm::vec3 p;
        if (evaluate_bspline(bg, u, p)) {
            pts.push_back(p);
        }
    }
    if (pts.size() < 2) return;

    auto nearest_index = [&](const glm::vec3 &p) {
        std::size_t best = 0;
        float best_d2 = std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < pts.size(); ++i) {
            const glm::vec3 d = pts[i] - p;
            const float d2 = glm::dot(d, d);
            if (d2 < best_d2) {
                best_d2 = d2;
                best = i;
            }
        }
        return best;
    };

    std::size_t ia = nearest_index(start_pt);
    std::size_t ib = nearest_index(end_pt);

    if (ia == ib && pts.size() >= 2) {
        if (ia + 1 < pts.size()) {
            ib = ia + 1;
        } else {
            ia = ia - 1;
        }
    }

    std::vector<glm::vec3> path;
    if (ia <= ib) {
        path.insert(path.end(), pts.begin() + static_cast<long>(ia), pts.begin() + static_cast<long>(ib + 1));
    } else {
        path.insert(path.end(), pts.begin() + static_cast<long>(ib), pts.begin() + static_cast<long>(ia + 1));
        std::reverse(path.begin(), path.end());
    }

    if (!same_sense) {
        std::reverse(path.begin(), path.end());
    }

    if (path.size() < 2) return;
    path.front() = start_pt;
    path.back() = end_pt;
    for (std::size_t i = 0; i + 1 < path.size(); ++i) {
        out_segs.emplace_back(path[i], path[i + 1]);
    }
}

bool Renderer::prepare_step_file_import(const std::string &path,
                                        PreparedImport &out_prepared,
                                        std::string &error_message,
                                        std::atomic<float> *progress) const
{
    error_message.clear();
    out_prepared.bodies.clear();
    if (progress) progress->store(0.01f);

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

    std::printf("[STEP] Loading: %s\n", path.c_str());
    log_step_header_metadata(raw);

    const std::vector<ParsedStepEntity> entities = parse_step_entities(raw);
    if (entities.empty()) {
        error_message = "No STEP entities found.";
        return false;
    }
    if (progress) progress->store(0.12f);

    // ---------- entity map ----------
    std::unordered_map<int, std::string> entity_map;
    entity_map.reserve(entities.size() * 2);
    for (const auto &e : entities) {
        entity_map[e.id] = e.body;
    }

    auto extract_type_name = [](const std::string &body) -> std::string {
        const std::size_t p = body.find('(');
        if (p == std::string::npos) {
            return "";
        }
        return uppercase_copy(trim_copy(body.substr(0, p)));
    };

    std::unordered_map<std::string, int> type_counts;
    type_counts.reserve(64);
    for (const auto &[id, body] : entity_map) {
        (void)id;
        const std::string type = extract_type_name(body);
        if (!type.empty()) {
            type_counts[type] += 1;
        }
    }

    auto count_type = [&](const char *type_name) -> int {
        const auto it = type_counts.find(type_name);
        return it == type_counts.end() ? 0 : it->second;
    };

    auto body_has_keyword = [](const std::string &body, const char *keyword) -> bool {
        return uppercase_copy(body).find(keyword) != std::string::npos;
    };

    auto count_keyword = [&](const char *keyword) -> int {
        int count = 0;
        for (const auto &[id, body] : entity_map) {
            (void)id;
            if (body_has_keyword(body, keyword)) {
                count += 1;
            }
        }
        return count;
    };

    std::printf("[STEP] Entity count: %zu\n", entities.size());
    std::printf("[STEP] Types: CARTESIAN_POINT=%d, DIRECTION=%d, AXIS2_PLACEMENT_3D=%d, LOCAL_PLACEMENT=%d\n",
                count_type("CARTESIAN_POINT"),
                count_type("DIRECTION"),
                count_type("AXIS2_PLACEMENT_3D"),
                count_type("LOCAL_PLACEMENT"));
    std::printf("[STEP] Types: MANIFOLD_SOLID_BREP=%d, BREP_WITH_VOIDS=%d, ADVANCED_FACE=%d, EDGE_CURVE=%d\n",
                count_type("MANIFOLD_SOLID_BREP"),
                count_type("BREP_WITH_VOIDS"),
                count_type("ADVANCED_FACE"),
                count_type("EDGE_CURVE"));
    std::printf("[STEP] Types: SHAPE_REPRESENTATION=%d, SHAPE_REPRESENTATION_RELATIONSHIP=%d, REPRESENTATION_MAP=%d, MAPPED_ITEM=%d\n",
                count_type("SHAPE_REPRESENTATION"),
                count_type("SHAPE_REPRESENTATION_RELATIONSHIP"),
                count_type("REPRESENTATION_MAP"),
                count_type("MAPPED_ITEM"));
    std::printf("[STEP] Types: ITEM_DEFINED_TRANSFORMATION=%d, SHAPE_REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION=%d\n",
                count_keyword("ITEM_DEFINED_TRANSFORMATION"),
                count_keyword("REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION"));
    if (count_type("LOCAL_PLACEMENT") > 0 || count_type("MAPPED_ITEM") > 0) {
        std::printf("[STEP][INFO] Placement/mapping entities detected. Assembly transforms are currently only partially handled and may affect body positions.\n");
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

    struct LocalPlacementRaw {
        int parent_local_id = -1;
        int relative_axis_id = -1;
    };

    struct TransformOperatorRaw {
        int origin_id = -1;
        int axis1_id = -1;
        int axis2_id = -1;
        float scale = 1.0f;
    };

    auto transform_point = [](const glm::mat4 &matrix, const glm::vec3 &point) -> glm::vec3 {
        const glm::vec4 transformed = matrix * glm::vec4(point, 1.0f);
        return glm::vec3(transformed.x, transformed.y, transformed.z);
    };

    auto extract_type = [&](int id) -> std::string {
        const auto it = entity_map.find(id);
        if (it == entity_map.end()) return "";
        return get_type(it->second);
    };

    std::unordered_map<int, std::vector<int>> reverse_refs;
    reverse_refs.reserve(entity_map.size() * 2);
    for (const auto &[id, body] : entity_map) {
        for (int ref_id : get_refs(body)) {
            reverse_refs[ref_id].push_back(id);
        }
    }

    // ---------- pass 1: geometry primitives ----------
    std::unordered_map<int, glm::vec3> points;
    std::unordered_map<int, int>       vertex_to_point;
    std::unordered_map<int, glm::vec3> directions;

    struct AxisPlacement { int origin_id = -1, z_id = -1, x_id = -1; };
    std::unordered_map<int, AxisPlacement> axis_placements;
    std::unordered_map<int, LocalPlacementRaw> local_placements;
    std::unordered_map<int, TransformOperatorRaw> transform_operators;
    struct ItemDefinedTransformRaw {
        int from_axis_id = -1;
        int to_axis_id = -1;
    };
    std::unordered_map<int, ItemDefinedTransformRaw> item_defined_transforms;

    struct CircleRaw { int placement_id = -1; float radius = 0.0f; };
    std::unordered_map<int, CircleRaw> circle_raws;
    struct EllipseRaw { int placement_id = -1; float major_radius = 0.0f; float minor_radius = 0.0f; };
    std::unordered_map<int, EllipseRaw> ellipse_raws;
    struct BSplineRaw {
        int degree = 0;
        std::vector<int> control_point_ids;
        std::vector<int> multiplicities;
        std::vector<float> knots;
        std::vector<float> weights;
    };
    std::unordered_map<int, BSplineRaw> bspline_raws;
    std::unordered_map<int, BSplineGeom> bspline_geoms;
    std::unordered_map<int, BSplineSurfaceRaw> bspline_surface_raws;
    std::unordered_map<int, BSplineSurfaceGeom> bspline_surface_geoms;
    struct CylindricalSurfaceRaw { int placement_id = -1; float radius = 0.0f; };
    std::unordered_map<int, CylindricalSurfaceRaw> cylindrical_surface_raws;

    auto parse_id_list = [](const std::string &token) {
        std::vector<int> ids;
        std::string t = trim_copy(token);
        const std::size_t op = t.find('(');
        const std::size_t cl = t.rfind(')');
        if (op == std::string::npos || cl == std::string::npos || cl <= op) return ids;
        std::string inner = t.substr(op + 1, cl - op - 1);
        std::size_t s = 0;
        while (s < inner.size()) {
            const std::size_t cm = inner.find(',', s);
            const std::string part = trim_copy(cm == std::string::npos ? inner.substr(s) : inner.substr(s, cm - s));
            int id = -1;
            if (parse_step_id_ref(part, id)) ids.push_back(id);
            if (cm == std::string::npos) break;
            s = cm + 1;
        }
        return ids;
    };

    auto parse_int_list = [](const std::string &token) {
        std::vector<int> vals;
        std::string t = trim_copy(token);
        const std::size_t op = t.find('(');
        const std::size_t cl = t.rfind(')');
        if (op == std::string::npos || cl == std::string::npos || cl <= op) return vals;
        std::string inner = t.substr(op + 1, cl - op - 1);
        std::size_t s = 0;
        while (s < inner.size()) {
            const std::size_t cm = inner.find(',', s);
            const std::string part = trim_copy(cm == std::string::npos ? inner.substr(s) : inner.substr(s, cm - s));
            if (!part.empty()) {
                try { vals.push_back(std::stoi(part)); } catch (...) {}
            }
            if (cm == std::string::npos) break;
            s = cm + 1;
        }
        return vals;
    };

    auto parse_float_list = [](const std::string &token) {
        std::vector<float> vals;
        std::string t = trim_copy(token);
        const std::size_t op = t.find('(');
        const std::size_t cl = t.rfind(')');
        if (op == std::string::npos || cl == std::string::npos || cl <= op) return vals;
        std::string inner = t.substr(op + 1, cl - op - 1);
        inner = trim_copy(inner);
        // Some STEP complex entities nest one more pair of parentheses, e.g.
        // RATIONAL_B_SPLINE_CURVE((w0,w1,...)).
        while (inner.size() >= 2 && inner.front() == '(' && inner.back() == ')') {
            inner = trim_copy(inner.substr(1, inner.size() - 2));
        }
        std::size_t s = 0;
        while (s < inner.size()) {
            const std::size_t cm = inner.find(',', s);
            const std::string part = trim_copy(cm == std::string::npos ? inner.substr(s) : inner.substr(s, cm - s));
            if (!part.empty()) {
                try { vals.push_back(std::stof(part)); } catch (...) {}
            }
            if (cm == std::string::npos) break;
            s = cm + 1;
        }
        return vals;
    };

    auto parse_id_grid = [](const std::string &token) {
        std::vector<std::vector<int>> rows;
        std::string t = trim_copy(token);
        const std::size_t op = t.find('(');
        const std::size_t cl = t.rfind(')');
        if (op == std::string::npos || cl == std::string::npos || cl <= op) return rows;

        const std::string inner = t.substr(op + 1, cl - op - 1);
        int depth = 0;
        std::string cur;
        for (char c : inner) {
            if (c == '(') {
                if (depth++ == 0) {
                    cur.clear();
                    continue;
                }
            } else if (c == ')') {
                if (--depth == 0) {
                    std::vector<int> ids;
                    std::size_t s = 0;
                    while (s < cur.size()) {
                        const std::size_t cm = cur.find(',', s);
                        const std::string part = trim_copy(cm == std::string::npos ? cur.substr(s) : cur.substr(s, cm - s));
                        int id = -1;
                        if (parse_step_id_ref(part, id)) ids.push_back(id);
                        if (cm == std::string::npos) break;
                        s = cm + 1;
                    }
                    if (!ids.empty()) rows.push_back(std::move(ids));
                    continue;
                }
            }

            if (depth >= 1) {
                cur.push_back(c);
            }
        }

        return rows;
    };

    auto parse_float_grid = [](const std::string &token) {
        std::vector<std::vector<float>> rows;
        std::string t = trim_copy(token);
        const std::size_t op = t.find('(');
        const std::size_t cl = t.rfind(')');
        if (op == std::string::npos || cl == std::string::npos || cl <= op) return rows;

        const std::string inner = t.substr(op + 1, cl - op - 1);
        int depth = 0;
        std::string cur;
        for (char c : inner) {
            if (c == '(') {
                if (depth++ == 0) {
                    cur.clear();
                    continue;
                }
            } else if (c == ')') {
                if (--depth == 0) {
                    std::vector<float> vals;
                    std::size_t s = 0;
                    while (s < cur.size()) {
                        const std::size_t cm = cur.find(',', s);
                        const std::string part = trim_copy(cm == std::string::npos ? cur.substr(s) : cur.substr(s, cm - s));
                        if (!part.empty()) {
                            try { vals.push_back(std::stof(part)); } catch (...) {}
                        }
                        if (cm == std::string::npos) break;
                        s = cm + 1;
                    }
                    if (!vals.empty()) rows.push_back(std::move(vals));
                    continue;
                }
            }

            if (depth >= 1) {
                cur.push_back(c);
            }
        }

        return rows;
    };

    auto extract_keyword_entity = [&](const std::string &body,
                                      const char *keyword,
                                      std::string &out_entity) -> bool {
        const std::string upper = uppercase_copy(body);
        const std::string key = std::string(keyword) + "(";
        const std::size_t pos = upper.find(key);
        if (pos == std::string::npos) return false;

        const std::size_t open = pos + std::string(keyword).size();
        if (open >= body.size() || body[open] != '(') return false;

        int depth = 0;
        std::size_t close = std::string::npos;
        for (std::size_t i = open; i < body.size(); ++i) {
            if (body[i] == '(') ++depth;
            else if (body[i] == ')') {
                --depth;
                if (depth == 0) {
                    close = i;
                    break;
                }
            }
        }
        if (close == std::string::npos || close < pos) return false;
        out_entity = body.substr(pos, close - pos + 1);
        return true;
    };

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
        } else if (type == "LOCAL_PLACEMENT") {
            const auto args = split_step_args(body);
            if (args.size() >= 3) {
                int parent_id = -1;
                int axis_id = -1;
                if (!args[1].empty() && args[1][0] == '#') {
                    parse_step_id_ref(args[1], parent_id);
                }
                if (parse_step_id_ref(args[2], axis_id)) {
                    local_placements[id] = {parent_id, axis_id};
                }
            }
        } else if (type == "CARTESIAN_TRANSFORMATION_OPERATOR_3D" ||
                   type == "CARTESIAN_TRANSFORMATION_OPERATOR_3D_NON_UNIFORM") {
            const auto args = split_step_args(body);
            if (args.size() >= 4) {
                int origin_id = -1;
                int axis1_id = -1;
                int axis2_id = -1;
                float scale = 1.0f;
                parse_step_id_ref(args[3], origin_id);
                if (args.size() >= 2 && !args[1].empty() && args[1][0] == '#') {
                    parse_step_id_ref(args[1], axis1_id);
                }
                if (args.size() >= 3 && !args[2].empty() && args[2][0] == '#') {
                    parse_step_id_ref(args[2], axis2_id);
                }
                if (args.size() >= 5) {
                    const std::string scale_token = trim_copy(args[4]);
                    if (!scale_token.empty() && scale_token != "$" && scale_token != "*") {
                        try {
                            scale = std::stof(scale_token);
                        } catch (...) {
                            scale = 1.0f;
                        }
                    }
                }
                if (origin_id >= 0) {
                    transform_operators[id] = {origin_id, axis1_id, axis2_id, scale};
                }
            }
        } else if (type == "ITEM_DEFINED_TRANSFORMATION" || body_has_keyword(body, "ITEM_DEFINED_TRANSFORMATION")) {
            const auto refs = get_refs(body);
            if (refs.size() >= 2) {
                item_defined_transforms[id] = {refs[0], refs[1]};
            }
        } else if (type == "CIRCLE" || body_has_keyword(body, "CIRCLE")) {
            std::string entity = body;
            if (type != "CIRCLE" && !extract_keyword_entity(body, "CIRCLE", entity)) continue;
            const auto args = split_step_args(entity);
            if (args.size() >= 3) {
                int pid = -1;
                if (parse_step_id_ref(args[1], pid)) {
                    try {
                        const float r = std::stof(trim_copy(args[2]));
                        if (r > 0.0f) circle_raws[id] = {pid, r};
                    } catch (...) {}
                }
            }
        } else if (type == "ELLIPSE" || body_has_keyword(body, "ELLIPSE")) {
            std::string entity = body;
            if (type != "ELLIPSE" && !extract_keyword_entity(body, "ELLIPSE", entity)) continue;
            const auto args = split_step_args(entity);
            if (args.size() >= 4) {
                int pid = -1;
                if (parse_step_id_ref(args[1], pid)) {
                    try {
                        const float major_r = std::stof(trim_copy(args[2]));
                        const float minor_r = std::stof(trim_copy(args[3]));
                        if (major_r > 0.0f && minor_r > 0.0f) {
                            ellipse_raws[id] = {pid, major_r, minor_r};
                        }
                    } catch (...) {}
                }
            }
        } else if (type == "B_SPLINE_CURVE_WITH_KNOTS" || body_has_keyword(body, "B_SPLINE_CURVE_WITH_KNOTS")) {
            std::string entity = body;
            if (type != "B_SPLINE_CURVE_WITH_KNOTS" && !extract_keyword_entity(body, "B_SPLINE_CURVE_WITH_KNOTS", entity)) continue;
            const auto args = split_step_args(entity);
            if (args.size() >= 8) {
                int degree = 0;
                try { degree = std::stoi(trim_copy(args[1])); } catch (...) { degree = 0; }
                const std::vector<int> ctrl_ids = parse_id_list(args[2]);
                const std::vector<int> mults = parse_int_list(args[6]);
                const std::vector<float> knots = parse_float_list(args[7]);

                if (degree > 0 && !ctrl_ids.empty() && !mults.empty() && mults.size() == knots.size()) {
                    bspline_raws[id] = {degree, ctrl_ids, mults, knots, {}};
                }
            } else if (args.size() >= 3) {
                // Complex-entity subtype form: (knot_multiplicities, knots, knot_spec)
                auto it = bspline_raws.find(id);
                if (it == bspline_raws.end()) {
                    // If this complex entity matched knot subtype first, bootstrap
                    // degree/control-points from the sibling B_SPLINE_CURVE part.
                    std::string base_entity;
                    if (extract_keyword_entity(body, "B_SPLINE_CURVE", base_entity)) {
                        const auto base_args = split_step_args(base_entity);
                        if (base_args.size() >= 2) {
                            int degree = 0;
                            try { degree = std::stoi(trim_copy(base_args[0])); } catch (...) { degree = 0; }
                            const std::vector<int> ctrl_ids = parse_id_list(base_args[1]);
                            if (degree > 0 && !ctrl_ids.empty()) {
                                bspline_raws[id] = {degree, ctrl_ids, {}, {}, {}};
                                it = bspline_raws.find(id);
                                if (id == 51608 || id == 51616) {
                                    std::printf("[STEP] Bootstrap B-spline #%d: degree=%d, cpts=%zu\n", id, degree, ctrl_ids.size());
                                }
                            }
                        }
                    }
                }
                if (it != bspline_raws.end()) {
                    const std::vector<int> mults = parse_int_list(args[0]);
                    const std::vector<float> knots = parse_float_list(args[1]);
                    if (id == 51608 || id == 51616) {
                        std::printf("[STEP] Parse knots #%d: mults=%zu, knots=%zu\n", id, mults.size(), knots.size());
                    }
                    if (!mults.empty() && mults.size() == knots.size()) {
                        it->second.multiplicities = mults;
                        it->second.knots = knots;
                    }
                }
            }
        } else if (type == "B_SPLINE_CURVE" || body_has_keyword(body, "B_SPLINE_CURVE")) {
            // Some files carry non-knot B-splines in complex entities.
            // Build a raw spline and synthesize an open-uniform knot vector later.
            std::string entity = body;
            if (type != "B_SPLINE_CURVE" && !extract_keyword_entity(body, "B_SPLINE_CURVE", entity)) continue;
            const auto args = split_step_args(entity);
            if (args.size() >= 3) {
                int degree = 0;
                try { degree = std::stoi(trim_copy(args[1])); } catch (...) { degree = 0; }
                const std::vector<int> ctrl_ids = parse_id_list(args[2]);
                if (degree > 0 && !ctrl_ids.empty()) {
                    // Preserve explicit knot-bearing variant if already parsed.
                    if (bspline_raws.find(id) == bspline_raws.end()) {
                        bspline_raws[id] = {degree, ctrl_ids, {}, {}, {}};
                    }
                }
            }
        } else if (type == "BEZIER_CURVE" || body_has_keyword(body, "BEZIER_CURVE")) {
            std::string entity = body;
            if (type != "BEZIER_CURVE" && !extract_keyword_entity(body, "BEZIER_CURVE", entity)) continue;
            const auto args = split_step_args(entity);
            if (args.size() >= 2) {
                const std::vector<int> ctrl_ids = parse_id_list(args[1]);
                if (ctrl_ids.size() >= 2) {
                    const int degree = static_cast<int>(ctrl_ids.size()) - 1;
                    if (bspline_raws.find(id) == bspline_raws.end()) {
                        bspline_raws[id] = {degree, ctrl_ids, {}, {}, {}};
                    }
                }
            }
        } else if (type == "B_SPLINE_SURFACE_WITH_KNOTS" || body_has_keyword(body, "B_SPLINE_SURFACE_WITH_KNOTS")) {
            std::string entity = body;
            if (type != "B_SPLINE_SURFACE_WITH_KNOTS" &&
                !extract_keyword_entity(body, "B_SPLINE_SURFACE_WITH_KNOTS", entity)) {
                continue;
            }

            const auto args = split_step_args(entity);
            if (args.size() >= 12) {
                int degree_u = 0;
                int degree_v = 0;
                try { degree_u = std::stoi(trim_copy(args[1])); } catch (...) { degree_u = 0; }
                try { degree_v = std::stoi(trim_copy(args[2])); } catch (...) { degree_v = 0; }

                const auto ctrl_grid = parse_id_grid(args[3]);
                const auto mult_u = parse_int_list(args[8]);
                const auto mult_v = parse_int_list(args[9]);
                const auto knots_u = parse_float_list(args[10]);
                const auto knots_v = parse_float_list(args[11]);

                if (degree_u > 0 && degree_v > 0 &&
                    !ctrl_grid.empty() &&
                    !mult_u.empty() && !mult_v.empty() &&
                    mult_u.size() == knots_u.size() &&
                    mult_v.size() == knots_v.size()) {
                    bspline_surface_raws[id] = {
                        degree_u,
                        degree_v,
                        ctrl_grid,
                        mult_u,
                        mult_v,
                        knots_u,
                        knots_v,
                        {}
                    };
                }
            }
        } else if (type == "CYLINDRICAL_SURFACE" || body_has_keyword(body, "CYLINDRICAL_SURFACE")) {
            std::string entity = body;
            if (type != "CYLINDRICAL_SURFACE" && !extract_keyword_entity(body, "CYLINDRICAL_SURFACE", entity)) continue;
            const auto args = split_step_args(entity);
            if (args.size() >= 3) {
                int pid = -1;
                if (parse_step_id_ref(args[1], pid)) {
                    try {
                        const float r = std::stof(trim_copy(args[2]));
                        if (r > 0.0f) cylindrical_surface_raws[id] = {pid, r};
                    } catch (...) {}
                }
            }
        }

        if (body_has_keyword(body, "RATIONAL_B_SPLINE_CURVE")) {
            std::string entity;
            if (extract_keyword_entity(body, "RATIONAL_B_SPLINE_CURVE", entity)) {
                const auto args = split_step_args(entity);
                if (!args.empty()) {
                    const std::vector<float> weights = parse_float_list(args.back());
                    auto it = bspline_raws.find(id);
                    if (id == 51608 || id == 51616) {
                        std::printf("[STEP] Weights for B-spline #%d: found=%s, weights=%zu\n", id, 
                            (it != bspline_raws.end() ? "yes" : "no"), weights.size());
                    }
                    if (it != bspline_raws.end() && !weights.empty()) {
                        it->second.weights = weights;
                    }
                }
            }
        }

        if (body_has_keyword(body, "RATIONAL_B_SPLINE_SURFACE")) {
            std::string entity;
            if (extract_keyword_entity(body, "RATIONAL_B_SPLINE_SURFACE", entity)) {
                const auto args = split_step_args(entity);
                if (!args.empty()) {
                    const auto weights = parse_float_grid(args.front());
                    auto it = bspline_surface_raws.find(id);
                    if (it != bspline_surface_raws.end() && !weights.empty()) {
                        it->second.weights = weights;
                    }
                }
            }
        }
    }
    if (points.empty()) {
        error_message = "No CARTESIAN_POINT entities found.";
        return false;
    }

    for (const auto &[sid, sr] : bspline_raws) {
        BSplineGeom bg;
        bg.degree = sr.degree;
        bg.control_points.reserve(sr.control_point_ids.size());
        for (int pid : sr.control_point_ids) {
            const auto pit = points.find(pid);
            if (pit != points.end()) {
                bg.control_points.push_back(pit->second);
            }
        }
        if (bg.control_points.size() < 2 || bg.degree <= 0) {
            continue;
        }

        if (!sr.knots.empty() && !sr.multiplicities.empty()) {
            for (std::size_t i = 0; i < sr.knots.size(); ++i) {
                const int m = (i < sr.multiplicities.size()) ? std::max(0, sr.multiplicities[i]) : 0;
                for (int k = 0; k < m; ++k) {
                    bg.knots.push_back(sr.knots[i]);
                }
            }
        } else {
            // Open-uniform synthesized knots for non-knot B-spline/Bezier variants.
            const int p = bg.degree;
            const int n = static_cast<int>(bg.control_points.size()) - 1;
            if (n >= p) {
                const int knot_count = n + p + 2;
                bg.knots.reserve(static_cast<std::size_t>(knot_count));
                for (int i = 0; i < knot_count; ++i) {
                    if (i <= p) {
                        bg.knots.push_back(0.0f);
                    } else if (i >= (n + 1)) {
                        bg.knots.push_back(1.0f);
                    } else {
                        const float t = static_cast<float>(i - p) / static_cast<float>(n - p + 1);
                        bg.knots.push_back(t);
                    }
                }
            }
        }

        if (sr.weights.size() == bg.control_points.size()) {
            bg.weights = sr.weights;
        }

        if (!bg.knots.empty()) {
            bspline_geoms[sid] = std::move(bg);
        } else {
            // Debug: check which entities have empty knots
            if (sid == 51608 || sid == 51616 || sid == 65293 || sid == 65919) {
                std::printf("[STEP] Spline #%d: control_pts=%zu, degree=%d, knots=%zu, mults=%zu\n",
                    sid, sr.control_point_ids.size(), sr.degree, sr.knots.size(), sr.multiplicities.size());
            }
        }
    }

    for (const auto &[sid, sr] : bspline_surface_raws) {
        if (sr.control_point_ids.empty() || sr.degree_u <= 0 || sr.degree_v <= 0) {
            continue;
        }

        BSplineSurfaceGeom sg;
        sg.degree_u = sr.degree_u;
        sg.degree_v = sr.degree_v;
        sg.control_points.reserve(sr.control_point_ids.size());

        bool valid_grid = true;
        std::size_t expected_cols = 0;
        for (const auto &row_ids : sr.control_point_ids) {
            if (row_ids.empty()) {
                valid_grid = false;
                break;
            }

            if (expected_cols == 0) {
                expected_cols = row_ids.size();
            } else if (row_ids.size() != expected_cols) {
                valid_grid = false;
                break;
            }

            std::vector<glm::vec3> row_points;
            row_points.reserve(row_ids.size());
            for (int pid : row_ids) {
                const auto pit = points.find(pid);
                if (pit == points.end()) {
                    valid_grid = false;
                    break;
                }
                row_points.push_back(pit->second);
            }
            if (!valid_grid) break;
            sg.control_points.push_back(std::move(row_points));
        }

        if (!valid_grid || sg.control_points.size() < 2 || expected_cols < 2) {
            continue;
        }

        for (std::size_t i = 0; i < sr.knots_u.size(); ++i) {
            const int m = (i < sr.multiplicities_u.size()) ? std::max(0, sr.multiplicities_u[i]) : 0;
            for (int k = 0; k < m; ++k) {
                sg.knots_u.push_back(sr.knots_u[i]);
            }
        }
        for (std::size_t i = 0; i < sr.knots_v.size(); ++i) {
            const int m = (i < sr.multiplicities_v.size()) ? std::max(0, sr.multiplicities_v[i]) : 0;
            for (int k = 0; k < m; ++k) {
                sg.knots_v.push_back(sr.knots_v[i]);
            }
        }

        const int n_u = static_cast<int>(sg.control_points.size()) - 1;
        const int n_v = static_cast<int>(expected_cols) - 1;
        if (n_u < sg.degree_u || n_v < sg.degree_v) {
            continue;
        }
        if (sg.knots_u.size() != static_cast<std::size_t>(n_u + sg.degree_u + 2) ||
            sg.knots_v.size() != static_cast<std::size_t>(n_v + sg.degree_v + 2)) {
            continue;
        }

        if (!sr.weights.empty()
            && sr.weights.size() == sg.control_points.size()
            && sr.weights.front().size() == expected_cols) {
            bool weights_ok = true;
            for (const auto &wrow : sr.weights) {
                if (wrow.size() != expected_cols) {
                    weights_ok = false;
                    break;
                }
            }
            if (weights_ok) {
                sg.weights = sr.weights;
            }
        }

        bspline_surface_geoms[sid] = std::move(sg);
    }

    auto vertex_pos = [&](int vid, glm::vec3 &out) -> bool {
        const auto vit = vertex_to_point.find(vid);
        if (vit == vertex_to_point.end()) return false;
        const auto pit = points.find(vit->second);
        if (pit == points.end()) return false;
        out = pit->second;
        return true;
    };

    std::unordered_map<int, glm::mat4> axis_matrix_cache;
    std::unordered_map<int, glm::mat4> local_matrix_cache;

    auto make_axis_matrix = [&](int axis_id, glm::mat4 &out_matrix) -> bool {
        const auto cache_it = axis_matrix_cache.find(axis_id);
        if (cache_it != axis_matrix_cache.end()) {
            out_matrix = cache_it->second;
            return true;
        }

        const auto ap_it = axis_placements.find(axis_id);
        if (ap_it == axis_placements.end()) {
            return false;
        }

        const auto &ap = ap_it->second;
        const auto origin_it = points.find(ap.origin_id);
        const auto z_it = directions.find(ap.z_id);
        if (origin_it == points.end() || z_it == directions.end()) {
            return false;
        }

        glm::vec3 z_axis = z_it->second;
        if (glm::length(z_axis) < 1e-8f) {
            return false;
        }
        z_axis = glm::normalize(z_axis);

        glm::vec3 x_axis(1.0f, 0.0f, 0.0f);
        if (ap.x_id >= 0) {
            const auto x_it = directions.find(ap.x_id);
            if (x_it != directions.end()) {
                const glm::vec3 candidate = x_it->second - z_axis * glm::dot(x_it->second, z_axis);
                if (glm::length(candidate) > 1e-8f) {
                    x_axis = glm::normalize(candidate);
                }
            }
        }

        if (glm::length(x_axis) < 1e-8f || std::fabs(glm::dot(x_axis, z_axis)) > 0.999f) {
            const glm::vec3 fallback_up = (std::fabs(z_axis.y) < 0.9f)
                ? glm::vec3(0.0f, 1.0f, 0.0f)
                : glm::vec3(1.0f, 0.0f, 0.0f);
            x_axis = glm::normalize(glm::cross(fallback_up, z_axis));
        }

        glm::vec3 y_axis = glm::cross(z_axis, x_axis);
        if (glm::length(y_axis) < 1e-8f) {
            return false;
        }
        y_axis = glm::normalize(y_axis);
        x_axis = glm::normalize(glm::cross(y_axis, z_axis));

        glm::mat4 matrix(1.0f);
        matrix[0] = glm::vec4(x_axis, 0.0f);
        matrix[1] = glm::vec4(y_axis, 0.0f);
        matrix[2] = glm::vec4(z_axis, 0.0f);
        matrix[3] = glm::vec4(origin_it->second, 1.0f);

        axis_matrix_cache[axis_id] = matrix;
        out_matrix = matrix;
        return true;
    };

    std::function<bool(int, glm::mat4&)> make_local_matrix = [&](int local_id, glm::mat4 &out_matrix) -> bool {
        const auto cache_it = local_matrix_cache.find(local_id);
        if (cache_it != local_matrix_cache.end()) {
            out_matrix = cache_it->second;
            return true;
        }

        const auto lp_it = local_placements.find(local_id);
        if (lp_it == local_placements.end()) {
            return false;
        }

        glm::mat4 relative(1.0f);
        if (!make_axis_matrix(lp_it->second.relative_axis_id, relative)) {
            return false;
        }

        glm::mat4 parent(1.0f);
        if (lp_it->second.parent_local_id >= 0) {
            glm::mat4 resolved_parent(1.0f);
            if (make_local_matrix(lp_it->second.parent_local_id, resolved_parent)) {
                parent = resolved_parent;
            }
        }

        const glm::mat4 world = parent * relative;
        local_matrix_cache[local_id] = world;
        out_matrix = world;
        return true;
    };

    auto make_transform_operator_matrix = [&](int transform_id, glm::mat4 &out_matrix) -> bool {
        const auto to_it = transform_operators.find(transform_id);
        if (to_it == transform_operators.end()) {
            return false;
        }

        const auto &raw_op = to_it->second;
        const auto origin_it = points.find(raw_op.origin_id);
        if (origin_it == points.end()) {
            return false;
        }

        glm::vec3 x_axis(1.0f, 0.0f, 0.0f);
        glm::vec3 y_axis(0.0f, 1.0f, 0.0f);

        if (raw_op.axis1_id >= 0) {
            const auto axis1_it = directions.find(raw_op.axis1_id);
            if (axis1_it != directions.end() && glm::length(axis1_it->second) > 1e-8f) {
                x_axis = glm::normalize(axis1_it->second);
            }
        }

        if (raw_op.axis2_id >= 0) {
            const auto axis2_it = directions.find(raw_op.axis2_id);
            if (axis2_it != directions.end() && glm::length(axis2_it->second) > 1e-8f) {
                y_axis = glm::normalize(axis2_it->second);
            }
        }

        if (std::fabs(glm::dot(x_axis, y_axis)) > 0.999f) {
            const glm::vec3 up = (std::fabs(x_axis.y) < 0.9f)
                ? glm::vec3(0.0f, 1.0f, 0.0f)
                : glm::vec3(1.0f, 0.0f, 0.0f);
            y_axis = glm::normalize(glm::cross(glm::cross(x_axis, up), x_axis));
        }

        glm::vec3 z_axis = glm::cross(x_axis, y_axis);
        if (glm::length(z_axis) < 1e-8f) {
            return false;
        }
        z_axis = glm::normalize(z_axis);
        y_axis = glm::normalize(glm::cross(z_axis, x_axis));

        const float s = (std::fabs(raw_op.scale) > 1e-8f) ? raw_op.scale : 1.0f;

        glm::mat4 matrix(1.0f);
        matrix[0] = glm::vec4(x_axis * s, 0.0f);
        matrix[1] = glm::vec4(y_axis * s, 0.0f);
        matrix[2] = glm::vec4(z_axis * s, 0.0f);
        matrix[3] = glm::vec4(origin_it->second, 1.0f);

        out_matrix = matrix;
        return true;
    };

    auto make_item_defined_transform_matrix = [&](int transform_id, glm::mat4 &out_matrix) -> bool {
        const auto idt_it = item_defined_transforms.find(transform_id);
        if (idt_it == item_defined_transforms.end()) {
            return false;
        }

        glm::mat4 from_matrix(1.0f);
        glm::mat4 to_matrix(1.0f);
        if (!make_axis_matrix(idt_it->second.from_axis_id, from_matrix)) {
            return false;
        }
        if (!make_axis_matrix(idt_it->second.to_axis_id, to_matrix)) {
            return false;
        }

        out_matrix = to_matrix * glm::inverse(from_matrix);
        return true;
    };

    auto resolve_transform_entity = [&](int entity_id, glm::mat4 &out_matrix) -> bool {
        const std::string type = extract_type(entity_id);
        if (type == "AXIS2_PLACEMENT_3D") {
            return make_axis_matrix(entity_id, out_matrix);
        }
        if (type == "LOCAL_PLACEMENT") {
            return make_local_matrix(entity_id, out_matrix);
        }
        if (type == "CARTESIAN_TRANSFORMATION_OPERATOR_3D" ||
            type == "CARTESIAN_TRANSFORMATION_OPERATOR_3D_NON_UNIFORM") {
            return make_transform_operator_matrix(entity_id, out_matrix);
        }
        if (type == "ITEM_DEFINED_TRANSFORMATION") {
            return make_item_defined_transform_matrix(entity_id, out_matrix);
        }
        return false;
    };

    struct RepresentationTransformEdge {
        int target_rep_id = -1;
        glm::mat4 matrix = glm::mat4(1.0f);
    };

    std::unordered_map<int, std::vector<int>> representation_items;
    representation_items.reserve(64);
    for (const auto &[id, body] : entity_map) {
        const std::string type = get_type(body);
        if (type.find("SHAPE_REPRESENTATION") == std::string::npos &&
            !body_has_keyword(body, "SHAPE_REPRESENTATION")) {
            continue;
        }

        const auto refs = get_refs(body);
        if (!refs.empty()) {
            representation_items[id] = refs;
        }
    }

    std::unordered_map<int, std::vector<RepresentationTransformEdge>> representation_edges;
    representation_edges.reserve(64);
    for (const auto &[id, body] : entity_map) {
        if (!body_has_keyword(body, "REPRESENTATION_RELATIONSHIP_WITH_TRANSFORMATION")) {
            continue;
        }

        const auto refs = get_refs(body);
        if (refs.size() < 3) {
            continue;
        }

        const int rep_from = refs[0];
        const int rep_to = refs[1];
        const int transform_id = refs[2];

        glm::mat4 relation_matrix(1.0f);
        if (!resolve_transform_entity(transform_id, relation_matrix)) {
            continue;
        }

        representation_edges[rep_from].push_back({rep_to, relation_matrix});
    }

    std::unordered_map<int, glm::mat4> representation_world_cache;
    std::function<glm::mat4(int)> resolve_representation_world = [&](int rep_id) -> glm::mat4 {
        const auto cache_it = representation_world_cache.find(rep_id);
        if (cache_it != representation_world_cache.end()) {
            return cache_it->second;
        }

        glm::mat4 world(1.0f);
        const auto edge_it = representation_edges.find(rep_id);
        if (edge_it != representation_edges.end() && !edge_it->second.empty()) {
            const RepresentationTransformEdge &edge = edge_it->second.front();
            world = resolve_representation_world(edge.target_rep_id) * edge.matrix;
        }

        representation_world_cache[rep_id] = world;
        return world;
    };

    // ---------- resolve circle geometries ----------
    std::unordered_map<int, CircleGeom> circle_geoms;
    for (const auto &[cid, cr] : circle_raws) {
        glm::vec3 origin(0.0f);
        glm::vec3 z_axis(0.0f, 0.0f, 1.0f);
        glm::vec3 x_axis(1.0f, 0.0f, 0.0f);
        bool ok = false;

        const auto ap_it = axis_placements.find(cr.placement_id);
        if (ap_it != axis_placements.end()) {
            const auto &ap = ap_it->second;
            const auto oit = points.find(ap.origin_id);
            const auto zit = directions.find(ap.z_id);
            if (oit != points.end() && zit != directions.end()) {
                origin = oit->second;
                z_axis = glm::normalize(zit->second);
                if (ap.x_id >= 0) {
                    const auto xit = directions.find(ap.x_id);
                    if (xit != directions.end()) {
                        const glm::vec3 candidate = xit->second - z_axis * glm::dot(xit->second, z_axis);
                        if (glm::length(candidate) > 1e-8f) {
                            x_axis = glm::normalize(candidate);
                            ok = true;
                        }
                    }
                } else {
                    ok = true;
                }
            }
        }

        // Fallback for circles placed with AXIS2_PLACEMENT_2D.
        if (!ok) {
            const auto pit = entity_map.find(cr.placement_id);
            if (pit != entity_map.end()) {
                std::string pbody = pit->second;
                if (get_type(pbody) != "AXIS2_PLACEMENT_2D") {
                    std::string extracted;
                    if (body_has_keyword(pbody, "AXIS2_PLACEMENT_2D") && extract_keyword_entity(pbody, "AXIS2_PLACEMENT_2D", extracted)) {
                        pbody = extracted;
                    }
                }
                if (get_type(pbody) == "AXIS2_PLACEMENT_2D") {
                    const auto args = split_step_args(pbody);
                    if (args.size() >= 2) {
                        int oid = -1;
                        if (parse_step_id_ref(args[1], oid)) {
                            const auto oit = points.find(oid);
                            if (oit != points.end()) {
                                origin = oit->second;
                                if (args.size() >= 3 && !args[2].empty() && args[2][0] == '#') {
                                    int xid = -1;
                                    if (parse_step_id_ref(args[2], xid)) {
                                        const auto xit = directions.find(xid);
                                        if (xit != directions.end() && glm::length(xit->second) > 1e-8f) {
                                            x_axis = glm::normalize(glm::vec3(xit->second.x, xit->second.y, 0.0f));
                                        }
                                    }
                                }
                                ok = true;
                            }
                        }
                    }
                }
            }
        }

        if (!ok) continue;
        if (glm::length(x_axis) < 1e-8f || std::fabs(glm::dot(x_axis, z_axis)) > 0.999f) {
            const glm::vec3 up = (std::fabs(z_axis.y) < 0.9f)
                ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
            x_axis = glm::normalize(glm::cross(up, z_axis));
            if (glm::length(x_axis) < 1e-8f) continue;
        }
        const glm::vec3 y_axis = glm::normalize(glm::cross(z_axis, x_axis));
        circle_geoms[cid] = {origin, x_axis, y_axis, cr.radius};
    }

    std::unordered_map<int, EllipseGeom> ellipse_geoms;
    for (const auto &[eid, er] : ellipse_raws) {
        const auto ap_it = axis_placements.find(er.placement_id);
        if (ap_it == axis_placements.end()) continue;
        const auto &ap = ap_it->second;
        const auto oit = points.find(ap.origin_id);
        const auto zit = directions.find(ap.z_id);
        if (oit == points.end() || zit == directions.end()) continue;

        glm::vec3 z_axis = zit->second;
        if (glm::length(z_axis) < 1e-8f) continue;
        z_axis = glm::normalize(z_axis);

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
        ellipse_geoms[eid] = {oit->second, x_axis, y_axis, er.major_radius, er.minor_radius};
    }

    // Resolve wrapped curve entities (TRIMMED_CURVE / SURFACE_CURVE / etc.)
    // back to analytic primitives used for tessellation.
    std::unordered_map<int, int> curve_to_circle_cache;
    std::unordered_map<int, int> curve_to_ellipse_cache;
    std::unordered_map<int, int> curve_to_bspline_cache;

    std::function<int(int, std::unordered_set<int>&)> resolve_to_circle =
        [&](int curve_id, std::unordered_set<int> &visiting) -> int {
            if (curve_id < 0) return -1;
            const auto cache_it = curve_to_circle_cache.find(curve_id);
            if (cache_it != curve_to_circle_cache.end()) return cache_it->second;
            if (circle_geoms.find(curve_id) != circle_geoms.end()) {
                curve_to_circle_cache[curve_id] = curve_id;
                return curve_id;
            }
            if (!visiting.insert(curve_id).second) {
                return -1;
            }
            int found = -1;
            for (int r : get_body_refs(curve_id)) {
                found = resolve_to_circle(r, visiting);
                if (found >= 0) break;
            }
            visiting.erase(curve_id);
            curve_to_circle_cache[curve_id] = found;
            return found;
        };

    std::function<int(int, std::unordered_set<int>&)> resolve_to_ellipse =
        [&](int curve_id, std::unordered_set<int> &visiting) -> int {
            if (curve_id < 0) return -1;
            const auto cache_it = curve_to_ellipse_cache.find(curve_id);
            if (cache_it != curve_to_ellipse_cache.end()) return cache_it->second;
            if (ellipse_geoms.find(curve_id) != ellipse_geoms.end()) {
                curve_to_ellipse_cache[curve_id] = curve_id;
                return curve_id;
            }
            if (!visiting.insert(curve_id).second) {
                return -1;
            }
            int found = -1;
            for (int r : get_body_refs(curve_id)) {
                found = resolve_to_ellipse(r, visiting);
                if (found >= 0) break;
            }
            visiting.erase(curve_id);
            curve_to_ellipse_cache[curve_id] = found;
            return found;
        };

    std::function<int(int, std::unordered_set<int>&)> resolve_to_bspline =
        [&](int curve_id, std::unordered_set<int> &visiting) -> int {
            if (curve_id < 0) return -1;
            const auto cache_it = curve_to_bspline_cache.find(curve_id);
            if (cache_it != curve_to_bspline_cache.end()) return cache_it->second;
            if (bspline_geoms.find(curve_id) != bspline_geoms.end()) {
                curve_to_bspline_cache[curve_id] = curve_id;
                return curve_id;
            }
            if (!visiting.insert(curve_id).second) {
                return -1;
            }
            int found = -1;
            for (int r : get_body_refs(curve_id)) {
                found = resolve_to_bspline(r, visiting);
                if (found >= 0) break;
            }
            visiting.erase(curve_id);
            curve_to_bspline_cache[curve_id] = found;
            return found;
        };

    // ---------- pass 2: EDGE_CURVE map (with curve ref + sense) ----------
    struct EdgeCurveEntry { int vs = -1, ve = -1, curve_id = -1; bool same_sense = true; };
    std::unordered_map<int, EdgeCurveEntry> edge_curves;
    for (const auto &[id, body] : entity_map) {
        std::string edge_entity = body;
        if (get_type(edge_entity) != "EDGE_CURVE") {
            if (!body_has_keyword(body, "EDGE_CURVE") || !extract_keyword_entity(body, "EDGE_CURVE", edge_entity)) {
                continue;
            }
        }

        int vs = -1;
        int ve = -1;
        int curve_id = -1;
        const auto args = split_step_args(edge_entity);
        if (args.size() >= 4) {
            parse_step_id_ref(args[1], vs);
            parse_step_id_ref(args[2], ve);
            parse_step_id_ref(args[3], curve_id);
        }

        // Fallback for malformed argument extraction.
        if (vs < 0 || ve < 0) {
            const auto refs = get_refs(edge_entity);
            if (refs.size() >= 2) {
                vs = refs[0];
                ve = refs[1];
            }
        }
        if (vs < 0 || ve < 0) continue;

        edge_curves[id] = {vs, ve, curve_id, get_orientation(edge_entity)};
    }

    int arc_circle_hits = 0;
    int arc_ellipse_hits = 0;
    int spline_hits = 0;
    int straight_line_hits = 0;
    int unresolved_curve_fallback_hits = 0;
    std::unordered_map<std::string, int> unresolved_types;
    std::unordered_map<int, int> unresolved_curve_ids;

    std::function<bool(int, std::unordered_set<int>&)> resolve_to_line =
        [&](int curve_id, std::unordered_set<int> &visiting) -> bool {
            if (curve_id < 0) return false;
            const std::string t = get_type_of(curve_id);
            if (t == "LINE") return true;
            if (!visiting.insert(curve_id).second) return false;
            for (int r : get_body_refs(curve_id)) {
                if (resolve_to_line(r, visiting)) {
                    visiting.erase(curve_id);
                    return true;
                }
            }
            visiting.erase(curve_id);
            return false;
        };

    // Tessellate one edge: arc if it has a resolved circle, else straight line.
    auto tessellate_edge_entry = [&](const EdgeCurveEntry &ec,
                                     std::vector<std::pair<glm::vec3,glm::vec3>> &out_segs) {
        glm::vec3 pa, pb;
        if (!vertex_pos(ec.vs, pa) || !vertex_pos(ec.ve, pb)) return;
        if (ec.curve_id >= 0) {
            std::unordered_set<int> visiting;
            const int circle_id = resolve_to_circle(ec.curve_id, visiting);
            if (circle_id >= 0) {
                const auto cit = circle_geoms.find(circle_id);
                if (cit != circle_geoms.end()) {
                    tessellate_arc(cit->second, pa, pb, ec.same_sense, out_segs);
                    arc_circle_hits += 1;
                    return;
                }
            }

            visiting.clear();
            const int ellipse_id = resolve_to_ellipse(ec.curve_id, visiting);
            if (ellipse_id >= 0) {
                const auto eit = ellipse_geoms.find(ellipse_id);
                if (eit != ellipse_geoms.end()) {
                    tessellate_ellipse_arc(eit->second, pa, pb, ec.same_sense, out_segs);
                    arc_ellipse_hits += 1;
                    return;
                }
            }

            visiting.clear();
            const int bspline_id = resolve_to_bspline(ec.curve_id, visiting);
            if (bspline_id >= 0) {
                const auto bit = bspline_geoms.find(bspline_id);
                if (bit != bspline_geoms.end()) {
                    tessellate_bspline_edge(bit->second, pa, pb, ec.same_sense, out_segs);
                    if (!out_segs.empty()) {
                        spline_hits += 1;
                        return;
                    }
                }
            }
        }
        if (ec.curve_id >= 0) {
            std::unordered_set<int> vis_line;
            if (resolve_to_line(ec.curve_id, vis_line)) {
                straight_line_hits += 1;
            } else {
                unresolved_curve_fallback_hits += 1;
                std::string t = get_type_of(ec.curve_id);
                if (t.empty()) t = "<UNKNOWN>";
                unresolved_types[t] += 1;
                unresolved_curve_ids[ec.curve_id] += 1;
            }
        } else {
            straight_line_hits += 1;
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
        glm::mat4 transform = glm::mat4(1.0f);
    };
    std::vector<BodySeed> body_seeds;
    std::unordered_set<int> assigned_faces;

    struct CylindricalSurfaceGeom {
        glm::vec3 origin = glm::vec3(0.0f);
        glm::vec3 axis = glm::vec3(0.0f, 0.0f, 1.0f);
        glm::vec3 x_axis = glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 y_axis = glm::vec3(0.0f, 1.0f, 0.0f);
        float radius = 0.0f;
    };
    std::unordered_map<int, CylindricalSurfaceGeom> cylindrical_surfaces;
    for (const auto &[sid, sr] : cylindrical_surface_raws) {
        const auto ap_it = axis_placements.find(sr.placement_id);
        if (ap_it == axis_placements.end()) continue;
        const auto &ap = ap_it->second;
        const auto oit = points.find(ap.origin_id);
        const auto zit = directions.find(ap.z_id);
        if (oit == points.end() || zit == directions.end()) continue;

        glm::vec3 axis = zit->second;
        if (glm::length(axis) < 1e-8f) continue;
        axis = glm::normalize(axis);

        glm::vec3 x_axis(1.0f, 0.0f, 0.0f);
        if (ap.x_id >= 0) {
            const auto xit = directions.find(ap.x_id);
            if (xit != directions.end()) {
                const glm::vec3 candidate = xit->second - axis * glm::dot(xit->second, axis);
                if (glm::length(candidate) > 1e-8f) {
                    x_axis = glm::normalize(candidate);
                }
            }
        }
        if (glm::length(x_axis) < 1e-8f || std::fabs(glm::dot(x_axis, axis)) > 0.999f) {
            const glm::vec3 up = (std::fabs(axis.y) < 0.9f)
                ? glm::vec3(0.0f, 1.0f, 0.0f)
                : glm::vec3(1.0f, 0.0f, 0.0f);
            x_axis = glm::normalize(glm::cross(up, axis));
        }
        glm::vec3 y_axis = glm::normalize(glm::cross(axis, x_axis));
        x_axis = glm::normalize(glm::cross(y_axis, axis));

        cylindrical_surfaces[sid] = {oit->second, axis, x_axis, y_axis, sr.radius};
    }

    auto find_nearest_local_placement_matrix = [&](int root_id, glm::mat4 &out_matrix) -> bool {
        std::vector<int> queue;
        queue.push_back(root_id);
        std::size_t head = 0;
        std::unordered_set<int> visited;
        visited.insert(root_id);

        while (head < queue.size()) {
            const int current = queue[head++];
            const auto rev_it = reverse_refs.find(current);
            if (rev_it == reverse_refs.end()) {
                continue;
            }
            for (int parent_id : rev_it->second) {
                if (!visited.insert(parent_id).second) {
                    continue;
                }
                if (extract_type(parent_id) == "LOCAL_PLACEMENT") {
                    glm::mat4 matrix(1.0f);
                    if (make_local_matrix(parent_id, matrix)) {
                        out_matrix = matrix;
                        return true;
                    }
                }
                queue.push_back(parent_id);
            }
        }

        return false;
    };

    auto find_representation_transform_for_item = [&](int item_id, glm::mat4 &out_matrix) -> bool {
        std::vector<int> queue;
        queue.push_back(item_id);
        std::size_t head = 0;
        std::unordered_set<int> visited;
        visited.insert(item_id);

        while (head < queue.size()) {
            const int current = queue[head++];
            const auto rev_it = reverse_refs.find(current);
            if (rev_it == reverse_refs.end()) {
                continue;
            }

            for (int parent_id : rev_it->second) {
                if (!visited.insert(parent_id).second) {
                    continue;
                }

                if (representation_items.find(parent_id) != representation_items.end()) {
                    out_matrix = resolve_representation_world(parent_id);
                    return true;
                }

                queue.push_back(parent_id);
            }
        }

        return false;
    };

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

    auto collect_solid_roots_from = [&](int root_id, std::unordered_set<int> &out_solids) {
        std::vector<int> stack{root_id};
        std::unordered_set<int> visited;
        while (!stack.empty()) {
            const int cur = stack.back();
            stack.pop_back();
            if (!visited.insert(cur).second) continue;

            const std::string type = extract_type(cur);
            if (type == "MANIFOLD_SOLID_BREP" || type == "BREP_WITH_VOIDS") {
                out_solids.insert(cur);
                continue;
            }

            for (int r : get_body_refs(cur)) stack.push_back(r);
        }
    };

    std::unordered_map<int, int> representation_map_to_representation;
    for (const auto &[id, body] : entity_map) {
        if (get_type(body) != "REPRESENTATION_MAP") continue;
        const auto refs = get_refs(body);
        if (refs.size() >= 2) {
            representation_map_to_representation[id] = refs[1];
        }
    }

    std::unordered_set<int> solids_with_mapped_instances;
    for (const auto &[id, body] : entity_map) {
        if (get_type(body) != "MAPPED_ITEM") continue;
        const auto refs = get_refs(body);
        if (refs.size() < 2) continue;

        const int rep_map_id = refs[0];
        const auto rep_it = representation_map_to_representation.find(rep_map_id);
        if (rep_it == representation_map_to_representation.end()) continue;

        std::unordered_set<int> solid_roots;
        collect_solid_roots_from(rep_it->second, solid_roots);
        solids_with_mapped_instances.insert(solid_roots.begin(), solid_roots.end());
    }

    for (const auto &[id, body] : entity_map) {
        const std::string type = get_type(body);
        if (type != "MANIFOLD_SOLID_BREP" && type != "BREP_WITH_VOIDS") continue;
        if (solids_with_mapped_instances.find(id) != solids_with_mapped_instances.end()) continue;
        const auto refs = get_refs(body);
        if (refs.empty()) continue;

        std::unordered_set<int> body_faces;
        for (int r : refs) collect_faces_from(r, body_faces);
        if (body_faces.empty()) continue;

        BodySeed seed;
        seed.label = parsed_name_or(body, "Body " + std::to_string(body_seeds.size() + 1));
        seed.face_ids.assign(body_faces.begin(), body_faces.end());
        std::sort(seed.face_ids.begin(), seed.face_ids.end());
        glm::mat4 placement(1.0f);
        if (find_nearest_local_placement_matrix(id, placement)) {
            seed.transform = placement;
        }
        glm::mat4 rep_transform(1.0f);
        if (find_representation_transform_for_item(id, rep_transform)) {
            seed.transform = rep_transform * seed.transform;
        }
        body_seeds.push_back(seed);
        assigned_faces.insert(body_faces.begin(), body_faces.end());
    }

    for (const auto &[id, body] : entity_map) {
        if (get_type(body) != "MAPPED_ITEM") continue;

        const auto refs = get_refs(body);
        if (refs.size() < 2) continue;

        const int rep_map_id = refs[0];
        const int target_transform_id = refs[1];

        const auto rep_it = representation_map_to_representation.find(rep_map_id);
        if (rep_it == representation_map_to_representation.end()) continue;

        glm::mat4 source_matrix(1.0f);
        const auto rep_map_entity_it = entity_map.find(rep_map_id);
        if (rep_map_entity_it != entity_map.end()) {
            const auto map_refs = get_refs(rep_map_entity_it->second);
            if (!map_refs.empty()) {
                glm::mat4 source_candidate(1.0f);
                if (resolve_transform_entity(map_refs[0], source_candidate)) {
                    source_matrix = source_candidate;
                }
            }
        }

        glm::mat4 target_matrix(1.0f);
        if (!resolve_transform_entity(target_transform_id, target_matrix)) {
            continue;
        }
        const glm::mat4 map_transform = target_matrix * glm::inverse(source_matrix);

        std::unordered_set<int> mapped_solid_roots;
        collect_solid_roots_from(rep_it->second, mapped_solid_roots);
        for (int solid_root_id : mapped_solid_roots) {
            std::unordered_set<int> body_faces;
            collect_faces_from(solid_root_id, body_faces);
            if (body_faces.empty()) continue;

            const auto solid_it = entity_map.find(solid_root_id);
            const std::string fallback_name = "Body " + std::to_string(body_seeds.size() + 1);
            const std::string base_name = (solid_it != entity_map.end())
                ? parsed_name_or(solid_it->second, fallback_name)
                : fallback_name;

            BodySeed seed;
            seed.label = base_name + " (inst " + std::to_string(id) + ")";
            seed.face_ids.assign(body_faces.begin(), body_faces.end());
            std::sort(seed.face_ids.begin(), seed.face_ids.end());
            seed.transform = map_transform;
            body_seeds.push_back(std::move(seed));
            assigned_faces.insert(body_faces.begin(), body_faces.end());
        }
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
        std::vector<glm::vec3> outer;
        std::vector<std::vector<glm::vec3>> holes;
        std::vector<std::vector<glm::vec3>> loops;
        int cylindrical_surface_id = -1;
        int bspline_surface_id = -1;
        int source_face_id = -1;
    };
    struct RawBody {
        std::string label;
        glm::mat4 transform = glm::mat4(1.0f);
        std::vector<RawFace> faces;
        std::vector<std::pair<glm::vec3, glm::vec3>> edges;
        std::unordered_set<unsigned long long> edge_dedupe;
        int edge_entity_count = 0;
    };

    auto face_loop_normal = [](const std::vector<glm::vec3> &loop) -> glm::vec3 {
        if (loop.size() < 3) return glm::vec3(0.0f, 0.0f, 1.0f);
        glm::vec3 n(0.0f);
        for (std::size_t i = 0; i < loop.size(); ++i) {
            const glm::vec3 &a = loop[i];
            const glm::vec3 &b = loop[(i + 1) % loop.size()];
            n.x += (a.y - b.y) * (a.z + b.z);
            n.y += (a.z - b.z) * (a.x + b.x);
            n.z += (a.x - b.x) * (a.y + b.y);
        }
        if (glm::length(n) < 1e-8f) return glm::vec3(0.0f, 0.0f, 1.0f);
        return glm::normalize(n);
    };

    auto make_face_basis = [&](const glm::vec3 &normal, glm::vec3 &u, glm::vec3 &v) {
        glm::vec3 n = normal;
        if (glm::length(n) < 1e-8f) n = glm::vec3(0.0f, 0.0f, 1.0f);
        n = glm::normalize(n);
        const glm::vec3 up = (std::fabs(n.z) < 0.9f)
            ? glm::vec3(0.0f, 0.0f, 1.0f)
            : glm::vec3(0.0f, 1.0f, 0.0f);
        u = glm::normalize(glm::cross(up, n));
        if (glm::length(u) < 1e-8f) {
            u = glm::vec3(1.0f, 0.0f, 0.0f);
        }
        v = glm::normalize(glm::cross(n, u));
    };

    auto project_loop_2d = [](const std::vector<glm::vec3> &loop,
                              const glm::vec3 &u,
                              const glm::vec3 &v) {
        std::vector<glm::vec2> out;
        out.reserve(loop.size());
        for (const glm::vec3 &p : loop) {
            out.emplace_back(glm::dot(p, u), glm::dot(p, v));
        }
        return out;
    };

    auto signed_area_2d = [](const std::vector<glm::vec2> &poly) -> float {
        if (poly.size() < 3) return 0.0f;
        float a = 0.0f;
        for (std::size_t i = 0; i < poly.size(); ++i) {
            const glm::vec2 &p = poly[i];
            const glm::vec2 &q = poly[(i + 1) % poly.size()];
            a += p.x * q.y - q.x * p.y;
        }
        return 0.5f * a;
    };

    auto clean_loop = [&](std::vector<glm::vec3> &loop, const glm::vec3 &normal) {
        if (loop.size() < 3) return;
        const float kPosEps = 1e-6f;
        const float kCrossEps = 1e-7f;

        bool changed = true;
        while (changed && loop.size() >= 3) {
            changed = false;
            for (std::size_t i = 0; i < loop.size(); ++i) {
                const std::size_t ip = (i + loop.size() - 1) % loop.size();
                const std::size_t in = (i + 1) % loop.size();
                if (glm::length(loop[i] - loop[ip]) <= kPosEps ||
                    glm::length(loop[i] - loop[in]) <= kPosEps) {
                    loop.erase(loop.begin() + static_cast<long>(i));
                    changed = true;
                    break;
                }

                const glm::vec3 a = loop[i] - loop[ip];
                const glm::vec3 b = loop[in] - loop[i];
                const float turn = glm::dot(glm::cross(a, b), normal);
                if (std::fabs(turn) <= kCrossEps && glm::dot(a, b) > 0.0f) {
                    loop.erase(loop.begin() + static_cast<long>(i));
                    changed = true;
                    break;
                }
            }
        }
    };

    auto point_in_triangle_2d = [](const glm::vec2 &p,
                                   const glm::vec2 &a,
                                   const glm::vec2 &b,
                                   const glm::vec2 &c) -> bool {
        const float eps = 1e-7f;
        const auto cross2 = [](const glm::vec2 &u, const glm::vec2 &w) {
            return u.x * w.y - u.y * w.x;
        };
        const float c1 = cross2(b - a, p - a);
        const float c2 = cross2(c - b, p - b);
        const float c3 = cross2(a - c, p - c);
        const bool has_neg = (c1 < -eps) || (c2 < -eps) || (c3 < -eps);
        const bool has_pos = (c1 > eps) || (c2 > eps) || (c3 > eps);
        return !(has_neg && has_pos);
    };

    auto point_in_polygon_2d = [](const glm::vec2 &p, const std::vector<glm::vec2> &poly) -> bool {
        bool inside = false;
        for (std::size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++) {
            const glm::vec2 &pi = poly[i];
            const glm::vec2 &pj = poly[j];
            const bool intersects = ((pi.y > p.y) != (pj.y > p.y))
                && (p.x < (pj.x - pi.x) * (p.y - pi.y) / ((pj.y - pi.y) + 1e-12f) + pi.x);
            if (intersects) inside = !inside;
        }
        return inside;
    };

    auto segments_intersect_2d = [](const glm::vec2 &a,
                                    const glm::vec2 &b,
                                    const glm::vec2 &c,
                                    const glm::vec2 &d) -> bool {
        const float eps = 1e-7f;
        const auto cross2 = [](const glm::vec2 &u, const glm::vec2 &w) {
            return u.x * w.y - u.y * w.x;
        };
        const auto orient = [&](const glm::vec2 &p, const glm::vec2 &q, const glm::vec2 &r) {
            return cross2(q - p, r - p);
        };
        const auto on_segment = [&](const glm::vec2 &p, const glm::vec2 &q, const glm::vec2 &r) {
            return q.x <= std::max(p.x, r.x) + eps && q.x + eps >= std::min(p.x, r.x)
                && q.y <= std::max(p.y, r.y) + eps && q.y + eps >= std::min(p.y, r.y);
        };

        const float o1 = orient(a, b, c);
        const float o2 = orient(a, b, d);
        const float o3 = orient(c, d, a);
        const float o4 = orient(c, d, b);

        if ((o1 * o2 < -eps) && (o3 * o4 < -eps)) return true;

        if (std::fabs(o1) <= eps && on_segment(a, c, b)) return true;
        if (std::fabs(o2) <= eps && on_segment(a, d, b)) return true;
        if (std::fabs(o3) <= eps && on_segment(c, a, d)) return true;
        if (std::fabs(o4) <= eps && on_segment(c, b, d)) return true;
        return false;
    };

    auto polygon_contains_point_exclusive = [&](const glm::vec2 &p, const std::vector<glm::vec2> &poly) {
        if (!point_in_polygon_2d(p, poly)) return false;
        const float eps = 1e-6f;
        for (std::size_t i = 0; i < poly.size(); ++i) {
            const glm::vec2 &a = poly[i];
            const glm::vec2 &b = poly[(i + 1) % poly.size()];
            const glm::vec2 ab = b - a;
            const glm::vec2 ap = p - a;
            const float cross = ab.x * ap.y - ab.y * ap.x;
            const float dot = glm::dot(ap, ab);
            if (std::fabs(cross) <= eps && dot >= -eps && dot <= glm::dot(ab, ab) + eps) {
                return false;
            }
        }
        return true;
    };

    auto bridge_hole = [&](std::vector<glm::vec3> &outer,
                           const std::vector<glm::vec3> &hole,
                           const glm::vec3 &normal) -> bool {
        if (outer.size() < 3 || hole.size() < 3) return false;

        glm::vec3 u(1.0f), v(0.0f, 1.0f, 0.0f);
        make_face_basis(normal, u, v);
        std::vector<glm::vec2> outer2 = project_loop_2d(outer, u, v);
        std::vector<glm::vec2> hole2 = project_loop_2d(hole, u, v);

        std::size_t hole_idx = 0;
        for (std::size_t i = 1; i < hole2.size(); ++i) {
            if (hole2[i].x > hole2[hole_idx].x + 1e-7f ||
                (std::fabs(hole2[i].x - hole2[hole_idx].x) <= 1e-7f && hole2[i].y < hole2[hole_idx].y)) {
                hole_idx = i;
            }
        }
        const glm::vec2 hp = hole2[hole_idx];

        int best_outer = -1;
        float best_dist2 = std::numeric_limits<float>::infinity();
        for (std::size_t oi = 0; oi < outer2.size(); ++oi) {
            const glm::vec2 op = outer2[oi];
            if (op.x <= hp.x + 1e-7f) continue;

            bool blocked = false;
            for (std::size_t i = 0; i < outer2.size(); ++i) {
                const std::size_t in = (i + 1) % outer2.size();
                if (i == oi || in == oi) continue;
                if (segments_intersect_2d(hp, op, outer2[i], outer2[in])) {
                    blocked = true;
                    break;
                }
            }
            if (blocked) continue;

            for (std::size_t i = 0; i < hole2.size(); ++i) {
                const std::size_t in = (i + 1) % hole2.size();
                if (i == hole_idx || in == hole_idx) continue;
                if (segments_intersect_2d(hp, op, hole2[i], hole2[in])) {
                    blocked = true;
                    break;
                }
            }
            if (blocked) continue;

            const glm::vec2 mid = 0.5f * (hp + op);
            if (!polygon_contains_point_exclusive(mid, outer2)) continue;

            const float d2 = glm::dot(op - hp, op - hp);
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_outer = static_cast<int>(oi);
            }
        }

        if (best_outer < 0) {
            for (std::size_t oi = 0; oi < outer2.size(); ++oi) {
                const glm::vec2 op = outer2[oi];
                const float d2 = glm::dot(op - hp, op - hp);
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_outer = static_cast<int>(oi);
                }
            }
        }
        if (best_outer < 0) return false;

        std::vector<glm::vec3> merged;
        merged.reserve(outer.size() + hole.size() + 2);

        for (int i = 0; i <= best_outer; ++i) merged.push_back(outer[static_cast<std::size_t>(i)]);

        for (std::size_t k = 0; k < hole.size(); ++k) {
            const std::size_t idx = (hole_idx + k) % hole.size();
            merged.push_back(hole[idx]);
        }
        merged.push_back(hole[hole_idx]);

        merged.push_back(outer[static_cast<std::size_t>(best_outer)]);
        for (std::size_t i = static_cast<std::size_t>(best_outer) + 1; i < outer.size(); ++i) {
            merged.push_back(outer[i]);
        }

        outer.swap(merged);
        clean_loop(outer, normal);
        return outer.size() >= 3;
    };

    auto triangulate_simple_polygon = [&](const std::vector<glm::vec3> &poly,
                                          const glm::vec3 &normal,
                                          std::vector<std::array<glm::vec3, 3>> &out_tris) {
        out_tris.clear();
        if (poly.size() < 3) return;

        glm::vec3 u(1.0f), v(0.0f, 1.0f, 0.0f);
        make_face_basis(normal, u, v);
        std::vector<glm::vec2> p2 = project_loop_2d(poly, u, v);
        if (p2.size() < 3) return;

        if (signed_area_2d(p2) < 0.0f) {
            std::reverse(p2.begin(), p2.end());
            std::vector<glm::vec3> rev = poly;
            std::reverse(rev.begin(), rev.end());
            std::vector<std::size_t> idxs(rev.size());
            for (std::size_t i = 0; i < idxs.size(); ++i) idxs[i] = i;

            int guard = static_cast<int>(idxs.size() * idxs.size());
            while (idxs.size() >= 3 && guard-- > 0) {
                bool clipped = false;
                for (std::size_t ii = 0; ii < idxs.size(); ++ii) {
                    const std::size_t ip = (ii + idxs.size() - 1) % idxs.size();
                    const std::size_t in = (ii + 1) % idxs.size();
                    const std::size_t a = idxs[ip], b = idxs[ii], c = idxs[in];
                    const glm::vec2 &pa = p2[a], &pb = p2[b], &pc = p2[c];
                    const float turn = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
                    if (turn <= 1e-8f) continue;

                    bool contains = false;
                    for (std::size_t kk = 0; kk < idxs.size(); ++kk) {
                        if (kk == ip || kk == ii || kk == in) continue;
                        if (point_in_triangle_2d(p2[idxs[kk]], pa, pb, pc)) {
                            contains = true;
                            break;
                        }
                    }
                    if (contains) continue;

                    out_tris.push_back({rev[a], rev[b], rev[c]});
                    idxs.erase(idxs.begin() + static_cast<long>(ii));
                    clipped = true;
                    break;
                }
                if (!clipped) break;
            }
            return;
        }

        std::vector<std::size_t> idxs(poly.size());
        for (std::size_t i = 0; i < idxs.size(); ++i) idxs[i] = i;

        int guard = static_cast<int>(idxs.size() * idxs.size());
        while (idxs.size() >= 3 && guard-- > 0) {
            bool clipped = false;
            for (std::size_t ii = 0; ii < idxs.size(); ++ii) {
                const std::size_t ip = (ii + idxs.size() - 1) % idxs.size();
                const std::size_t in = (ii + 1) % idxs.size();
                const std::size_t a = idxs[ip], b = idxs[ii], c = idxs[in];
                const glm::vec2 &pa = p2[a], &pb = p2[b], &pc = p2[c];
                const float turn = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
                if (turn <= 1e-8f) continue;

                bool contains = false;
                for (std::size_t kk = 0; kk < idxs.size(); ++kk) {
                    if (kk == ip || kk == ii || kk == in) continue;
                    if (point_in_triangle_2d(p2[idxs[kk]], pa, pb, pc)) {
                        contains = true;
                        break;
                    }
                }
                if (contains) continue;

                out_tris.push_back({poly[a], poly[b], poly[c]});
                idxs.erase(idxs.begin() + static_cast<long>(ii));
                clipped = true;
                break;
            }
            if (!clipped) break;
        }
    };

    auto triangulate_face = [&](const RawFace &face, std::vector<std::array<glm::vec3, 3>> &out_tris) {
        out_tris.clear();

        auto clean_loop_2d3d = [&](std::vector<glm::vec3> &loop3, std::vector<glm::vec2> &loop2) {
            if (loop3.size() != loop2.size()) return;
            const float kPosEps = 1e-6f;
            const float kCrossEps = 1e-8f;

            bool changed = true;
            while (changed && loop3.size() >= 3) {
                changed = false;
                for (std::size_t i = 0; i < loop2.size(); ++i) {
                    const std::size_t ip = (i + loop2.size() - 1) % loop2.size();
                    const std::size_t in = (i + 1) % loop2.size();

                    if (glm::length(loop2[i] - loop2[ip]) <= kPosEps ||
                        glm::length(loop2[i] - loop2[in]) <= kPosEps) {
                        loop2.erase(loop2.begin() + static_cast<long>(i));
                        loop3.erase(loop3.begin() + static_cast<long>(i));
                        changed = true;
                        break;
                    }

                    const glm::vec2 a = loop2[i] - loop2[ip];
                    const glm::vec2 b = loop2[in] - loop2[i];
                    const float turn = a.x * b.y - a.y * b.x;
                    if (std::fabs(turn) <= kCrossEps && glm::dot(a, b) > 0.0f) {
                        loop2.erase(loop2.begin() + static_cast<long>(i));
                        loop3.erase(loop3.begin() + static_cast<long>(i));
                        changed = true;
                        break;
                    }
                }
            }
        };

        auto triangulate_poly_2d3d = [&](const std::vector<glm::vec3> &poly3,
                                         const std::vector<glm::vec2> &poly2,
                                         std::vector<std::array<glm::vec3, 3>> &tris) {
            tris.clear();
            if (poly3.size() < 3 || poly3.size() != poly2.size()) return;

            std::vector<std::size_t> idxs(poly2.size());
            for (std::size_t i = 0; i < idxs.size(); ++i) idxs[i] = i;

            int guard = static_cast<int>(idxs.size() * idxs.size());
            while (idxs.size() >= 3 && guard-- > 0) {
                bool clipped = false;
                for (std::size_t ii = 0; ii < idxs.size(); ++ii) {
                    const std::size_t ip = (ii + idxs.size() - 1) % idxs.size();
                    const std::size_t in = (ii + 1) % idxs.size();
                    const std::size_t a = idxs[ip], b = idxs[ii], c = idxs[in];

                    const glm::vec2 &pa = poly2[a];
                    const glm::vec2 &pb = poly2[b];
                    const glm::vec2 &pc = poly2[c];
                    const float turn = (pb.x - pa.x) * (pc.y - pa.y) - (pb.y - pa.y) * (pc.x - pa.x);
                    if (turn <= 1e-8f) continue;

                    bool contains = false;
                    for (std::size_t kk = 0; kk < idxs.size(); ++kk) {
                        if (kk == ip || kk == ii || kk == in) continue;
                        if (point_in_triangle_2d(poly2[idxs[kk]], pa, pb, pc)) {
                            contains = true;
                            break;
                        }
                    }
                    if (contains) continue;

                    tris.push_back({poly3[a], poly3[b], poly3[c]});
                    idxs.erase(idxs.begin() + static_cast<long>(ii));
                    clipped = true;
                    break;
                }
                if (!clipped) break;
            }
        };

        auto bridge_hole_2d3d = [&](std::vector<glm::vec3> &outer3,
                                    std::vector<glm::vec2> &outer2,
                                    const std::vector<glm::vec3> &hole3,
                                    const std::vector<glm::vec2> &hole2) -> bool {
            if (outer2.size() < 3 || hole2.size() < 3) return false;

            std::size_t hole_idx = 0;
            for (std::size_t i = 1; i < hole2.size(); ++i) {
                if (hole2[i].x > hole2[hole_idx].x + 1e-7f ||
                    (std::fabs(hole2[i].x - hole2[hole_idx].x) <= 1e-7f && hole2[i].y < hole2[hole_idx].y)) {
                    hole_idx = i;
                }
            }
            const glm::vec2 hp = hole2[hole_idx];

            int best_outer = -1;
            float best_dist2 = std::numeric_limits<float>::infinity();
            for (std::size_t oi = 0; oi < outer2.size(); ++oi) {
                const glm::vec2 op = outer2[oi];
                if (op.x <= hp.x + 1e-7f) continue;

                bool blocked = false;
                for (std::size_t i = 0; i < outer2.size(); ++i) {
                    const std::size_t in = (i + 1) % outer2.size();
                    if (i == oi || in == oi) continue;
                    if (segments_intersect_2d(hp, op, outer2[i], outer2[in])) {
                        blocked = true;
                        break;
                    }
                }
                if (blocked) continue;

                for (std::size_t i = 0; i < hole2.size(); ++i) {
                    const std::size_t in = (i + 1) % hole2.size();
                    if (i == hole_idx || in == hole_idx) continue;
                    if (segments_intersect_2d(hp, op, hole2[i], hole2[in])) {
                        blocked = true;
                        break;
                    }
                }
                if (blocked) continue;

                const glm::vec2 mid = 0.5f * (hp + op);
                if (!polygon_contains_point_exclusive(mid, outer2)) continue;

                const float d2 = glm::dot(op - hp, op - hp);
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_outer = static_cast<int>(oi);
                }
            }

            if (best_outer < 0) {
                for (std::size_t oi = 0; oi < outer2.size(); ++oi) {
                    const glm::vec2 op = outer2[oi];
                    const float d2 = glm::dot(op - hp, op - hp);
                    if (d2 < best_dist2) {
                        best_dist2 = d2;
                        best_outer = static_cast<int>(oi);
                    }
                }
            }
            if (best_outer < 0) return false;

            std::vector<glm::vec3> merged3;
            std::vector<glm::vec2> merged2;
            merged3.reserve(outer3.size() + hole3.size() + 2);
            merged2.reserve(outer2.size() + hole2.size() + 2);

            for (int i = 0; i <= best_outer; ++i) {
                merged3.push_back(outer3[static_cast<std::size_t>(i)]);
                merged2.push_back(outer2[static_cast<std::size_t>(i)]);
            }

            for (std::size_t k = 0; k < hole3.size(); ++k) {
                const std::size_t idx = (hole_idx + k) % hole3.size();
                merged3.push_back(hole3[idx]);
                merged2.push_back(hole2[idx]);
            }
            merged3.push_back(hole3[hole_idx]);
            merged2.push_back(hole2[hole_idx]);

            merged3.push_back(outer3[static_cast<std::size_t>(best_outer)]);
            merged2.push_back(outer2[static_cast<std::size_t>(best_outer)]);
            for (std::size_t i = static_cast<std::size_t>(best_outer) + 1; i < outer3.size(); ++i) {
                merged3.push_back(outer3[i]);
                merged2.push_back(outer2[i]);
            }

            outer3.swap(merged3);
            outer2.swap(merged2);
            clean_loop_2d3d(outer3, outer2);
            return outer3.size() >= 3;
        };

        if (face.bspline_surface_id >= 0 && face.holes.empty()) {
            const auto bs_it = bspline_surface_geoms.find(face.bspline_surface_id);
            if (bs_it != bspline_surface_geoms.end()) {
                tessellate_bspline_surface(bs_it->second, out_tris);
                if (!out_tris.empty()) {
                    return;
                }
            }
        }

        // Dedicated cylindrical meshing using UV unwrapping.
        if (face.cylindrical_surface_id >= 0) {
            const auto cyl_it = cylindrical_surfaces.find(face.cylindrical_surface_id);
            if (cyl_it != cylindrical_surfaces.end()) {
                std::vector<std::vector<glm::vec3>> cyl_loops;
                if (!face.loops.empty()) {
                    cyl_loops = face.loops;
                } else {
                    if (!face.outer.empty()) cyl_loops.push_back(face.outer);
                    for (const auto &h : face.holes) cyl_loops.push_back(h);
                }

                auto unwrap_cyl_loop = [&](const std::vector<glm::vec3> &loop) {
                    std::vector<glm::vec2> uv;
                    uv.reserve(loop.size());
                    if (loop.empty()) return uv;

                    float prev = 0.0f;
                    bool first = true;
                    for (const glm::vec3 &p : loop) {
                        const glm::vec3 rel = p - cyl_it->second.origin;
                        float ang = std::atan2(glm::dot(rel, cyl_it->second.y_axis),
                                               glm::dot(rel, cyl_it->second.x_axis));
                        const float z = glm::dot(rel, cyl_it->second.axis);

                        if (first) {
                            prev = ang;
                            first = false;
                        } else {
                            while (ang - prev > glm::pi<float>()) ang -= glm::two_pi<float>();
                            while (ang - prev < -glm::pi<float>()) ang += glm::two_pi<float>();
                            prev = ang;
                        }
                        uv.emplace_back(ang, z);
                    }
                    return uv;
                };

                auto align_hole_angles_to_outer = [&](const std::vector<glm::vec2> &outer_uv,
                                                     std::vector<glm::vec2> &hole_uv) {
                    if (outer_uv.empty() || hole_uv.empty()) {
                        return;
                    }

                    float outer_min = std::numeric_limits<float>::infinity();
                    float outer_max = -std::numeric_limits<float>::infinity();
                    float outer_sum = 0.0f;
                    for (const auto &p : outer_uv) {
                        outer_min = std::min(outer_min, p.x);
                        outer_max = std::max(outer_max, p.x);
                        outer_sum += p.x;
                    }
                    const float outer_center = outer_sum / static_cast<float>(outer_uv.size());

                    float hole_sum = 0.0f;
                    for (const auto &p : hole_uv) {
                        hole_sum += p.x;
                    }
                    const float hole_center = hole_sum / static_cast<float>(hole_uv.size());

                    const float two_pi = glm::two_pi<float>();
                    const float base_shift = std::round((outer_center - hole_center) / two_pi) * two_pi;
                    for (auto &p : hole_uv) {
                        p.x += base_shift;
                    }

                    float hole_min = std::numeric_limits<float>::infinity();
                    float hole_max = -std::numeric_limits<float>::infinity();
                    for (const auto &p : hole_uv) {
                        hole_min = std::min(hole_min, p.x);
                        hole_max = std::max(hole_max, p.x);
                    }

                    while (hole_max < outer_min) {
                        for (auto &p : hole_uv) p.x += two_pi;
                        hole_min += two_pi;
                        hole_max += two_pi;
                    }
                    while (hole_min > outer_max) {
                        for (auto &p : hole_uv) p.x -= two_pi;
                        hole_min -= two_pi;
                        hole_max -= two_pi;
                    }
                };

                std::vector<glm::vec3> outer3 = face.outer;
                std::vector<glm::vec2> outer2 = unwrap_cyl_loop(outer3);
                if (outer3.size() == outer2.size() && outer3.size() >= 3) {
                    clean_loop_2d3d(outer3, outer2);
                    if (outer3.size() >= 3) {
                        if (signed_area_2d(outer2) < 0.0f) {
                            std::reverse(outer3.begin(), outer3.end());
                            std::reverse(outer2.begin(), outer2.end());
                        }

                        bool holes_ok = true;
                        for (const auto &hole_loop : face.holes) {
                            std::vector<glm::vec3> hole3 = hole_loop;
                            std::vector<glm::vec2> hole2 = unwrap_cyl_loop(hole3);
                            if (hole3.size() != hole2.size() || hole3.size() < 3) continue;
                            clean_loop_2d3d(hole3, hole2);
                            if (hole3.size() < 3) continue;

                            align_hole_angles_to_outer(outer2, hole2);

                            if (signed_area_2d(hole2) > 0.0f) {
                                std::reverse(hole3.begin(), hole3.end());
                                std::reverse(hole2.begin(), hole2.end());
                            }

                            if (!bridge_hole_2d3d(outer3, outer2, hole3, hole2)) {
                                holes_ok = false;
                                break;
                            }
                        }

                        if (holes_ok) {
                            triangulate_poly_2d3d(outer3, outer2, out_tris);
                            // For cylindrical faces with holes, strict (n-2) matching can
                            // be too brittle after loop cleanup/bridging. If we produced
                            // a non-empty valid UV triangulation, prefer it over strip
                            // fallback to avoid seam-to-hole artifacts.
                            if (!out_tris.empty()) {
                                return;
                            }
                        }

                        if (!face.holes.empty()) {
                            std::printf("[STEP][CYL] Face #%d: UV hole triangulation produced no tris (holes=%zu), trying fallback strips\n",
                                        face.source_face_id,
                                        face.holes.size());
                        }
                    }
                }

                // Fallback A: stitch ordered boundary loops into strips.
                if (cyl_loops.size() >= 2) {
                    auto resample_loop = [](const std::vector<glm::vec3> &loop, int samples) {
                        std::vector<glm::vec3> out;
                        if (loop.size() < 2 || samples < 3) return out;

                        std::vector<float> cum(loop.size() + 1, 0.0f);
                        for (std::size_t i = 1; i <= loop.size(); ++i) {
                            const glm::vec3 &a = loop[i - 1];
                            const glm::vec3 &b = loop[i % loop.size()];
                            cum[i] = cum[i - 1] + glm::length(b - a);
                        }
                        const float total = cum.back();
                        if (total <= 1e-8f) return out;

                        out.reserve(static_cast<std::size_t>(samples));
                        for (int i = 0; i < samples; ++i) {
                            const float t = (static_cast<float>(i) / static_cast<float>(samples)) * total;
                            std::size_t seg = 1;
                            while (seg < cum.size() && cum[seg] < t) ++seg;
                            seg = std::min(seg, loop.size());
                            const std::size_t i0 = seg - 1;
                            const std::size_t i1 = seg % loop.size();
                            const float d0 = cum[seg - 1];
                            const float d1 = cum[seg];
                            const float alpha = (d1 > d0) ? ((t - d0) / (d1 - d0)) : 0.0f;
                            out.push_back(glm::mix(loop[i0], loop[i1], alpha));
                        }
                        return out;
                    };

                    auto loop_mean_axis_coord = [&](const std::vector<glm::vec3> &loop) {
                        float sum = 0.0f;
                        for (const auto &p : loop) {
                            sum += glm::dot(p - cyl_it->second.origin, cyl_it->second.axis);
                        }
                        return sum / static_cast<float>(std::max<std::size_t>(1, loop.size()));
                    };

                    std::vector<std::size_t> order(cyl_loops.size());
                    for (std::size_t i = 0; i < cyl_loops.size(); ++i) order[i] = i;
                    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                        return loop_mean_axis_coord(cyl_loops[a]) < loop_mean_axis_coord(cyl_loops[b]);
                    });

                    constexpr int kStripSamples = 56;
                    for (std::size_t oi = 0; oi + 1 < order.size(); ++oi) {
                        auto a = resample_loop(cyl_loops[order[oi]], kStripSamples);
                        auto b = resample_loop(cyl_loops[order[oi + 1]], kStripSamples);
                        if (a.size() != static_cast<std::size_t>(kStripSamples) ||
                            b.size() != static_cast<std::size_t>(kStripSamples)) {
                            continue;
                        }

                        std::size_t best_shift = 0;
                        float best_err = std::numeric_limits<float>::infinity();
                        for (std::size_t s = 0; s < b.size(); ++s) {
                            float err = 0.0f;
                            for (int k = 0; k < 8; ++k) {
                                const std::size_t i = (static_cast<std::size_t>(k) * b.size()) / 8;
                                const glm::vec3 d = a[i] - b[(i + s) % b.size()];
                                err += glm::dot(d, d);
                            }
                            if (err < best_err) {
                                best_err = err;
                                best_shift = s;
                            }
                        }

                        std::vector<glm::vec3> brot;
                        brot.reserve(b.size());
                        for (std::size_t i = 0; i < b.size(); ++i) {
                            brot.push_back(b[(i + best_shift) % b.size()]);
                        }

                        float fwd = 0.0f;
                        float rev = 0.0f;
                        for (int k = 0; k < 8; ++k) {
                            const std::size_t i = (static_cast<std::size_t>(k) * brot.size()) / 8;
                            const std::size_t in = (i + 1) % brot.size();
                            fwd += glm::length(a[in] - brot[in]);
                            rev += glm::length(a[in] - brot[(brot.size() + i - 1) % brot.size()]);
                        }
                        if (rev < fwd) {
                            std::reverse(brot.begin(), brot.end());
                        }

                        for (std::size_t i = 0; i < a.size(); ++i) {
                            const std::size_t in = (i + 1) % a.size();
                            out_tris.push_back({a[i], brot[i], brot[in]});
                            out_tris.push_back({a[i], brot[in], a[in]});
                        }
                    }

                    if (!out_tris.empty()) {
                        if (!face.holes.empty()) {
                            std::printf("[STEP][CYL] Face #%d: using strip fallback (tri_count=%zu)\n",
                                        face.source_face_id,
                                        out_tris.size());
                        }
                        return;
                    }
                }

                // Fallback B: infer two axial rings from a single boundary loop.
                if (cyl_loops.size() == 1) {
                    const auto &loop = cyl_loops[0];
                    if (loop.size() >= 8) {
                        float z_min = std::numeric_limits<float>::infinity();
                        float z_max = -std::numeric_limits<float>::infinity();
                        std::vector<float> zvals;
                        zvals.reserve(loop.size());
                        for (const auto &p : loop) {
                            const float z = glm::dot(p - cyl_it->second.origin, cyl_it->second.axis);
                            zvals.push_back(z);
                            z_min = std::min(z_min, z);
                            z_max = std::max(z_max, z);
                        }
                        const float z_span = z_max - z_min;
                        if (z_span > 1e-4f) {
                            const float band = std::max(1e-4f, z_span * 0.15f);
                            std::vector<std::pair<float, glm::vec3>> low;
                            std::vector<std::pair<float, glm::vec3>> high;
                            for (std::size_t i = 0; i < loop.size(); ++i) {
                                const glm::vec3 rel = loop[i] - cyl_it->second.origin;
                                const float ang = std::atan2(glm::dot(rel, cyl_it->second.y_axis),
                                                             glm::dot(rel, cyl_it->second.x_axis));
                                if (zvals[i] <= z_min + band) low.emplace_back(ang, loop[i]);
                                if (zvals[i] >= z_max - band) high.emplace_back(ang, loop[i]);
                            }

                            if (low.size() >= 6 && high.size() >= 6) {
                                auto by_angle = [](const auto &a, const auto &b) {
                                    return a.first < b.first;
                                };
                                std::sort(low.begin(), low.end(), by_angle);
                                std::sort(high.begin(), high.end(), by_angle);

                                constexpr int kRingSamples = 56;
                                auto resample_angular = [&](const std::vector<std::pair<float, glm::vec3>> &ring) {
                                    std::vector<glm::vec3> out;
                                    out.reserve(kRingSamples);
                                    for (int i = 0; i < kRingSamples; ++i) {
                                        const float t = -glm::pi<float>() + (glm::two_pi<float>() * static_cast<float>(i) / static_cast<float>(kRingSamples));
                                        std::size_t best = 0;
                                        float best_d = std::numeric_limits<float>::infinity();
                                        for (std::size_t k = 0; k < ring.size(); ++k) {
                                            float d = std::fabs(ring[k].first - t);
                                            d = std::min(d, glm::two_pi<float>() - d);
                                            if (d < best_d) {
                                                best_d = d;
                                                best = k;
                                            }
                                        }
                                        out.push_back(ring[best].second);
                                    }
                                    return out;
                                };

                                std::vector<glm::vec3> a = resample_angular(low);
                                std::vector<glm::vec3> b = resample_angular(high);
                                for (std::size_t i = 0; i < a.size(); ++i) {
                                    const std::size_t in = (i + 1) % a.size();
                                    out_tris.push_back({a[i], b[i], b[in]});
                                    out_tris.push_back({a[i], b[in], a[in]});
                                }
                                if (!out_tris.empty()) {
                                    if (!face.holes.empty()) {
                                        std::printf("[STEP][CYL] Face #%d: using inferred-ring fallback (tri_count=%zu)\n",
                                                    face.source_face_id,
                                                    out_tris.size());
                                    }
                                    return;
                                }
                            }
                        }
                    }
                }
            }
        }

        if (face.outer.size() < 3) return;

        glm::vec3 normal = face_loop_normal(face.outer);
        std::vector<glm::vec3> merged = face.outer;
        clean_loop(merged, normal);
        if (merged.size() < 3) return;

        glm::vec3 u(1.0f), v(0.0f, 1.0f, 0.0f);
        make_face_basis(normal, u, v);
        std::vector<glm::vec2> merged2 = project_loop_2d(merged, u, v);
        if (signed_area_2d(merged2) < 0.0f) {
            std::reverse(merged.begin(), merged.end());
        }

        bool hole_bridge_failed = false;
        for (auto hole : face.holes) {
            clean_loop(hole, normal);
            if (hole.size() < 3) continue;

            std::vector<glm::vec2> hole2 = project_loop_2d(hole, u, v);
            if (signed_area_2d(hole2) > 0.0f) {
                std::reverse(hole.begin(), hole.end());
            }

            if (!bridge_hole(merged, hole, normal)) {
                hole_bridge_failed = true;
                break;
            }
        }

        if (hole_bridge_failed) {
            out_tris.clear();
            return;
        }

        triangulate_simple_polygon(merged, normal, out_tris);
        const std::size_t merged_expected = merged.size() >= 3 ? (merged.size() - 2) : 0;
        if (out_tris.size() == merged_expected) {
            return;
        }

        // Never fallback to a solid outer fill when the face has holes.
        if (!face.holes.empty()) {
            out_tris.clear();
            return;
        }

        // Fallback for difficult topology/surface cases: at least render the outer loop.
        std::vector<glm::vec3> outer_only = face.outer;
        clean_loop(outer_only, normal);
        if (outer_only.size() < 3) {
            out_tris.clear();
            return;
        }

        std::vector<glm::vec2> outer2 = project_loop_2d(outer_only, u, v);
        if (signed_area_2d(outer2) < 0.0f) {
            std::reverse(outer_only.begin(), outer_only.end());
        }

        triangulate_simple_polygon(outer_only, normal, out_tris);
        const std::size_t outer_expected = outer_only.size() >= 3 ? (outer_only.size() - 2) : 0;
        if (out_tris.size() == outer_expected) {
            return;
        }

        // Last-resort fan to avoid dropping visible faces.
        out_tris.clear();
        for (std::size_t i = 1; i + 1 < outer_only.size(); ++i) {
            out_tris.push_back({outer_only[0], outer_only[i], outer_only[i + 1]});
        }
    };

    auto append_face_geometry = [&](int face_id, RawBody &out_body) {
        const auto face_it = entity_map.find(face_id);
        if (face_it == entity_map.end()) return;
        const std::string &face_body = face_it->second;
        if (get_type(face_body) != "ADVANCED_FACE") return;

        RawFace face;
        face.source_face_id = face_id;
        for (int r : get_refs(face_body)) {
            if (get_type_of(r) == "CYLINDRICAL_SURFACE") {
                face.cylindrical_surface_id = r;
            }

            const auto surf_it = entity_map.find(r);
            if (surf_it != entity_map.end()) {
                const std::string surf_type = get_type(surf_it->second);
                if (surf_type == "B_SPLINE_SURFACE_WITH_KNOTS" ||
                    body_has_keyword(surf_it->second, "B_SPLINE_SURFACE_WITH_KNOTS")) {
                    face.bspline_surface_id = r;
                }
            }
        }

        for (int bound_id : get_refs(face_body)) {
            const std::string btype = get_type_of(bound_id);
            if (btype != "FACE_OUTER_BOUND" && btype != "FACE_BOUND") continue;

            int loop_id = -1;
            for (int br : get_body_refs(bound_id)) {
                if (get_type_of(br) == "EDGE_LOOP") { loop_id = br; break; }
            }
            if (loop_id < 0) continue;

            std::vector<glm::vec3> loop;
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
                for (auto &seg : edge_segs) {
                    seg.first = transform_point(out_body.transform, seg.first);
                    seg.second = transform_point(out_body.transform, seg.second);
                }
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

                if (loop.empty()) {
                    loop.insert(loop.end(), edge_pts.begin(), edge_pts.end());
                } else {
                    std::size_t start_idx = 0;
                    if (glm::length(loop.back() - edge_pts.front()) <= kJoinEps)
                        start_idx = 1;
                    for (std::size_t i = start_idx; i < edge_pts.size(); ++i)
                        loop.push_back(edge_pts[i]);
                }
            }
            if (loop.size() >= 2
                && glm::length(loop.front() - loop.back()) <= kJoinEps)
                loop.pop_back();
            if (loop.size() < 3) continue;

            face.loops.push_back(loop);

            if (btype == "FACE_OUTER_BOUND") {
                if (face.outer.empty()) {
                    face.outer = std::move(loop);
                } else {
                    face.holes.push_back(std::move(loop));
                }
            } else {
                face.holes.push_back(std::move(loop));
            }
        }

        if (face.outer.empty() && !face.holes.empty()) {
            std::size_t best = 0;
            float best_area = -1.0f;
            for (std::size_t i = 0; i < face.holes.size(); ++i) {
                const glm::vec3 n = face_loop_normal(face.holes[i]);
                glm::vec3 u(1.0f), v(0.0f, 1.0f, 0.0f);
                make_face_basis(n, u, v);
                const float a = std::fabs(signed_area_2d(project_loop_2d(face.holes[i], u, v)));
                if (a > best_area) {
                    best_area = a;
                    best = i;
                }
            }
            face.outer = std::move(face.holes[best]);
            face.holes.erase(face.holes.begin() + static_cast<long>(best));
        }
        if (face.outer.size() >= 3) {
            out_body.faces.push_back(std::move(face));
        }
    };

    std::vector<RawBody> raw_bodies;
    raw_bodies.reserve(std::max<std::size_t>(1, body_seeds.size()));
    for (const auto &seed : body_seeds) {
        RawBody body;
        body.label = seed.label;
        body.transform = seed.transform;
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
    std::printf("[STEP] Edge tessellation paths: circle=%d ellipse=%d spline=%d straight=%d unresolved_fallback=%d\n",
                arc_circle_hits,
                arc_ellipse_hits,
                spline_hits,
                straight_line_hits,
                unresolved_curve_fallback_hits);
    if (!unresolved_types.empty()) {
        std::vector<std::pair<std::string, int>> unresolved_vec(unresolved_types.begin(), unresolved_types.end());
        std::sort(unresolved_vec.begin(), unresolved_vec.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });
        const std::size_t max_print = std::min<std::size_t>(8, unresolved_vec.size());
        std::printf("[STEP] Unresolved curve types:");
        for (std::size_t i = 0; i < max_print; ++i) {
            std::printf(" %s=%d", unresolved_vec[i].first.c_str(), unresolved_vec[i].second);
        }
        std::printf("\n");
    }
    if (!unresolved_curve_ids.empty()) {
        std::vector<std::pair<int, int>> unresolved_id_vec(unresolved_curve_ids.begin(), unresolved_curve_ids.end());
        std::sort(unresolved_id_vec.begin(), unresolved_id_vec.end(),
                  [](const auto &a, const auto &b) { return a.second > b.second; });
        const std::size_t max_print = std::min<std::size_t>(8, unresolved_id_vec.size());
        std::printf("[STEP] Unresolved curve ids:");
        for (std::size_t i = 0; i < max_print; ++i) {
            const int cid = unresolved_id_vec[i].first;
            const int cnt = unresolved_id_vec[i].second;
            const bool exists = entity_map.find(cid) != entity_map.end();
            std::printf(" #%d=%d(%s)", cid, cnt, exists ? "exists" : "missing");
        }
        std::printf("\n");
    }
    if (progress) progress->store(0.55f);

    // ---------- bounding box ----------
    glm::vec3 bb_min(std::numeric_limits<float>::infinity());
    glm::vec3 bb_max(-std::numeric_limits<float>::infinity());
    for (const auto &body : raw_bodies) {
        for (const auto &f : body.faces) {
            for (const auto &v : f.outer) {
                bb_min = glm::min(bb_min, v);
                bb_max = glm::max(bb_max, v);
            }
            for (const auto &hole : f.holes) {
                for (const auto &v : hole) {
                    bb_min = glm::min(bb_min, v);
                    bb_max = glm::max(bb_max, v);
                }
            }
        }
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
    std::printf("[STEP] Raw bounds min=(%.6f, %.6f, %.6f) max=(%.6f, %.6f, %.6f) size=(%.6f, %.6f, %.6f)\n",
                bb_min.x, bb_min.y, bb_min.z,
                bb_max.x, bb_max.y, bb_max.z,
                bb_size.x, bb_size.y, bb_size.z);
    std::printf("[STEP] Normalization center=(%.6f, %.6f, %.6f) scale=%.9f\n",
                center.x, center.y, center.z, scale);
    auto xform = [&](const glm::vec3 &v) -> glm::vec3 { return (v - center) * scale; };

    // ---------- build CPU vertex buffers ----------
    const glm::vec3 solid_color(0.44f, 0.44f, 0.50f);
    const glm::vec3 edge_color(0.75f, 0.75f, 0.75f);

    std::vector<PreparedImportBody> built_bodies(raw_bodies.size());
    std::vector<unsigned char> has_body(raw_bodies.size(), 0);

    std::atomic<std::size_t> next_index{0};
    std::atomic<std::size_t> completed{0};

    auto build_body = [&](std::size_t idx) {
        const auto &raw_body = raw_bodies[idx];

        ImportedBodyItem body_item;
        body_item.label = raw_body.label;
        body_item.face_count = static_cast<int>(raw_body.faces.size());
        body_item.edge_count = raw_body.edge_entity_count;
        body_item.visible = true;

        PreparedImportBody body_prepared;
        body_prepared.item = std::move(body_item);

        for (const auto &face : raw_body.faces) {
            std::vector<std::array<glm::vec3, 3>> tris;
            triangulate_face(face, tris);
            if (tris.empty()) continue;

            for (const auto &tri : tris) {
                for (const glm::vec3 &v : tri) {
                    const glm::vec3 t = xform(v);
                    body_prepared.solid_verts.push_back(t.x); body_prepared.solid_verts.push_back(t.y); body_prepared.solid_verts.push_back(t.z);
                    body_prepared.solid_verts.push_back(solid_color.r); body_prepared.solid_verts.push_back(solid_color.g); body_prepared.solid_verts.push_back(solid_color.b);
                }
            }
        }

        for (const auto &[a, b] : raw_body.edges) {
            const glm::vec3 ta = xform(a), tb = xform(b);
            body_prepared.line_verts.push_back(ta.x); body_prepared.line_verts.push_back(ta.y); body_prepared.line_verts.push_back(ta.z);
            body_prepared.line_verts.push_back(edge_color.r); body_prepared.line_verts.push_back(edge_color.g); body_prepared.line_verts.push_back(edge_color.b);
            body_prepared.line_verts.push_back(tb.x); body_prepared.line_verts.push_back(tb.y); body_prepared.line_verts.push_back(tb.z);
            body_prepared.line_verts.push_back(edge_color.r); body_prepared.line_verts.push_back(edge_color.g); body_prepared.line_verts.push_back(edge_color.b);
        }

        if (!body_prepared.line_verts.empty() || !body_prepared.solid_verts.empty()) {
            built_bodies[idx] = std::move(body_prepared);
            has_body[idx] = 1;
        }

        if (progress) {
            const std::size_t done = completed.fetch_add(1) + 1;
            const float frac = static_cast<float>(done) / static_cast<float>(std::max<std::size_t>(1, raw_bodies.size()));
            progress->store(0.55f + (frac * 0.43f));
        }
    };

    const unsigned int hw = std::max(1u, std::thread::hardware_concurrency());
    const std::size_t worker_count = std::min<std::size_t>(raw_bodies.size(), static_cast<std::size_t>(hw));

    if (worker_count <= 1) {
        for (std::size_t i = 0; i < raw_bodies.size(); ++i) {
            build_body(i);
        }
    } else {
        std::vector<std::future<void>> workers;
        workers.reserve(worker_count);
        for (std::size_t w = 0; w < worker_count; ++w) {
            workers.push_back(std::async(std::launch::async, [&]() {
                while (true) {
                    const std::size_t idx = next_index.fetch_add(1);
                    if (idx >= raw_bodies.size()) break;
                    build_body(idx);
                }
            }));
        }
        for (auto &worker : workers) {
            worker.get();
        }
    }

    out_prepared.bodies.clear();
    out_prepared.bodies.reserve(raw_bodies.size());
    for (std::size_t i = 0; i < built_bodies.size(); ++i) {
        if (has_body[i] != 0) {
            out_prepared.bodies.push_back(std::move(built_bodies[i]));
        }
    }

    if (progress) progress->store(1.0f);
    return !out_prepared.bodies.empty();
}

bool Renderer::apply_prepared_step_import(PreparedImport &&prepared, std::string &error_message)
{
    error_message.clear();
    release_imported_gpu_buffers();
    m_imported_bodies.clear();
    m_imported_body_render_data.clear();

    for (auto &prepared_body : prepared.bodies) {
        ImportedBodyRenderData body_render;
        upload_geometry_vao(body_render.solid_vao,
                            body_render.solid_vbo,
                            body_render.solid_vertex_count,
                            prepared_body.solid_verts,
                            GL_STATIC_DRAW);
        upload_geometry_vao(body_render.line_vao,
                            body_render.line_vbo,
                            body_render.line_vertex_count,
                            prepared_body.line_verts,
                            GL_STATIC_DRAW);

        if (body_render.line_vertex_count <= 0 && body_render.solid_vertex_count <= 0) {
            continue;
        }

        m_imported_bodies.push_back(std::move(prepared_body.item));
        m_imported_body_render_data.push_back(std::move(body_render));
    }

    m_has_imported_model = !m_imported_body_render_data.empty();
    if (!m_has_imported_model) {
        error_message = "Prepared STEP import had no uploadable geometry.";
        return false;
    }

    std::printf("[STEP] Imported bodies: %zu\n", m_imported_bodies.size());
    clear_hovered();
    clear_selection();
    camera.reset();
    return true;
}

bool Renderer::import_step_file(const std::string &path, std::string &error_message)
{
    return import_model_file(path, error_message);
}

void Renderer::ensure_model_loaders()
{
    if (!m_model_loaders.empty()) {
        return;
    }

    m_model_loaders.push_back(std::make_unique<loaders::StepLoader>());
}

bool Renderer::import_model_file(const std::string &path, std::string &error_message)
{
    ensure_model_loaders();

    for (const auto &loader : m_model_loaders) {
        if (!loader->can_load(path)) {
            continue;
        }
        return loader->import(*this, path, error_message);
    }

    error_message = "No loader registered for this file type.";
    return false;
}

bool Renderer::load_debug_cylinder_scene(std::string &error_message)
{
    PreparedImport prepared;
    prepared.bodies.reserve(4);

    struct DebugCylinderPlacement {
        glm::vec3 position;
        glm::vec3 euler_deg;
        const char *label;
    };

    const std::array<DebugCylinderPlacement, 4> placements = {{
        {{-0.95f,  0.00f,  0.10f}, {  0.0f,   0.0f,   0.0f}, "Body 1"},
        {{-0.25f,  0.12f, -0.35f}, {  0.0f,   0.0f,  35.0f}, "Body 2"},
        {{ 0.45f, -0.10f,  0.20f}, { 48.0f,   0.0f,   0.0f}, "Body 3"},
        {{ 1.05f,  0.06f, -0.20f}, { 34.0f,  18.0f, -26.0f}, "Body 4"},
    }};

    for (const auto &placement : placements) {
        PreparedImportBody body;
        body.item.label = placement.label;
        body.item.face_count = 3;
        body.item.edge_count = 3;
        body.item.visible = true;

        glm::mat4 transform(1.0f);
        transform = glm::translate(transform, placement.position);
        transform = glm::rotate(transform, glm::radians(placement.euler_deg.z), glm::vec3(0.0f, 0.0f, 1.0f));
        transform = glm::rotate(transform, glm::radians(placement.euler_deg.y), glm::vec3(0.0f, 1.0f, 0.0f));
        transform = glm::rotate(transform, glm::radians(placement.euler_deg.x), glm::vec3(1.0f, 0.0f, 0.0f));

        append_debug_cylinder_mesh(body,
                                   transform,
                                   0.22f,
                                   0.58f,
                                   64,
                                   glm::vec3(1.0f, 1.0f, 1.0f),
                                   glm::vec3(0.18f, 0.18f, 0.18f));

        prepared.bodies.push_back(std::move(body));
    }

    return apply_prepared_step_import(std::move(prepared), error_message);
}

void Renderer::release_imported_gpu_buffers()
{
    for (auto &body : m_imported_body_render_data) {
        if (body.line_vao != 0) {
            glDeleteVertexArrays(1, &body.line_vao);
            body.line_vao = 0;
        }
        if (body.line_vbo != 0) {
            glDeleteBuffers(1, &body.line_vbo);
            body.line_vbo = 0;
        }
        if (body.solid_vao != 0) {
            glDeleteVertexArrays(1, &body.solid_vao);
            body.solid_vao = 0;
        }
        if (body.solid_vbo != 0) {
            glDeleteBuffers(1, &body.solid_vbo);
            body.solid_vbo = 0;
        }
        body.line_vertex_count = 0;
        body.solid_vertex_count = 0;
    }
}

void Renderer::clear_imported_model()
{
    release_imported_gpu_buffers();
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
        return m_selected_hit.world_pos;
    }

    if (m_selected_hit.type == PrimitiveType::Edge) {
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
        configure_sketch_plane(selected_hit.world_pos, selected_hit.normal);
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
    release_imported_gpu_buffers();
    glDeleteVertexArrays(1, &m_axis_vao);
    glDeleteBuffers(1, &m_axis_vbo);
    glDeleteVertexArrays(1, &m_axis_cone_vao);
    glDeleteBuffers(1, &m_axis_cone_vbo);
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
