#pragma once
#include <epoxy/gl.h>
#include <chrono>
#include <string>
#include <vector>
#include "OrbitCamera.h"

class Renderer {
public:
    enum class ProjectionMode {
        Perspective,
        Isometric,
    };

    enum class PrimitiveType {
        None,
        Face,
        Edge,
    };

    struct PrimitiveHit {
        PrimitiveType type = PrimitiveType::None;
        int index = -1;
        glm::vec3 world_pos = glm::vec3(0.0f);
        glm::vec3 normal = glm::vec3(0.0f, 0.0f, 1.0f);
        float depth = 1.0f;

        bool valid() const { return type != PrimitiveType::None && index >= 0; }
    };

    struct SketchPlane {
        glm::vec3 origin = glm::vec3(0.0f);
        glm::vec3 x_axis = glm::vec3(1.0f, 0.0f, 0.0f);
        glm::vec3 y_axis = glm::vec3(0.0f, 1.0f, 0.0f);
        glm::vec3 normal = glm::vec3(0.0f, 0.0f, 1.0f);
    };

    enum class SketchTool {
        Line,
        Rectangle,
        Circle,
    };

    enum class SketchConstraintType {
        None,
        Horizontal,
        Vertical,
        Radius,
    };

    struct SketchLine {
        glm::vec2 start = glm::vec2(0.0f);
        glm::vec2 end = glm::vec2(0.0f);
        SketchConstraintType constraint = SketchConstraintType::None;
    };

    struct SketchRectangle {
        glm::vec2 min_corner = glm::vec2(0.0f);
        glm::vec2 max_corner = glm::vec2(0.0f);
    };

    struct SketchCircle {
        glm::vec2 center = glm::vec2(0.0f);
        float radius = 0.0f;
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

    void update_hovered(double mouse_x, double mouse_y);
    void clear_hovered();
    void select_hovered();
    void clear_selection();
    bool set_pivot_from_click(double mouse_x, double mouse_y);

    void begin_sketch(const PrimitiveHit &selected_hit);
    void end_sketch();
    bool sketch_active() const { return m_sketch_active; }
    bool sketch_has_preview() const { return m_sketch_entity_in_progress; }
    bool update_sketch_cursor(double mouse_x, double mouse_y);
    bool handle_sketch_click(double mouse_x, double mouse_y);
    void cancel_sketch_preview();
    const SketchPlane &sketch_plane() const { return m_sketch_plane; }
    void set_sketch_tool(SketchTool tool);
    SketchTool sketch_tool() const { return m_sketch_tool; }
    const char *sketch_tool_name() const;
    std::string sketch_constraint_hint() const;

    const PrimitiveHit &hovered_hit() const { return m_hovered_hit; }
    const PrimitiveHit &selected_hit() const { return m_selected_hit; }

    // Exposed so input handlers can manipulate the camera directly.
    OrbitCamera camera;

private:
    // Shader helpers
    GLuint compile_shader(GLenum type, const char *src);
    GLuint link_program(GLuint vert, GLuint frag);

    GLuint m_program = 0;
    GLuint m_uMVP    = 0;   // uniform location
    GLuint m_uColorMul = 0;
    GLuint m_uColorAdd = 0;

    // Axis gizmo (3 coloured lines)
    GLuint m_axis_vao = 0;
    GLuint m_axis_vbo = 0;

    // Wireframe unit cube (12 edges)
    GLuint m_cube_vao  = 0;
    GLuint m_cube_vbo  = 0;

    // Solid unit cube (12 triangles)
    GLuint m_solid_cube_vao = 0;
    GLuint m_solid_cube_vbo = 0;

    // Temporary pivot marker
    GLuint m_pivot_marker_vao = 0;
    GLuint m_pivot_marker_vbo = 0;

    // Dynamic line overlays (sketch grid/entities)
    GLuint m_overlay_vao = 0;
    GLuint m_overlay_vbo = 0;

    int m_width  = 1280;
    int m_height = 720;
    bool m_wireframe = false;
    ProjectionMode m_projection_mode = ProjectionMode::Perspective;

    PrimitiveHit m_hovered_hit{};
    PrimitiveHit m_selected_hit{};
    glm::vec3 m_pivot_marker_pos = glm::vec3(0.0f);
    glm::vec3 m_pivot_marker_normal = glm::vec3(0.0f, 0.0f, 1.0f);
    std::chrono::steady_clock::time_point m_pivot_marker_until{};

    bool m_sketch_active = false;
    SketchPlane m_sketch_plane{};
    SketchTool m_sketch_tool = SketchTool::Line;
    std::vector<SketchLine> m_sketch_lines;
    std::vector<SketchRectangle> m_sketch_rectangles;
    std::vector<SketchCircle> m_sketch_circles;
    bool m_sketch_cursor_valid = false;
    glm::vec2 m_sketch_cursor_local = glm::vec2(0.0f);
    glm::vec3 m_sketch_cursor_world = glm::vec3(0.0f);
    bool m_sketch_cursor_snapped = false;
    bool m_sketch_entity_in_progress = false;
    glm::vec2 m_sketch_anchor = glm::vec2(0.0f);
    glm::vec2 m_sketch_preview_local = glm::vec2(0.0f);
    glm::vec3 m_sketch_preview_world = glm::vec3(0.0f);
    SketchConstraintType m_sketch_preview_constraint = SketchConstraintType::None;
    bool m_sketch_snap_to_grid = true;
    float m_sketch_grid_spacing = 0.25f;

    glm::mat4 projection_matrix() const;
    PrimitiveHit pick_face(double mouse_x, double mouse_y) const;
    PrimitiveHit pick_edge(double mouse_x, double mouse_y) const;
    glm::vec2 project_to_screen(const glm::vec3 &world, float &ndc_z, bool &ok) const;
    bool pivot_marker_visible() const;
    void draw_overlay_lines(const std::vector<float> &verts, float line_width) const;
    glm::vec3 sketch_world_from_local(const glm::vec2 &local) const;
    glm::vec2 sketch_local_from_world(const glm::vec3 &world) const;
    bool raycast_to_sketch_plane(double mouse_x, double mouse_y, glm::vec3 &world, glm::vec2 &local) const;
    glm::vec2 snapped_sketch_local(const glm::vec2 &local, bool &snapped_to_vertex) const;
    glm::vec2 constrained_line_endpoint(const glm::vec2 &anchor,
                                        const glm::vec2 &candidate,
                                        SketchConstraintType &constraint) const;
    void configure_sketch_plane(const glm::vec3 &origin, const glm::vec3 &normal);
    glm::vec3 cube_face_center(int face_index) const;
    glm::vec3 cube_face_normal(int face_index) const;
};
