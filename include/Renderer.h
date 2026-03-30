#pragma once
#include <epoxy/gl.h>
#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include "OrbitCamera.h"
#include "loaders/ModelLoader.h"

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

    struct ImportedBodyItem {
        std::string label;
        int face_count = 0;
        int edge_count = 0;
        bool visible = true;
    };

    struct PreparedImportBody {
        ImportedBodyItem item;
        std::vector<float> line_verts;
        std::vector<float> solid_verts;
    };

    struct PreparedImport {
        std::vector<PreparedImportBody> bodies;
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

    void set_lighting_enabled(bool enabled) { m_lighting_enabled = enabled; }
    bool lighting_enabled() const { return m_lighting_enabled; }
    void set_light_ambient(float ambient);
    float light_ambient() const { return m_light_ambient; }
    void set_light_direction(const glm::vec3 &dir);
    const glm::vec3 &light_direction() const { return m_light_dir; }
    void rotate_light(float yaw_degrees, float pitch_degrees);

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

    // Return a geometric center for the current selection (face centroid, edge midpoint,
    // or fallback to the hit world position). Useful for camera focusing and centering.
    glm::vec3 selection_center() const;

    // Helpers for UI: project world/local sketch coords to screen space
    bool world_to_screen(const glm::vec3 &world, glm::vec2 &out_screen) const;
    bool sketch_preview_rectangle_local(SketchRectangle &out_rect) const;
    bool sketch_local_to_screen(const glm::vec2 &local, glm::vec2 &out_screen) const;
    // Adjust the in-progress rectangle preview dimensions (meters).
    void set_sketch_preview_rectangle_dims(float width_m, float height_m);
    bool commit_sketch_preview();
    // Modify the sketch preview local coordinate (used while creating rectangles)
    void set_sketch_preview_local(const glm::vec2 &local);
    // Return the current sketch anchor local coordinate
    glm::vec2 sketch_anchor_local() const;

    // Import a STEP file (edge wireframe only for now) and fit it to view.
    bool import_model_file(const std::string &path, std::string &error_message);
    bool import_step_file(const std::string &path, std::string &error_message);
    bool load_debug_cylinder_scene(std::string &error_message);
    bool prepare_step_file_import(const std::string &path,
                                  PreparedImport &out_prepared,
                                  std::string &error_message,
                                  std::atomic<float> *progress = nullptr) const;
    bool apply_prepared_step_import(PreparedImport &&prepared, std::string &error_message);
    void clear_imported_model();
    bool has_imported_model() const { return m_has_imported_model; }
    const std::vector<ImportedBodyItem> &imported_bodies() const { return m_imported_bodies; }
    bool set_imported_body_visible(std::size_t body_index, bool visible);
    bool toggle_imported_body_visible(std::size_t body_index);

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
    GLuint m_uLit = 0;
    GLuint m_uLightDir = 0;
    GLuint m_uAmbient = 0;

    bool m_lighting_enabled = true;
    glm::vec3 m_light_dir = glm::normalize(glm::vec3(0.55f, 0.75f, 0.40f));
    float m_light_ambient = 0.36f;

    // Axis gizmo (3 coloured lines)
    GLuint m_axis_vao = 0;
    GLuint m_axis_vbo = 0;
    GLuint m_axis_cone_vao = 0;
    GLuint m_axis_cone_vbo = 0;

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

    bool m_has_imported_model = false;
    std::vector<ImportedBodyItem> m_imported_bodies;

    struct ImportedBodyRenderData {
        std::vector<float> line_verts;
        std::vector<float> solid_verts;
        GLuint line_vao = 0;
        GLuint line_vbo = 0;
        GLsizei line_vertex_count = 0;
        GLuint solid_vao = 0;
        GLuint solid_vbo = 0;
        GLsizei solid_vertex_count = 0;
    };
    std::vector<ImportedBodyRenderData> m_imported_body_render_data;

    std::vector<std::unique_ptr<loaders::ModelLoader>> m_model_loaders;
    void ensure_model_loaders();

    void release_imported_gpu_buffers();

    glm::mat4 projection_matrix() const;
    PrimitiveHit pick_face(double mouse_x, double mouse_y) const;
    PrimitiveHit pick_edge(double mouse_x, double mouse_y) const;
    glm::vec2 project_to_screen(const glm::vec3 &world, float &ndc_z, bool &ok) const;
    bool pivot_marker_visible() const;
    float effective_sketch_grid_spacing() const;
    void draw_overlay_lines(const std::vector<float> &verts, float line_width) const;
    void draw_overlay_triangles(const std::vector<float> &verts) const;
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
