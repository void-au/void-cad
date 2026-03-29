#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <epoxy/gl.h>

#include "Renderer.h"
#include "UiRenderer.h"

namespace {

constexpr float kTextScale = 1.0f;
constexpr float kOuterMargin = 8.0f;
constexpr float kButtonHeight = 30.0f;
constexpr float kButtonPaddingX = 10.0f;
constexpr float kButtonGap = 6.0f;
constexpr float kMenuGap = 3.0f;
constexpr float kMenuPanelPadding = 6.0f;
constexpr float kBadgePaddingX = 8.0f;
constexpr float kBadgeHeight = 24.0f;
constexpr float kHierarchyTopOffset = 46.0f;
constexpr float kHierarchyWidth = 220.0f; // moved to left, smaller width
constexpr float kHierarchyRowHeight = 18.0f; // slightly smaller rows
constexpr float kHierarchyRowGap = 2.0f;
constexpr float kHierarchyEyeWidth = 20.0f;
constexpr float kHierarchyPaddingX = 6.0f;
constexpr double kDoubleClickTimeSeconds = 0.30;
constexpr double kDoubleClickDistancePx = 8.0;

enum class ToolMode {
    None,
    Sketch,
};
static const std::array<const char *, 4> kMenuTitles = {
    "FILE", "EDIT", "VIEW", "HELP"
};

static const std::vector<std::string> kFileItems = {
    "NEW", "OPEN", "SAVE", "SAVE AS", "QUIT"
};
static const std::vector<std::string> kEditItems = {
    "UNDO", "REDO", "PREFERENCES"
};
static const std::vector<std::string> kViewItems = {
    "RESET CAMERA", "WIREFRAME", "ISO/PERSP", "HIERARCHY"
};


struct UiLayout {
    std::array<UiRect, 4> menu_buttons{};
    std::array<UiRect, 4> toolbar_buttons{};
    UiRect open_menu_panel{};
    std::vector<UiRect> open_menu_items;
    std::array<UiRect, 3> sketch_tool_buttons{};
    UiRect hierarchy_root_rect{};
    std::vector<UiRect> hierarchy_row_rects;
    std::vector<UiRect> hierarchy_eye_rects;
};

struct AppState {
    GLFWwindow *window = nullptr;
    Renderer renderer;
    UiRenderer ui;

    int framebuffer_width = 1280;
    int framebuffer_height = 720;
    int window_width = 1280;
    int window_height = 720;

    double mouse_x = 0.0;
    double mouse_y = 0.0;
    double last_drag_x = 0.0;
    double last_drag_y = 0.0;

    bool left_down = false;
    bool middle_down = false;
    bool right_down = false;
    bool left_captured_by_ui = false;
    bool left_dragged = false;
    bool middle_captured_by_ui = false;
    bool middle_dragged = false;
    bool right_dragged = false;
    double left_press_x = 0.0;
    double left_press_y = 0.0;
    double last_middle_click_time = -1.0;
    double last_middle_click_x = 0.0;
    double last_middle_click_y = 0.0;

    int open_menu = -1;
    bool wireframe = false;
    bool show_hierarchy = true;
    ToolMode tool_mode = ToolMode::None;
    float hierarchy_scroll = 0.0f;

    std::string status = "CUSTOM UI READY";
    double fps = 0.0;
    int frame_counter = 0;
    double fps_accumulator = 0.0;

    // Camera animation state for smooth transitions
    struct CameraAnim {
        bool active = false;
        double start_time = 0.0; // glfwGetTime() epoch
        double duration = 0.4;   // seconds

        glm::vec3 start_target = glm::vec3(0.0f);
        glm::vec3 end_target = glm::vec3(0.0f);

        float start_distance = 4.0f;
        float end_distance = 4.0f;

        float start_az = 0.0f;
        float end_az = 0.0f;

        float start_el = 0.0f;
        float end_el = 0.0f;
    } camera_anim;
    // Rectangle measurement input state for sketch UI
    enum class MeasureFocus { None = 0, Width = 1, Height = 2 };
    MeasureFocus measure_focus = MeasureFocus::None;
    bool measure_editing = false;
    bool measure_width_locked = false;
    bool measure_height_locked = false;
    float measure_width_m = 0.0f;
    float measure_height_m = 0.0f;
    std::string measure_input;
    UiRect measure_width_rect{};
    UiRect measure_height_rect{};
};
static const std::vector<std::string> kHelpItems = {
    "ABOUT"
};

static const std::array<const std::vector<std::string> *, 4> kMenuItems = {
    &kFileItems, &kEditItems, &kViewItems, &kHelpItems
};

static const std::array<const char *, 4> kToolbarLabels = {
    "RESET CAMERA", "WIREFRAME", "ISO/PERSP", "SKETCH"
};

static const std::array<const char *, 3> kSketchToolLabels = {
    "LINE", "RECT", "CIRCLE"
};

UiColor rgba(float r, float g, float b, float a)
{
    return {r, g, b, a};
}

float button_width(const UiRenderer &ui, std::string_view label)
{
    return ui.measure_text(label, kTextScale) + (kButtonPaddingX * 2.0f);
}

UiLayout build_ui_layout(const AppState &app)
{
    UiLayout layout;

    float menu_x = kOuterMargin;
    const float top_y = kOuterMargin;
    for (std::size_t i = 0; i < kMenuTitles.size(); ++i) {
        const float width = button_width(app.ui, kMenuTitles[i]);
        layout.menu_buttons[i] = {menu_x, top_y, width, kButtonHeight};
        menu_x += width + kMenuGap;
    }

    float toolbar_width = 0.0f;
    for (const char *label : kToolbarLabels) {
        toolbar_width += button_width(app.ui, label);
    }
    toolbar_width += kButtonGap * static_cast<float>(kToolbarLabels.size() - 1);

    float toolbar_x = static_cast<float>(app.window_width) - kOuterMargin - toolbar_width;
    for (std::size_t i = 0; i < kToolbarLabels.size(); ++i) {
        const float width = button_width(app.ui, kToolbarLabels[i]);
        layout.toolbar_buttons[i] = {toolbar_x, top_y, width, kButtonHeight};
        toolbar_x += width + kButtonGap;
    }

    if (app.open_menu >= 0 && app.open_menu < static_cast<int>(kMenuTitles.size())) {
        const auto &items = *kMenuItems[static_cast<std::size_t>(app.open_menu)];
        float panel_width = 180.0f;
        for (const std::string &item : items) {
            panel_width = std::max(panel_width, button_width(app.ui, item));
        }
        panel_width += kMenuPanelPadding * 2.0f;

        const float item_height = kButtonHeight;
        const float panel_height = (item_height * static_cast<float>(items.size())) + (kMenuPanelPadding * 2.0f);
        const UiRect anchor = layout.menu_buttons[static_cast<std::size_t>(app.open_menu)];

        layout.open_menu_panel = {
            anchor.x,
            anchor.y + anchor.h + 4.0f,
            panel_width,
            panel_height,
        };

        layout.open_menu_items.reserve(items.size());
        for (std::size_t i = 0; i < items.size(); ++i) {
            layout.open_menu_items.push_back({
                layout.open_menu_panel.x + kMenuPanelPadding,
                layout.open_menu_panel.y + kMenuPanelPadding + (item_height * static_cast<float>(i)),
                panel_width - (kMenuPanelPadding * 2.0f),
                item_height,
            });
        }
    }

    if (app.tool_mode == ToolMode::Sketch) {
        float total_width = 0.0f;
        for (const char *label : kSketchToolLabels) {
            total_width += button_width(app.ui, label);
        }
        total_width += kButtonGap * static_cast<float>(kSketchToolLabels.size() - 1);

        float x = static_cast<float>(app.window_width) - kOuterMargin - total_width;
        const float y = kOuterMargin + kButtonHeight + 8.0f;
        for (std::size_t i = 0; i < kSketchToolLabels.size(); ++i) {
            const float width = button_width(app.ui, kSketchToolLabels[i]);
            layout.sketch_tool_buttons[i] = {x, y, width, kButtonHeight - 4.0f};
            x += width + kButtonGap;
        }
    }

    if (app.show_hierarchy && app.renderer.has_imported_model()) {
        const auto &bodies = app.renderer.imported_bodies();
        const float x = kOuterMargin; // left side now
        const float y = kOuterMargin + kHierarchyTopOffset;
        const float row_pitch = kHierarchyRowHeight + kHierarchyRowGap;

        const float available_bottom = static_cast<float>(app.window_height) - kBadgeHeight - kOuterMargin - 8.0f;
        layout.hierarchy_root_rect = {x, y, kHierarchyWidth, kHierarchyRowHeight};
        layout.hierarchy_row_rects.reserve(bodies.size());
        layout.hierarchy_eye_rects.reserve(bodies.size());

        // Build rows, applying scroll offset and only include visible rows
        for (std::size_t i = 0; i < bodies.size(); ++i) {
            const float row_y = y + row_pitch * static_cast<float>(i + 1) - app.hierarchy_scroll;
            if (row_y + kHierarchyRowHeight < y) continue; // above view
            if (row_y > available_bottom) break; // below view
            const UiRect row = {x + 10.0f, row_y, kHierarchyWidth - 10.0f, kHierarchyRowHeight};
            const UiRect eye = {row.x + kHierarchyPaddingX, row.y + 2.0f, kHierarchyEyeWidth, kHierarchyRowHeight - 4.0f};
            layout.hierarchy_row_rects.push_back(row);
            layout.hierarchy_eye_rects.push_back(eye);
        }
    }

    return layout;
}

bool point_hits_ui(const AppState &app)
{
    const UiLayout layout = build_ui_layout(app);

    for (const UiRect &rect : layout.menu_buttons) {
        if (rect.contains(app.mouse_x, app.mouse_y)) {
            return true;
        }
    }

    for (const UiRect &rect : layout.toolbar_buttons) {
        if (rect.contains(app.mouse_x, app.mouse_y)) {
            return true;
        }
    }

    if (app.open_menu >= 0) {
        if (layout.open_menu_panel.contains(app.mouse_x, app.mouse_y)) {
            return true;
        }
        for (const UiRect &rect : layout.open_menu_items) {
            if (rect.contains(app.mouse_x, app.mouse_y)) {
                return true;
            }
        }
    }

    if (app.tool_mode == ToolMode::Sketch) {
        for (const UiRect &rect : layout.sketch_tool_buttons) {
            if (rect.contains(app.mouse_x, app.mouse_y)) {
                return true;
            }
        }

        if (app.measure_width_rect.contains(app.mouse_x, app.mouse_y) ||
            app.measure_height_rect.contains(app.mouse_x, app.mouse_y)) {
            return true;
        }
    }

    if (app.show_hierarchy) {
        if (layout.hierarchy_root_rect.contains(app.mouse_x, app.mouse_y)) return true;
        for (const UiRect &row : layout.hierarchy_row_rects)
            if (row.contains(app.mouse_x, app.mouse_y)) return true;
        for (const UiRect &eye : layout.hierarchy_eye_rects)
            if (eye.contains(app.mouse_x, app.mouse_y)) return true;
    }

    if (app.open_menu >= 0) {
        return true;
    }

    return false;
}

void set_status(AppState &app, std::string text)
{
    app.status = std::move(text);
}

bool has_step_extension(const std::string &path)
{
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }

    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".step" || ext == ".stp";
}

std::string trim_whitespace_copy(std::string s)
{
    const std::size_t start = s.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return "";
    }
    const std::size_t end = s.find_last_not_of(" \t\r\n");
    return s.substr(start, end - start + 1);
}

void import_step_into_scene(AppState &app, const std::string &path)
{
    const std::string cleaned = trim_whitespace_copy(path);
    if (cleaned.empty()) {
        set_status(app, "STEP IMPORT CANCELED");
        return;
    }
    if (!has_step_extension(cleaned)) {
        set_status(app, "OPEN FAILED — EXPECTED .STEP OR .STP");
        return;
    }

    std::string error;
    if (app.renderer.import_step_file(cleaned, error)) {
        set_status(app, "STEP IMPORTED — " + cleaned);
    } else {
        set_status(app, "STEP IMPORT FAILED — " + error);
    }
}

void reset_measure_state(AppState &app)
{
    app.measure_focus = AppState::MeasureFocus::None;
    app.measure_editing = false;
    app.measure_width_locked = false;
    app.measure_height_locked = false;
    app.measure_width_m = 0.0f;
    app.measure_height_m = 0.0f;
    app.measure_input.clear();
    app.measure_width_rect = {};
    app.measure_height_rect = {};
}

bool current_preview_dimensions(const AppState &app, float &out_width_m, float &out_height_m)
{
    Renderer::SketchRectangle rect_local;
    if (!app.renderer.sketch_preview_rectangle_local(rect_local)) {
        return false;
    }

    out_width_m = std::fabs(rect_local.max_corner.x - rect_local.min_corner.x);
    out_height_m = std::fabs(rect_local.max_corner.y - rect_local.min_corner.y);
    return true;
}

void apply_locked_rectangle_dimensions(AppState &app)
{
    if (app.tool_mode != ToolMode::Sketch || app.renderer.sketch_tool() != Renderer::SketchTool::Rectangle) {
        return;
    }

    float width_m = 0.0f;
    float height_m = 0.0f;
    if (!current_preview_dimensions(app, width_m, height_m)) {
        return;
    }

    if (app.measure_width_locked) {
        width_m = app.measure_width_m;
    }
    if (app.measure_height_locked) {
        height_m = app.measure_height_m;
    }

    app.renderer.set_sketch_preview_rectangle_dims(width_m, height_m);
}

void begin_rectangle_dimension_entry(AppState &app)
{
    reset_measure_state(app);

    float width_m = 0.0f;
    float height_m = 0.0f;
    if (current_preview_dimensions(app, width_m, height_m)) {
        app.measure_width_m = width_m;
        app.measure_height_m = height_m;
    }

    app.measure_editing = true;
    app.measure_focus = AppState::MeasureFocus::Width;
}

bool commit_active_rectangle_dimension(AppState &app)
{
    float width_m = 0.0f;
    float height_m = 0.0f;
    if (!current_preview_dimensions(app, width_m, height_m)) {
        return false;
    }

    float committed_value_m = 0.0f;
    if (!app.measure_input.empty()) {
        try {
            committed_value_m = std::stof(app.measure_input) / 1000.0f;
        } catch (...) {
            return false;
        }
    }

    if (app.measure_focus == AppState::MeasureFocus::Width) {
        if (app.measure_input.empty()) {
            committed_value_m = width_m;
        }
        app.measure_width_locked = true;
        app.measure_width_m = committed_value_m;
        app.renderer.set_sketch_preview_rectangle_dims(app.measure_width_m, height_m);
        app.measure_focus = AppState::MeasureFocus::Height;
        app.measure_input.clear();
        app.measure_editing = true;
        return true;
    }

    if (app.measure_focus == AppState::MeasureFocus::Height) {
        if (app.measure_input.empty()) {
            committed_value_m = height_m;
        }
        app.measure_height_locked = true;
        app.measure_height_m = committed_value_m;
        app.renderer.set_sketch_preview_rectangle_dims(app.measure_width_locked ? app.measure_width_m : width_m,
                                                       app.measure_height_m);
        const bool committed = app.renderer.commit_sketch_preview();
        if (committed) {
            set_status(app, "RECTANGLE CREATED");
        }
        reset_measure_state(app);
        return committed;
    }

    return false;
}

std::string primitive_status_label(const Renderer::PrimitiveHit &hit)
{
    if (!hit.valid()) {
        return "SELECTION CLEARED";
    }

    if (hit.type == Renderer::PrimitiveType::Face) {
        return "FACE " + std::to_string(hit.index + 1) + " SELECTED";
    }

    if (hit.type == Renderer::PrimitiveType::Edge) {
        return "EDGE " + std::to_string(hit.index + 1) + " SELECTED";
    }

    return "SELECTION CLEARED";
}

std::string sketch_status_label(const AppState &app)
{
    std::string label = std::string("SKETCH — ") + app.renderer.sketch_tool_name();
    if (app.renderer.sketch_has_preview()) {
        label += " ACTIVE";
    }
    return label;
}

std::string sketch_constraint_label(const AppState &app)
{
    return std::string("CONSTRAINT — ") + app.renderer.sketch_constraint_hint();
}

void exit_sketch_mode(AppState &app)
{
    app.renderer.end_sketch();
    app.tool_mode = ToolMode::None;
    reset_measure_state(app);
    set_status(app, "SKETCH MODE EXITED");
}

// Forward declaration
static void compute_az_el_from_offset(const glm::vec3 &offset, float &out_az, float &out_el);

void enter_sketch_mode(AppState &app)
{
    app.renderer.begin_sketch(app.renderer.selected_hit());
    app.tool_mode = ToolMode::Sketch;
    reset_measure_state(app);

    const auto &selected = app.renderer.selected_hit();
    if (selected.valid() && selected.type == Renderer::PrimitiveType::Face) {
        set_status(app, "SKETCH MODE — FACE PLANE");
    } else {
        set_status(app, "SKETCH MODE — DEFAULT XY PLANE");
    }

    // Start a smooth camera animation to face the sketch plane
    {
        const double now = glfwGetTime();

        glm::vec3 start_eye = app.renderer.camera.eye();
        glm::vec3 start_tgt = app.renderer.camera.target();
        glm::vec3 start_offset = start_eye - start_tgt;
        float start_dist = app.renderer.camera.distance();
        float s_az=0.0f, s_el=0.0f;
        compute_az_el_from_offset(start_offset, s_az, s_el);

        glm::vec3 end_tgt;
        glm::vec3 end_normal;
        if (selected.valid()) {
            end_tgt = app.renderer.selection_center();
            end_normal = selected.normal;
        } else {
            const auto &plane = app.renderer.sketch_plane();
            end_tgt = plane.origin;
            end_normal = plane.normal;
        }

        // Ensure normal is unit length
        if (glm::length(end_normal) > 1e-6f) {
            end_normal = glm::normalize(end_normal);
        }

        float end_dist = start_dist;
        // Position the eye along the face normal so the face faces the camera.
        // Choose the sign so the final eye stays on the same hemisphere as the
        // current camera to avoid rotating through the object.
        glm::vec3 start_offset_norm = glm::normalize(start_offset);
        float side = (glm::dot(end_normal, start_offset_norm) >= 0.0f) ? 1.0f : -1.0f;
        glm::vec3 end_eye = end_tgt + (end_normal * end_dist * side);
        glm::vec3 end_offset = end_eye - end_tgt;
        float e_az=0.0f, e_el=0.0f;
        compute_az_el_from_offset(end_offset, e_az, e_el);

        app.camera_anim.active = true;
        app.camera_anim.start_time = now;
        app.camera_anim.duration = 0.4; // 400 ms

        app.camera_anim.start_target = start_tgt;
        app.camera_anim.end_target = end_tgt;
        app.camera_anim.start_distance = start_dist;
        app.camera_anim.end_distance = end_dist;
        app.camera_anim.start_az = s_az;
        app.camera_anim.end_az = e_az;
        app.camera_anim.start_el = s_el;
        app.camera_anim.end_el = e_el;
    }

    app.renderer.update_sketch_cursor(app.mouse_x, app.mouse_y);
}

// Helper: compute azimuth/elevation (degrees) from eye->target offset
static void compute_az_el_from_offset(const glm::vec3 &offset, float &out_az, float &out_el)
{
    const float dist = glm::length(offset);
    if (dist <= 1e-6f) {
        out_az = 0.0f; out_el = 0.0f; return;
    }
    out_el = static_cast<float>(std::asin(offset.y / dist) * 180.0 / M_PI);
    out_az = static_cast<float>(std::atan2(offset.x, offset.z) * 180.0 / M_PI);
}

void set_sketch_tool(AppState &app, Renderer::SketchTool tool)
{
    app.renderer.set_sketch_tool(tool);
    if (tool != Renderer::SketchTool::Rectangle) {
        reset_measure_state(app);
    }
    set_status(app, std::string("SKETCH TOOL — ") + app.renderer.sketch_tool_name());
}

void apply_viewport_selection_click(AppState &app)
{
    app.renderer.update_hovered(app.mouse_x, app.mouse_y);
    if (app.renderer.hovered_hit().valid()) {
        app.renderer.select_hovered();
        set_status(app, primitive_status_label(app.renderer.selected_hit()));
    } else {
        app.renderer.clear_selection();
        set_status(app, "SELECTION CLEARED");
    }
}

void toggle_tool_mode(AppState &app, ToolMode mode)
{
    if (app.tool_mode == mode) {
        if (mode == ToolMode::Sketch) {
            exit_sketch_mode(app);
        }
        return;
    }

    if (mode == ToolMode::Sketch) {
        enter_sketch_mode(app);
    }
}

void toggle_wireframe(AppState &app)
{
    app.wireframe = !app.wireframe;
    app.renderer.set_wireframe(app.wireframe);
    set_status(app, app.wireframe ? "WIREFRAME ENABLED" : "SOLID MODE ENABLED");
}

void reset_camera(AppState &app)
{
    app.renderer.camera.reset();
    set_status(app, "CAMERA RESET");
}

void toggle_projection_mode(AppState &app)
{
    if (app.renderer.is_isometric()) {
        app.renderer.set_projection_mode(Renderer::ProjectionMode::Perspective);
        set_status(app, "PERSPECTIVE CAMERA");
    } else {
        app.renderer.set_projection_mode(Renderer::ProjectionMode::Isometric);
        set_status(app, "ISOMETRIC CAMERA");
    }
}

void handle_toolbar_action(AppState &app, int button_index)
{
    app.open_menu = -1;

    switch (button_index) {
    case 0:
        reset_camera(app);
        break;
    case 1:
        toggle_wireframe(app);
        break;
    case 2:
        toggle_projection_mode(app);
        break;
    case 3:
        toggle_tool_mode(app, ToolMode::Sketch);
        break;
    default:
        break;
    }
}

void handle_menu_action(AppState &app, int menu_index, int item_index)
{
    app.open_menu = -1;

    switch (menu_index) {
    case 0:
        switch (item_index) {
        case 0:
            app.renderer.clear_imported_model();
            set_status(app, "NEW SCENE");
            break;
        case 1: {
            // Lightweight fallback input flow until a native file picker is added.
            std::printf("Enter STEP path (.step/.stp): ");
            std::fflush(stdout);

            char buffer[2048] = {};
            if (std::fgets(buffer, sizeof(buffer), stdin) == nullptr) {
                set_status(app, "STEP IMPORT CANCELED");
                break;
            }
            import_step_into_scene(app, std::string(buffer));
            break;
        }
        case 2: set_status(app, "SAVE NOT IMPLEMENTED"); break;
        case 3: set_status(app, "SAVE AS NOT IMPLEMENTED"); break;
        case 4: glfwSetWindowShouldClose(app.window, GLFW_TRUE); break;
        default: break;
        }
        break;
    case 1:
        switch (item_index) {
        case 0: set_status(app, "UNDO NOT IMPLEMENTED"); break;
        case 1: set_status(app, "REDO NOT IMPLEMENTED"); break;
        case 2: set_status(app, "PREFERENCES NOT IMPLEMENTED"); break;
        default: break;
        }
        break;
    case 2:
        switch (item_index) {
        case 0: reset_camera(app); break;
        case 1: toggle_wireframe(app); break;
        case 2: toggle_projection_mode(app); break;
        case 3:
            app.show_hierarchy = !app.show_hierarchy;
            set_status(app, app.show_hierarchy ? "MODEL HIERARCHY SHOWN" : "MODEL HIERARCHY HIDDEN");
            break;
        default: break;
        }
        break;
    case 3:
        if (item_index == 0) {
            set_status(app, "VOID CAD CUSTOM UI BUILD");
        }
        break;
    default:
        break;
    }
}

void drop_callback(GLFWwindow *window, int path_count, const char **paths)
{
    auto *app = static_cast<AppState *>(glfwGetWindowUserPointer(window));
    if (app == nullptr || path_count <= 0 || paths == nullptr) {
        return;
    }

    import_step_into_scene(*app, paths[0]);
}

bool handle_left_press(AppState &app)
{
    const UiLayout layout = build_ui_layout(app);

    if (app.tool_mode == ToolMode::Sketch && app.renderer.sketch_tool() == Renderer::SketchTool::Rectangle) {
        if (app.measure_width_rect.contains(app.mouse_x, app.mouse_y) ||
            app.measure_height_rect.contains(app.mouse_x, app.mouse_y)) {
            app.measure_editing = true;
            app.measure_input.clear();
            app.measure_focus = app.measure_width_rect.contains(app.mouse_x, app.mouse_y)
                ? AppState::MeasureFocus::Width
                : AppState::MeasureFocus::Height;
            return true;
        }
    }

    if (app.open_menu >= 0) {
        for (std::size_t i = 0; i < layout.open_menu_items.size(); ++i) {
            if (layout.open_menu_items[i].contains(app.mouse_x, app.mouse_y)) {
                handle_menu_action(app, app.open_menu, static_cast<int>(i));
                return true;
            }
        }
    }

    for (std::size_t i = 0; i < layout.menu_buttons.size(); ++i) {
        if (layout.menu_buttons[i].contains(app.mouse_x, app.mouse_y)) {
            const int menu_index = static_cast<int>(i);
            app.open_menu = (app.open_menu == menu_index) ? -1 : menu_index;
            return true;
        }
    }

    for (std::size_t i = 0; i < layout.toolbar_buttons.size(); ++i) {
        if (layout.toolbar_buttons[i].contains(app.mouse_x, app.mouse_y)) {
            handle_toolbar_action(app, static_cast<int>(i));
            return true;
        }
    }

    if (app.show_hierarchy && app.renderer.has_imported_model()) {
        const auto &bodies = app.renderer.imported_bodies();
        const std::size_t row_count = std::min(layout.hierarchy_eye_rects.size(), bodies.size());
        for (std::size_t i = 0; i < row_count; ++i) {
            if (layout.hierarchy_eye_rects[i].contains(app.mouse_x, app.mouse_y)) {
                if (app.renderer.toggle_imported_body_visible(i)) {
                    const auto &updated = app.renderer.imported_bodies()[i];
                    set_status(app, updated.visible
                        ? ("BODY SHOWN — " + updated.label)
                        : ("BODY HIDDEN — " + updated.label));
                }
                return true;
            }
        }
    }

    if (app.tool_mode == ToolMode::Sketch) {
        for (std::size_t i = 0; i < layout.sketch_tool_buttons.size(); ++i) {
            if (layout.sketch_tool_buttons[i].contains(app.mouse_x, app.mouse_y)) {
                switch (i) {
                case 0: set_sketch_tool(app, Renderer::SketchTool::Line); break;
                case 1: set_sketch_tool(app, Renderer::SketchTool::Rectangle); break;
                case 2: set_sketch_tool(app, Renderer::SketchTool::Circle); break;
                default: break;
                }
                return true;
            }
        }
    }

    if (app.open_menu >= 0) {
        app.open_menu = -1;
        return true;
    }

    return false;
}

void draw_button(UiRenderer &ui,
                 const UiRect &rect,
                 std::string_view label,
                 bool hovered,
                 bool active)
{
    UiColor fill = rgba(0.0f, 0.0f, 0.0f, 0.35f);
    UiColor outline = rgba(1.0f, 1.0f, 1.0f, 0.20f);
    UiColor text_color = rgba(1.0f, 1.0f, 1.0f, 1.0f);

    if (active) {
        fill = rgba(1.0f, 1.0f, 1.0f, 0.92f);
        outline = rgba(1.0f, 1.0f, 1.0f, 1.0f);
        text_color = rgba(0.0f, 0.0f, 0.0f, 1.0f);
    } else if (hovered) {
        fill = rgba(0.0f, 0.0f, 0.0f, 0.75f);
        outline = rgba(1.0f, 1.0f, 1.0f, 0.45f);
    }

    ui.filled_rect(rect, fill);
    ui.outline_rect(rect, 1.0f, outline);

        const float text_width = ui.measure_text(label, kTextScale);
        const float text_height = ui.line_height(kTextScale);
    ui.text(rect.x + (rect.w - text_width) * 0.5f,
            rect.y + (rect.h - text_height) * 0.5f,
            label,
            kTextScale,
            text_color);
}

void draw_menu_item(UiRenderer &ui,
                    const UiRect &rect,
                    std::string_view label,
                    bool hovered)
{
    const UiColor fill = hovered
        ? rgba(1.0f, 1.0f, 1.0f, 0.18f)
        : rgba(0.0f, 0.0f, 0.0f, 0.0f);

    ui.filled_rect(rect, fill);

    const float text_height = ui.line_height(kTextScale);
    ui.text(rect.x + 10.0f,
            rect.y + (rect.h - text_height) * 0.5f,
            label,
            kTextScale,
            rgba(1.0f, 1.0f, 1.0f, 1.0f));
}

UiRect badge_rect(const UiRenderer &ui, float x, float y, std::string_view label)
{
    return {
        x,
        y,
        ui.measure_text(label, kTextScale) + (kBadgePaddingX * 2.0f),
        kBadgeHeight,
    };
}

void draw_badge(UiRenderer &ui, const UiRect &rect, std::string_view label)
{
    ui.filled_rect(rect, rgba(0.0f, 0.0f, 0.0f, 0.65f));
    ui.outline_rect(rect, 1.0f, rgba(1.0f, 1.0f, 1.0f, 0.25f));

    const float text_height = ui.line_height(kTextScale);
    ui.text(rect.x + kBadgePaddingX,
            rect.y + (rect.h - text_height) * 0.5f,
            label,
            kTextScale,
            rgba(1.0f, 1.0f, 1.0f, 1.0f));
}

std::string fps_label(const AppState &app)
{
    char buffer[32] = {};
    std::snprintf(buffer, sizeof(buffer), "FPS %.0f", app.fps);
    return buffer;
}

void render_ui(AppState &app)
{
    const UiLayout layout = build_ui_layout(app);
    app.ui.begin_frame(app.window_width, app.window_height);

    for (std::size_t i = 0; i < layout.menu_buttons.size(); ++i) {
        draw_button(app.ui,
                    layout.menu_buttons[i],
                    kMenuTitles[i],
                    layout.menu_buttons[i].contains(app.mouse_x, app.mouse_y),
                    app.open_menu == static_cast<int>(i));
    }

    draw_button(app.ui,
                layout.toolbar_buttons[0],
                kToolbarLabels[0],
                layout.toolbar_buttons[0].contains(app.mouse_x, app.mouse_y),
                false);
    draw_button(app.ui,
                layout.toolbar_buttons[1],
                kToolbarLabels[1],
                layout.toolbar_buttons[1].contains(app.mouse_x, app.mouse_y),
                app.wireframe);
    draw_button(app.ui,
                layout.toolbar_buttons[2],
                kToolbarLabels[2],
                layout.toolbar_buttons[2].contains(app.mouse_x, app.mouse_y),
                app.renderer.is_isometric());
    draw_button(app.ui,
                layout.toolbar_buttons[3],
                kToolbarLabels[3],
                layout.toolbar_buttons[3].contains(app.mouse_x, app.mouse_y),
                app.tool_mode == ToolMode::Sketch);

    if (app.show_hierarchy && app.renderer.has_imported_model()) {
        const auto &bodies = app.renderer.imported_bodies();
        app.ui.text(layout.hierarchy_root_rect.x,
                    layout.hierarchy_root_rect.y,
                    "BODIES",
                    kTextScale,
                    rgba(1.0f, 1.0f, 1.0f, 0.95f));

        const std::size_t row_count = std::min(layout.hierarchy_row_rects.size(), bodies.size());
        for (std::size_t i = 0; i < row_count; ++i) {
            const auto &body = bodies[i];
            const UiRect &row = layout.hierarchy_row_rects[i];
            const UiRect &eye = layout.hierarchy_eye_rects[i];
            const bool eye_hover = eye.contains(app.mouse_x, app.mouse_y);
            const bool row_hover = row.contains(app.mouse_x, app.mouse_y);

            app.ui.filled_rect(eye, eye_hover
                ? rgba(1.0f, 1.0f, 1.0f, 0.18f)
                : rgba(0.0f, 0.0f, 0.0f, 0.22f));
            app.ui.outline_rect(eye, 1.0f, body.visible
                ? rgba(1.0f, 1.0f, 1.0f, 0.60f)
                : rgba(0.60f, 0.60f, 0.60f, 0.50f));
            app.ui.text(eye.x + 6.0f,
                        eye.y + 2.0f,
                        body.visible ? "o" : "x",
                        kTextScale,
                        body.visible ? rgba(1.0f, 1.0f, 1.0f, 0.95f) : rgba(0.65f, 0.65f, 0.65f, 0.95f));

            std::string label = body.label + "  [F:" + std::to_string(body.face_count)
                              + " E:" + std::to_string(body.edge_count) + "]";
            app.ui.text(eye.x + eye.w + 8.0f,
                        row.y + 2.0f,
                        label,
                        kTextScale,
                        body.visible
                            ? (row_hover ? rgba(1.0f, 1.0f, 1.0f, 1.0f) : rgba(0.88f, 0.88f, 0.88f, 0.95f))
                            : rgba(0.58f, 0.58f, 0.58f, 0.95f));
        }

        if (row_count < bodies.size()) {
            const float y = layout.hierarchy_root_rect.y
                          + (kHierarchyRowHeight + kHierarchyRowGap) * static_cast<float>(row_count + 1);
            app.ui.text(layout.hierarchy_root_rect.x + 12.0f,
                        y,
                        "...",
                        kTextScale,
                        rgba(0.70f, 0.70f, 0.70f, 0.9f));
        }
    }

    if (app.open_menu >= 0) {
        app.ui.filled_rect(layout.open_menu_panel, rgba(0.0f, 0.0f, 0.0f, 0.92f));
        app.ui.outline_rect(layout.open_menu_panel, 1.0f, rgba(1.0f, 1.0f, 1.0f, 0.35f));

        const auto &items = *kMenuItems[static_cast<std::size_t>(app.open_menu)];
        for (std::size_t i = 0; i < items.size(); ++i) {
            draw_menu_item(app.ui,
                           layout.open_menu_items[i],
                           items[i],
                           layout.open_menu_items[i].contains(app.mouse_x, app.mouse_y));
        }
    }

    const UiRect status = badge_rect(app.ui,
                                     kOuterMargin,
                                     static_cast<float>(app.window_height) - kOuterMargin - kBadgeHeight,
                                     app.status);
    const std::string fps = fps_label(app);
    const float fps_width = badge_rect(app.ui, 0.0f, 0.0f, fps).w;
    const UiRect fps_badge = badge_rect(app.ui,
                                        static_cast<float>(app.window_width) - kOuterMargin - fps_width,
                                        static_cast<float>(app.window_height) - kOuterMargin - kBadgeHeight,
                                        fps);

    draw_badge(app.ui, status, app.status);
    draw_badge(app.ui, fps_badge, fps);

    if (app.tool_mode == ToolMode::Sketch) {
        // Draw custom cursor crosshair at fixed screen size
        const float cursor_x = static_cast<float>(app.mouse_x);
        const float cursor_y = static_cast<float>(app.mouse_y);
        const float crosshair_size = 10.0f; // pixels
        const UiColor cursor_color = rgba(0.8f, 0.8f, 0.8f, 1.0f);
        app.ui.outline_rect({cursor_x - crosshair_size, cursor_y - 2.0f, crosshair_size * 2.0f, 4.0f}, 1.0f, cursor_color);
        app.ui.outline_rect({cursor_x - 2.0f, cursor_y - crosshair_size, 4.0f, crosshair_size * 2.0f}, 1.0f, cursor_color);

        // Left-side sketch badges removed per UX request.
        for (std::size_t i = 0; i < layout.sketch_tool_buttons.size(); ++i) {
            Renderer::SketchTool tool = Renderer::SketchTool::Line;
            if (i == 1) tool = Renderer::SketchTool::Rectangle;
            if (i == 2) tool = Renderer::SketchTool::Circle;
            draw_button(app.ui,
                        layout.sketch_tool_buttons[i],
                        kSketchToolLabels[i],
                        layout.sketch_tool_buttons[i].contains(app.mouse_x, app.mouse_y),
                        app.renderer.sketch_tool() == tool);
        }

        // If we're sketching a rectangle and a preview exists, show left/right
        // dimension badges (in mm) near the preview for quick entry.
        if (app.renderer.sketch_tool() == Renderer::SketchTool::Rectangle && app.renderer.sketch_has_preview()) {
            Renderer::SketchRectangle rect_local;
            if (app.renderer.sketch_preview_rectangle_local(rect_local)) {
                const glm::vec2 center_local((rect_local.min_corner.x + rect_local.max_corner.x) * 0.5f,
                                              (rect_local.min_corner.y + rect_local.max_corner.y) * 0.5f);
                const glm::vec2 top_local(center_local.x, rect_local.min_corner.y);
                const glm::vec2 right_local(rect_local.max_corner.x, center_local.y);

                glm::vec2 top_screen, right_screen;
                if (app.renderer.sketch_local_to_screen(top_local, top_screen) &&
                    app.renderer.sketch_local_to_screen(right_local, right_screen)) {

                    const float width = std::fabs(rect_local.max_corner.x - rect_local.min_corner.x);
                    const float height = std::fabs(rect_local.max_corner.y - rect_local.min_corner.y);
                    const int mm_w = static_cast<int>(std::round(width * 1000.0f));
                    const int mm_h = static_cast<int>(std::round(height * 1000.0f));
                    const std::string label_w = std::to_string(mm_w) + " mm";
                    const std::string label_h = std::to_string(mm_h) + " mm";

                    const UiRect width_badge = badge_rect(app.ui, top_screen.x - 40.0f, top_screen.y - 32.0f, label_w);
                    const UiRect height_badge = badge_rect(app.ui, right_screen.x + 8.0f, right_screen.y - 12.0f, label_h);
                    app.measure_width_rect = width_badge;
                    app.measure_height_rect = height_badge;
                    if (!app.measure_width_locked) {
                        app.measure_width_m = width;
                    }
                    if (!app.measure_height_locked) {
                        app.measure_height_m = height;
                    }

                    if (app.measure_editing && app.measure_focus == AppState::MeasureFocus::Width) {
                        draw_badge(app.ui, width_badge, app.measure_input.empty() ? label_w : app.measure_input + " mm");
                    } else {
                        draw_badge(app.ui, width_badge, label_w);
                    }

                    if (app.measure_editing && app.measure_focus == AppState::MeasureFocus::Height) {
                        draw_badge(app.ui, height_badge, app.measure_input.empty() ? label_h : app.measure_input + " mm");
                    } else {
                        draw_badge(app.ui, height_badge, label_h);
                    }
                }
            }
        }
    }

    app.ui.flush();
}

void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    auto *app = static_cast<AppState *>(glfwGetWindowUserPointer(window));
    if (app == nullptr) {
        return;
    }

    app->framebuffer_width = width;
    app->framebuffer_height = height;
    glViewport(0, 0, width, height);
    app->renderer.set_viewport(width, height);
}

void window_size_callback(GLFWwindow *window, int width, int height)
{
    auto *app = static_cast<AppState *>(glfwGetWindowUserPointer(window));
    if (app == nullptr) {
        return;
    }

    app->window_width = width;
    app->window_height = height;
}

void cursor_position_callback(GLFWwindow *window, double xpos, double ypos)
{
    auto *app = static_cast<AppState *>(glfwGetWindowUserPointer(window));
    if (app == nullptr) {
        return;
    }

    app->mouse_x = xpos;
    app->mouse_y = ypos;

    const double dx = xpos - app->last_drag_x;
    const double dy = ypos - app->last_drag_y;

    if (app->left_down && !app->left_captured_by_ui && (std::fabs(dx) > 0.5 || std::fabs(dy) > 0.5)) {
        app->left_dragged = true;
    }

    if (app->middle_down && !app->middle_captured_by_ui && (std::fabs(dx) > 0.5 || std::fabs(dy) > 0.5)) {
        app->middle_dragged = true;
    }

    if (app->right_down && (std::fabs(dx) > 0.5 || std::fabs(dy) > 0.5)) {
        app->right_dragged = true;
    }

    if (app->middle_down) {
        app->renderer.camera.pan(static_cast<float>(dx), static_cast<float>(dy));
    } else if (app->right_down) {
        app->renderer.camera.orbit(static_cast<float>(dx) * 0.3f,
                                   static_cast<float>(-dy) * 0.3f);
    }

    if (app->tool_mode == ToolMode::Sketch) {
        if (point_hits_ui(*app)) {
            app->renderer.clear_hovered();
        } else {
            app->renderer.update_sketch_cursor(xpos, ypos);
            apply_locked_rectangle_dimensions(*app);
        }
    } else {
        if (point_hits_ui(*app) || app->left_down || app->middle_down || app->right_down) {
            app->renderer.clear_hovered();
        } else {
            app->renderer.update_hovered(xpos, ypos);
        }
    }

    app->last_drag_x = xpos;
    app->last_drag_y = ypos;
}

void mouse_button_callback(GLFWwindow *window, int button, int action, int /*mods*/)
{
    auto *app = static_cast<AppState *>(glfwGetWindowUserPointer(window));
    if (app == nullptr) {
        return;
    }

    glfwGetCursorPos(window, &app->mouse_x, &app->mouse_y);
    app->last_drag_x = app->mouse_x;
    app->last_drag_y = app->mouse_y;

    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            app->left_down = true;
            app->left_dragged = false;
            app->left_press_x = app->mouse_x;
            app->left_press_y = app->mouse_y;
            app->left_captured_by_ui = handle_left_press(*app);
        } else if (action == GLFW_RELEASE) {
            if (!app->left_captured_by_ui && !point_hits_ui(*app)) {
                if (app->tool_mode == ToolMode::Sketch) {
                    if (app->renderer.handle_sketch_click(app->mouse_x, app->mouse_y)) {
                        set_status(*app, sketch_status_label(*app));
                        if (app->renderer.sketch_tool() == Renderer::SketchTool::Rectangle && app->renderer.sketch_has_preview()) {
                            begin_rectangle_dimension_entry(*app);
                        } else if (!app->renderer.sketch_has_preview()) {
                            reset_measure_state(*app);
                        }
                    }
                } else {
                    apply_viewport_selection_click(*app);
                }
            }
            app->left_down = false;
            app->left_captured_by_ui = false;
            app->left_dragged = false;
        }
    }

    if (button == GLFW_MOUSE_BUTTON_MIDDLE) {
        if (action == GLFW_PRESS) {
            app->middle_down = true;
            app->middle_dragged = false;
            app->middle_captured_by_ui = point_hits_ui(*app);
        } else if (action == GLFW_RELEASE) {
            if (!app->middle_captured_by_ui && !app->middle_dragged && !point_hits_ui(*app)) {
                const double now = glfwGetTime();
                const double click_dx = app->mouse_x - app->last_middle_click_x;
                const double click_dy = app->mouse_y - app->last_middle_click_y;
                const double click_dist = std::sqrt((click_dx * click_dx) + (click_dy * click_dy));
                const bool is_double_click =
                    app->last_middle_click_time >= 0.0 &&
                    (now - app->last_middle_click_time) <= kDoubleClickTimeSeconds &&
                    click_dist <= kDoubleClickDistancePx;

                if (is_double_click) {
                    if (app->renderer.set_pivot_from_click(app->mouse_x, app->mouse_y)) {
                        set_status(*app, "ROTATION PIVOT SET");
                    } else {
                        set_status(*app, "NO VALID PIVOT TARGET");
                    }
                    app->last_middle_click_time = -1.0;
                } else {
                    app->last_middle_click_time = now;
                    app->last_middle_click_x = app->mouse_x;
                    app->last_middle_click_y = app->mouse_y;
                }
            }

            app->middle_down = false;
            app->middle_captured_by_ui = false;
            app->middle_dragged = false;
        }
    }

    if (button == GLFW_MOUSE_BUTTON_RIGHT) {
        if (action == GLFW_PRESS) {
            app->right_down = !point_hits_ui(*app);
            app->right_dragged = false;
        } else if (action == GLFW_RELEASE) {
            if (app->tool_mode == ToolMode::Sketch && app->renderer.sketch_has_preview() &&
                !point_hits_ui(*app) && !app->right_dragged) {
                app->renderer.cancel_sketch_preview();
                reset_measure_state(*app);
                set_status(*app, "SKETCH PREVIEW CANCELED");
            }
            app->right_down = false;
            app->right_dragged = false;
        }
    }
}

void scroll_callback(GLFWwindow *window, double /*xoffset*/, double yoffset)
{
    auto *app = static_cast<AppState *>(glfwGetWindowUserPointer(window));
    if (app == nullptr || point_hits_ui(*app)) {
        return;
    }

    // If hovering the hierarchy, scroll the body list instead of zooming the camera.
    if (app->show_hierarchy && app->renderer.has_imported_model()) {
        const UiLayout layout = build_ui_layout(*app);
        if (layout.hierarchy_root_rect.contains(app->mouse_x, app->mouse_y)) {
            // Compute scroll limits
            const auto &bodies = app->renderer.imported_bodies();
            const float row_pitch = kHierarchyRowHeight + kHierarchyRowGap;
            const float y = kOuterMargin + kHierarchyTopOffset;
            const float available_bottom = static_cast<float>(app->window_height) - kBadgeHeight - kOuterMargin - 8.0f;
            const float total_height = row_pitch * static_cast<float>(bodies.size() + 1);
            const float available_space = std::max(0.0f, available_bottom - y);
            const float max_scroll = std::max(0.0f, total_height - available_space);

            const float scroll_step = 18.0f; // pixels per wheel notch
            app->hierarchy_scroll += static_cast<float>(-yoffset) * scroll_step;
            if (app->hierarchy_scroll < 0.0f) app->hierarchy_scroll = 0.0f;
            if (app->hierarchy_scroll > max_scroll) app->hierarchy_scroll = max_scroll;
            return;
        }
    }

    if (point_hits_ui(*app)) return;

    app->renderer.camera.zoom(static_cast<float>(yoffset) * 0.5f);
    if (app->tool_mode == ToolMode::Sketch) {
        app->renderer.update_sketch_cursor(app->mouse_x, app->mouse_y);
    }
}

void key_callback(GLFWwindow *window, int key, int /*scancode*/, int action, int /*mods*/)
{
    auto *app = static_cast<AppState *>(glfwGetWindowUserPointer(window));
    if (app == nullptr || action != GLFW_PRESS) {
        return;
    }

    // Rectangle measurement entry is handled as a focused state machine.
    if (app->measure_editing) {
        if (key == GLFW_KEY_TAB) {
            commit_active_rectangle_dimension(*app);
            return;
        }
        if (key == GLFW_KEY_BACKSPACE) {
            if (!app->measure_input.empty()) app->measure_input.pop_back();
            return;
        }
        if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) {
            commit_active_rectangle_dimension(*app);
            return;
        }

        if (key == GLFW_KEY_ESCAPE) {
            reset_measure_state(*app);
            return;
        }

        return;
    }

    if (key == GLFW_KEY_ESCAPE && app->open_menu >= 0) {
        app->open_menu = -1;
        set_status(*app, "MENU CLOSED");
        return;
    }

    if (key == GLFW_KEY_ESCAPE && app->tool_mode == ToolMode::Sketch) {
        if (app->renderer.sketch_has_preview()) {
            app->renderer.cancel_sketch_preview();
            reset_measure_state(*app);
            set_status(*app, "SKETCH PREVIEW CANCELED");
        } else {
            exit_sketch_mode(*app);
        }
        return;
    }

    if (app->tool_mode == ToolMode::Sketch) {
        if (key == GLFW_KEY_L) {
            set_sketch_tool(*app, Renderer::SketchTool::Line);
        } else if (key == GLFW_KEY_R) {
            set_sketch_tool(*app, Renderer::SketchTool::Rectangle);
        } else if (key == GLFW_KEY_C) {
            set_sketch_tool(*app, Renderer::SketchTool::Circle);
        }
    }
}


// Character input callback for numeric entry
static void char_input_callback(GLFWwindow *window, unsigned int codepoint)
{
    auto *app = static_cast<AppState *>(glfwGetWindowUserPointer(window));
    if (app == nullptr) return;
    if (!app->measure_editing) return;
    if (codepoint < 32 || codepoint > 126) return;
    const char c = static_cast<char>(codepoint);
    if ((c >= '0' && c <= '9') || c == '.' || c == '-') {
        app->measure_input.push_back(c);
    }
}
} // namespace

int main()
{
    if (!glfwInit()) {
        std::fprintf(stderr, "Failed to initialize GLFW.\n");
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow *window = glfwCreateWindow(1280, 720, "Void CAD", nullptr, nullptr);
    if (window == nullptr) {
        std::fprintf(stderr, "Failed to create GLFW window.\n");
        glfwTerminate();
        return 1;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);

    {
        AppState app;
        app.window = window;
        glfwSetWindowUserPointer(window, &app);

        glfwGetFramebufferSize(window, &app.framebuffer_width, &app.framebuffer_height);
        glfwGetWindowSize(window, &app.window_width, &app.window_height);

        glViewport(0, 0, app.framebuffer_width, app.framebuffer_height);
        app.renderer.init();
        app.renderer.set_viewport(app.framebuffer_width, app.framebuffer_height);
        app.renderer.set_wireframe(app.wireframe);
        app.ui.init();

        glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
        glfwSetWindowSizeCallback(window, window_size_callback);
        glfwSetCursorPosCallback(window, cursor_position_callback);
        glfwSetMouseButtonCallback(window, mouse_button_callback);
        glfwSetScrollCallback(window, scroll_callback);
        glfwSetKeyCallback(window, key_callback);
        glfwSetCharCallback(window, char_input_callback);
        glfwSetDropCallback(window, drop_callback);

        double previous_time = glfwGetTime();

        while (!glfwWindowShouldClose(window)) {
            const double now = glfwGetTime();
            const double dt = now - previous_time;
            previous_time = now;

            app.frame_counter += 1;
            app.fps_accumulator += dt;
            if (app.fps_accumulator >= 0.5) {
                app.fps = static_cast<double>(app.frame_counter) / app.fps_accumulator;
                app.frame_counter = 0;
                app.fps_accumulator = 0.0;
            }

            glfwPollEvents();

            // Update camera animation if active
            if (app.camera_anim.active) {
                const double now = glfwGetTime();
                double t = (now - app.camera_anim.start_time) / app.camera_anim.duration;
                if (t >= 1.0) t = 1.0;

                auto lerp_f = [](float a, float b, double t) { return static_cast<float>(a + (b - a) * t); };
                auto lerp_v3 = [](const glm::vec3 &a, const glm::vec3 &b, double t) {
                    return a + static_cast<float>(t) * (b - a);
                };

                float diff = app.camera_anim.end_az - app.camera_anim.start_az;
                while (diff > 180.0f) diff -= 360.0f;
                while (diff < -180.0f) diff += 360.0f;
                float az = app.camera_anim.start_az + diff * static_cast<float>(t);
                float el = lerp_f(app.camera_anim.start_el, app.camera_anim.end_el, t);
                float dist = lerp_f(app.camera_anim.start_distance, app.camera_anim.end_distance, t);
                glm::vec3 tgt = lerp_v3(app.camera_anim.start_target, app.camera_anim.end_target, t);

                app.renderer.camera.set_state(tgt, dist, az, el);

                if (t >= 1.0) app.camera_anim.active = false;
            }

            app.renderer.draw();
            render_ui(app);

            glfwSwapBuffers(window);
        }

        glfwSetWindowUserPointer(window, nullptr);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
