#pragma once

#include <array>
#include <string>
#include <string_view>
#include <vector>

#include <glm/glm.hpp>

#include "Renderer.h"
#include "UiRenderer.h"

struct GLFWwindow;

namespace app {

enum class ToolMode {
    None,
    Sketch,
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

namespace ui {
inline constexpr float kTextScale = 1.0f;
inline constexpr float kOuterMargin = 8.0f;
inline constexpr float kButtonHeight = 30.0f;
inline constexpr float kButtonPaddingX = 10.0f;
inline constexpr float kButtonGap = 6.0f;
inline constexpr float kMenuGap = 3.0f;
inline constexpr float kMenuPanelPadding = 6.0f;
inline constexpr float kBadgePaddingX = 8.0f;
inline constexpr float kBadgeHeight = 24.0f;
inline constexpr float kHierarchyTopOffset = 46.0f;
inline constexpr float kHierarchyWidth = 220.0f;
inline constexpr float kHierarchyRowHeight = 18.0f;
inline constexpr float kHierarchyRowGap = 2.0f;
inline constexpr float kHierarchyEyeWidth = 20.0f;
inline constexpr float kHierarchyPaddingX = 6.0f;
inline constexpr double kDoubleClickTimeSeconds = 0.30;
inline constexpr double kDoubleClickDistancePx = 8.0;

inline constexpr std::array<std::string_view, 4> kMenuTitles = {
    "FILE", "EDIT", "VIEW", "HELP"
};

inline const std::array<std::vector<std::string>, 4> kMenuItems = {
    std::vector<std::string>{"NEW", "OPEN", "SAVE", "SAVE AS", "QUIT"},
    std::vector<std::string>{"UNDO", "REDO", "PREFERENCES"},
    std::vector<std::string>{"RESET CAMERA", "WIREFRAME", "ISO/PERSP", "HIERARCHY"},
    std::vector<std::string>{"ABOUT"}
};

inline constexpr std::array<std::string_view, 4> kToolbarLabels = {
    "RESET CAMERA", "WIREFRAME", "ISO/PERSP", "SKETCH"
};

inline constexpr std::array<std::string_view, 3> kSketchToolLabels = {
    "LINE", "RECT", "CIRCLE"
};
} // namespace ui

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

    struct CameraAnim {
        bool active = false;
        double start_time = 0.0;
        double duration = 0.4;

        glm::vec3 start_target = glm::vec3(0.0f);
        glm::vec3 end_target = glm::vec3(0.0f);

        float start_distance = 4.0f;
        float end_distance = 4.0f;

        float start_az = 0.0f;
        float end_az = 0.0f;

        float start_el = 0.0f;
        float end_el = 0.0f;
    } camera_anim;

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

} // namespace app
