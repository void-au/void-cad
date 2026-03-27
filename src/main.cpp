#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
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
constexpr double kDoubleClickTimeSeconds = 0.30;
constexpr double kDoubleClickDistancePx = 8.0;

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
    ToolMode tool_mode = ToolMode::None;

    std::string status = "CUSTOM UI READY";
    double fps = 0.0;
    int frame_counter = 0;
    double fps_accumulator = 0.0;
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
    "RESET CAMERA", "WIREFRAME", "ISO/PERSP"
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
    }

    return false;
}

void set_status(AppState &app, std::string text)
{
    app.status = std::move(text);
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
    set_status(app, "SKETCH MODE EXITED");
}

void enter_sketch_mode(AppState &app)
{
    app.renderer.begin_sketch(app.renderer.selected_hit());
    app.tool_mode = ToolMode::Sketch;

    const auto &selected = app.renderer.selected_hit();
    if (selected.valid() && selected.type == Renderer::PrimitiveType::Face) {
        set_status(app, "SKETCH MODE — FACE PLANE");
    } else {
        set_status(app, "SKETCH MODE — DEFAULT XY PLANE");
    }

    app.renderer.update_sketch_cursor(app.mouse_x, app.mouse_y);
}

void set_sketch_tool(AppState &app, Renderer::SketchTool tool)
{
    app.renderer.set_sketch_tool(tool);
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
        case 0: set_status(app, "NEW FILE NOT IMPLEMENTED"); break;
        case 1: set_status(app, "OPEN NOT IMPLEMENTED"); break;
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

bool handle_left_press(AppState &app)
{
    const UiLayout layout = build_ui_layout(app);

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
        const UiRect sketch_badge = badge_rect(app.ui,
                                               kOuterMargin,
                                               kOuterMargin + kButtonHeight + 10.0f,
                                               sketch_status_label(app));
        draw_badge(app.ui, sketch_badge, sketch_status_label(app));

        const UiRect constraint_badge = badge_rect(app.ui,
                                                   kOuterMargin,
                                                   sketch_badge.y + sketch_badge.h + 6.0f,
                                                   sketch_constraint_label(app));
        draw_badge(app.ui, constraint_badge, sketch_constraint_label(app));

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

    if (key == GLFW_KEY_ESCAPE && app->open_menu >= 0) {
        app->open_menu = -1;
        set_status(*app, "MENU CLOSED");
        return;
    }

    if (key == GLFW_KEY_ESCAPE && app->tool_mode == ToolMode::Sketch) {
        if (app->renderer.sketch_has_preview()) {
            app->renderer.cancel_sketch_preview();
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
