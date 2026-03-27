#include <gtk/gtk.h>
#include <epoxy/gl.h>
#include "Renderer.h"

// ---------------------------------------------------------------------------
// Per-drag state: tracks the previous offset so we can compute per-event
// deltas (GtkGestureDrag gives cumulative offsets from drag start).
// ---------------------------------------------------------------------------

struct DragState {
    Renderer *renderer = nullptr;
    double prev_x      = 0.0;
    double prev_y      = 0.0;
};

struct FpsState {
    GtkLabel *label       = nullptr;
    int       frame_count = 0;
    guint     timeout_id  = 0;
};

static gboolean fps_tick(gpointer user_data)
{
    auto *fps = static_cast<FpsState *>(user_data);
    char buf[32];
    g_snprintf(buf, sizeof(buf), "%d FPS", fps->frame_count * 2);
    gtk_label_set_text(fps->label, buf);
    fps->frame_count = 0;
    return G_SOURCE_CONTINUE;
}

// ---------------------------------------------------------------------------
// GtkGLArea signal handlers
// ---------------------------------------------------------------------------

static void on_realize(GtkGLArea *area, gpointer user_data)
{
    gtk_gl_area_make_current(area);
    if (gtk_gl_area_get_error(area) != nullptr)
        return;

    static_cast<Renderer *>(user_data)->init();
}

static gboolean on_render(GtkGLArea *area, GdkGLContext * /*ctx*/, gpointer user_data)
{
    static_cast<Renderer *>(user_data)->draw();

    auto *fps = static_cast<FpsState *>(g_object_get_data(G_OBJECT(area), "fps_state"));
    if (fps) fps->frame_count++;

    return TRUE;
}

static void on_resize(GtkGLArea *area, int width, int height, gpointer user_data)
{
    gtk_gl_area_make_current(area);
    glViewport(0, 0, width, height);
    static_cast<Renderer *>(user_data)->set_viewport(width, height);
}

// ---------------------------------------------------------------------------
// Rotate (left-button drag)
// ---------------------------------------------------------------------------

static void on_rotate_begin(GtkGestureDrag * /*gesture*/, double /*x*/, double /*y*/,
                            gpointer user_data)
{
    auto *state   = static_cast<DragState *>(user_data);
    state->prev_x = 0.0;
    state->prev_y = 0.0;
}

static void on_rotate_update(GtkGestureDrag *gesture, double offset_x, double offset_y,
                              gpointer user_data)
{
    auto *state = static_cast<DragState *>(user_data);
    double dx   = offset_x - state->prev_x;
    double dy   = offset_y - state->prev_y;
    state->prev_x = offset_x;
    state->prev_y = offset_y;

    // Scale: 0.3 degrees per pixel feels natural at default FOV
    state->renderer->camera.orbit(static_cast<float>(dx) * 0.3f,
                                  static_cast<float>(-dy) * 0.3f);

    GtkWidget *gl_area = gtk_event_controller_get_widget(
        GTK_EVENT_CONTROLLER(gesture));
    gtk_widget_queue_draw(gl_area);
}

// ---------------------------------------------------------------------------
// Pan (middle-button drag)
// ---------------------------------------------------------------------------

static void on_pan_begin(GtkGestureDrag * /*gesture*/, double /*x*/, double /*y*/,
                         gpointer user_data)
{
    auto *state   = static_cast<DragState *>(user_data);
    state->prev_x = 0.0;
    state->prev_y = 0.0;
}

static void on_pan_update(GtkGestureDrag *gesture, double offset_x, double offset_y,
                           gpointer user_data)
{
    auto *state = static_cast<DragState *>(user_data);
    double dx   = offset_x - state->prev_x;
    double dy   = offset_y - state->prev_y;
    state->prev_x = offset_x;
    state->prev_y = offset_y;

    state->renderer->camera.pan(static_cast<float>(dx),
                                static_cast<float>(dy));

    GtkWidget *gl_area = gtk_event_controller_get_widget(
        GTK_EVENT_CONTROLLER(gesture));
    gtk_widget_queue_draw(gl_area);
}

// ---------------------------------------------------------------------------
// Zoom (scroll wheel)
// ---------------------------------------------------------------------------

static gboolean on_scroll(GtkEventControllerScroll *ctrl,
                           double /*dx*/, double dy,
                           gpointer user_data)
{
    auto *state = static_cast<DragState *>(user_data);
    // dy > 0 = scroll down = zoom out; negate so scroll-up zooms in
    state->renderer->camera.zoom(static_cast<float>(-dy) * 0.5f);

    GtkWidget *gl_area = gtk_event_controller_get_widget(GTK_EVENT_CONTROLLER(ctrl));
    gtk_widget_queue_draw(gl_area);

    return TRUE;
}

// ---------------------------------------------------------------------------
// Application
// ---------------------------------------------------------------------------

static void on_activate(GtkApplication *app, gpointer /*user_data*/)
{
    GtkWidget *window = gtk_application_window_new(app);
    gtk_window_set_title(GTK_WINDOW(window), "Void CAD");
    gtk_window_set_default_size(GTK_WINDOW(window), 1280, 720);

    Renderer *renderer = new Renderer();
    g_object_set_data_full(G_OBJECT(window), "renderer", renderer,
                           [](gpointer p) { delete static_cast<Renderer *>(p); });

    // Create the OpenGL viewport
    GtkWidget *gl_area = gtk_gl_area_new();
    gtk_gl_area_set_required_version(GTK_GL_AREA(gl_area), 3, 3);
    gtk_gl_area_set_has_depth_buffer(GTK_GL_AREA(gl_area), TRUE);

    g_signal_connect(gl_area, "realize", G_CALLBACK(on_realize), renderer);
    g_signal_connect(gl_area, "render",  G_CALLBACK(on_render),  renderer);
    g_signal_connect(gl_area, "resize",  G_CALLBACK(on_resize),  renderer);

    // FPS counter
    auto *fps_state = new FpsState{};
    g_object_set_data_full(G_OBJECT(gl_area), "fps_state", fps_state,
                           [](gpointer p){
                               auto *s = static_cast<FpsState *>(p);
                               if (s->timeout_id) g_source_remove(s->timeout_id);
                               delete s;
                           });

    GtkCssProvider *css = gtk_css_provider_new();
    gtk_css_provider_load_from_string(css,
        ".fps-label { color: #ffffff; background: rgba(0,0,0,0.5);"
        " border-radius: 4px; padding: 2px 6px;"
        " font-size: 12px; font-family: monospace; }");
    gtk_style_context_add_provider_for_display(
        gdk_display_get_default(),
        GTK_STYLE_PROVIDER(css),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(css);

    GtkWidget *fps_label = gtk_label_new("-- FPS");
    gtk_widget_add_css_class(fps_label, "fps-label");
    gtk_widget_set_halign(fps_label, GTK_ALIGN_END);
    gtk_widget_set_valign(fps_label, GTK_ALIGN_END);
    gtk_widget_set_margin_end(fps_label, 10);
    gtk_widget_set_margin_bottom(fps_label, 10);
    fps_state->label      = GTK_LABEL(fps_label);
    fps_state->timeout_id = g_timeout_add(500, fps_tick, fps_state);

    GtkWidget *overlay = gtk_overlay_new();
    gtk_overlay_set_child(GTK_OVERLAY(overlay), gl_area);
    gtk_overlay_add_overlay(GTK_OVERLAY(overlay), fps_label);

    // Rotate gesture (LMB)
    auto *rotate_state = new DragState{renderer};
    g_object_set_data_full(G_OBJECT(window), "rotate_state", rotate_state,
                           [](gpointer p){ delete static_cast<DragState*>(p); });

    GtkGesture *rotate_drag = gtk_gesture_drag_new();
    gtk_gesture_single_set_button(GTK_GESTURE_SINGLE(rotate_drag), 1); // LMB
    g_signal_connect(rotate_drag, "drag-begin",  G_CALLBACK(on_rotate_begin),  rotate_state);
    g_signal_connect(rotate_drag, "drag-update", G_CALLBACK(on_rotate_update), rotate_state);
    gtk_widget_add_controller(gl_area, GTK_EVENT_CONTROLLER(rotate_drag));

    // Pan gesture (MMB)
    auto *pan_state = new DragState{renderer};
    g_object_set_data_full(G_OBJECT(window), "pan_state", pan_state,
                           [](gpointer p){ delete static_cast<DragState*>(p); });

    GtkGesture *pan_drag = gtk_gesture_drag_new();
    gtk_gesture_single_set_button(GTK_GESTURE_SINGLE(pan_drag), 2); // MMB
    g_signal_connect(pan_drag, "drag-begin",  G_CALLBACK(on_pan_begin),  pan_state);
    g_signal_connect(pan_drag, "drag-update", G_CALLBACK(on_pan_update), pan_state);
    gtk_widget_add_controller(gl_area, GTK_EVENT_CONTROLLER(pan_drag));

    // Scroll controller (zoom)
    auto *scroll_state = new DragState{renderer};
    g_object_set_data_full(G_OBJECT(window), "scroll_state", scroll_state,
                           [](gpointer p){ delete static_cast<DragState*>(p); });

    GtkEventController *scroll_ctrl =
        gtk_event_controller_scroll_new(GTK_EVENT_CONTROLLER_SCROLL_VERTICAL);
    g_signal_connect(scroll_ctrl, "scroll", G_CALLBACK(on_scroll), scroll_state);
    gtk_widget_add_controller(gl_area, scroll_ctrl);

    gtk_window_set_child(GTK_WINDOW(window), overlay);
    gtk_window_present(GTK_WINDOW(window));
}

int main(int argc, char *argv[])
{
    GtkApplication *app = gtk_application_new("org.voidcad.app", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(on_activate), nullptr);
    int status = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    return status;
}
