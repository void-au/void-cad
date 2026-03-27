#pragma once

#include <epoxy/gl.h>
#include <unordered_map>
#include <string_view>
#include <vector>

struct UiColor {
    float r;
    float g;
    float b;
    float a;
};

struct UiRect {
    float x;
    float y;
    float w;
    float h;

    bool contains(double px, double py) const
    {
        return px >= x && px <= (x + w) && py >= y && py <= (y + h);
    }
};

class UiRenderer {
public:
    UiRenderer() = default;
    ~UiRenderer();

    void init();
    void begin_frame(int width, int height);
    void filled_rect(const UiRect &rect, UiColor color);
    void outline_rect(const UiRect &rect, float thickness, UiColor color);
    void text(float x, float y, std::string_view text, float scale, UiColor color);
    void flush();

    float measure_text(std::string_view text, float scale) const;
    float line_height(float scale) const;

private:
    struct Vertex {
        float x;
        float y;
        float r;
        float g;
        float b;
        float a;
    };

    struct GlyphBitmap {
        int width = 0;
        int height = 0;
        int bearing_x = 0;
        int bearing_y = 0;
        float advance_px = 0.0f;
        std::vector<unsigned char> alpha;
    };

    GLuint compile_shader(GLenum type, const char *source);
    GLuint link_program(GLuint vertex_shader, GLuint fragment_shader);
    void push_triangle(const Vertex &a, const Vertex &b, const Vertex &c);
    bool load_ubuntu_mono();
    void draw_fallback_text(float x, float y, std::string_view text, float scale, UiColor color);

    GLuint m_program = 0;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLint  m_u_projection = -1;

    std::unordered_map<unsigned char, GlyphBitmap> m_glyphs;
    bool m_has_ubuntu_mono = false;
    float m_font_ascent = 0.0f;
    float m_font_line_height = 8.0f;
    float m_fallback_advance = 6.0f;

    int m_width = 0;
    int m_height = 0;
    std::vector<Vertex> m_vertices;
};
