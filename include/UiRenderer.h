#pragma once

#include <epoxy/gl.h>
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

    static float measure_text(std::string_view text, float scale);
    static float line_height(float scale);

private:
    struct Vertex {
        float x;
        float y;
        float r;
        float g;
        float b;
        float a;
    };

    GLuint compile_shader(GLenum type, const char *source);
    GLuint link_program(GLuint vertex_shader, GLuint fragment_shader);
    void push_triangle(const Vertex &a, const Vertex &b, const Vertex &c);

    GLuint m_program = 0;
    GLuint m_vao = 0;
    GLuint m_vbo = 0;
    GLint  m_u_projection = -1;

    int m_width = 0;
    int m_height = 0;
    std::vector<Vertex> m_vertices;
};
