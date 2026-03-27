#include "UiRenderer.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdio>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

namespace {

constexpr int GLYPH_WIDTH = 5;
constexpr int GLYPH_HEIGHT = 7;
constexpr int GLYPH_ADVANCE = 6;

std::array<unsigned char, GLYPH_HEIGHT> glyph_rows(char c)
{
    switch (static_cast<unsigned char>(std::toupper(static_cast<unsigned char>(c)))) {
    case 'A': return {0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'B': return {0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E};
    case 'C': return {0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E};
    case 'D': return {0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E};
    case 'E': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F};
    case 'F': return {0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10};
    case 'G': return {0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0E};
    case 'H': return {0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11};
    case 'I': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x1F};
    case 'J': return {0x07, 0x02, 0x02, 0x02, 0x12, 0x12, 0x0C};
    case 'K': return {0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11};
    case 'L': return {0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F};
    case 'M': return {0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11};
    case 'N': return {0x11, 0x19, 0x15, 0x13, 0x11, 0x11, 0x11};
    case 'O': return {0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'P': return {0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10};
    case 'Q': return {0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D};
    case 'R': return {0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11};
    case 'S': return {0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E};
    case 'T': return {0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04};
    case 'U': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E};
    case 'V': return {0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04};
    case 'W': return {0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A};
    case 'X': return {0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11};
    case 'Y': return {0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04};
    case 'Z': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F};
    case '0': return {0x0E, 0x11, 0x13, 0x15, 0x19, 0x11, 0x0E};
    case '1': return {0x04, 0x0C, 0x04, 0x04, 0x04, 0x04, 0x0E};
    case '2': return {0x0E, 0x11, 0x01, 0x02, 0x04, 0x08, 0x1F};
    case '3': return {0x1E, 0x01, 0x01, 0x0E, 0x01, 0x01, 0x1E};
    case '4': return {0x02, 0x06, 0x0A, 0x12, 0x1F, 0x02, 0x02};
    case '5': return {0x1F, 0x10, 0x10, 0x1E, 0x01, 0x01, 0x1E};
    case '6': return {0x0E, 0x10, 0x10, 0x1E, 0x11, 0x11, 0x0E};
    case '7': return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x08, 0x08};
    case '8': return {0x0E, 0x11, 0x11, 0x0E, 0x11, 0x11, 0x0E};
    case '9': return {0x0E, 0x11, 0x11, 0x0F, 0x01, 0x01, 0x0E};
    case '.': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C};
    case '-': return {0x00, 0x00, 0x00, 0x1F, 0x00, 0x00, 0x00};
    case '/': return {0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10};
    case ' ': return {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};
    default:  return {0x1F, 0x01, 0x02, 0x04, 0x08, 0x00, 0x08};
    }
}

} // namespace

UiRenderer::~UiRenderer()
{
    glDeleteBuffers(1, &m_vbo);
    glDeleteVertexArrays(1, &m_vao);
    glDeleteProgram(m_program);
}

GLuint UiRenderer::compile_shader(GLenum type, const char *source)
{
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, nullptr);
    glCompileShader(shader);

    GLint ok = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[512] = {};
        glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
        std::fprintf(stderr, "UI shader compile error: %s\n", log);
    }
    return shader;
}

GLuint UiRenderer::link_program(GLuint vertex_shader, GLuint fragment_shader)
{
    GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    GLint ok = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[512] = {};
        glGetProgramInfoLog(program, sizeof(log), nullptr, log);
        std::fprintf(stderr, "UI program link error: %s\n", log);
    }

    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);
    return program;
}

void UiRenderer::init()
{
    static const char *VERT_SRC = R"glsl(#version 330 core
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec4 aColor;
uniform mat4 uProjection;
out vec4 vColor;
void main()
{
    gl_Position = uProjection * vec4(aPos, 0.0, 1.0);
    vColor = aColor;
}
)glsl";

    static const char *FRAG_SRC = R"glsl(#version 330 core
in vec4 vColor;
out vec4 FragColor;
void main()
{
    FragColor = vColor;
}
)glsl";

    GLuint vertex_shader = compile_shader(GL_VERTEX_SHADER, VERT_SRC);
    GLuint fragment_shader = compile_shader(GL_FRAGMENT_SHADER, FRAG_SRC);
    m_program = link_program(vertex_shader, fragment_shader);
    m_u_projection = glGetUniformLocation(m_program, "uProjection");

    glGenVertexArrays(1, &m_vao);
    glGenBuffers(1, &m_vbo);

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(0));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex), reinterpret_cast<void *>(2 * sizeof(float)));
    glEnableVertexAttribArray(1);
    glBindVertexArray(0);
}

void UiRenderer::begin_frame(int width, int height)
{
    m_width = width;
    m_height = height;
    m_vertices.clear();
}

void UiRenderer::push_triangle(const Vertex &a, const Vertex &b, const Vertex &c)
{
    m_vertices.push_back(a);
    m_vertices.push_back(b);
    m_vertices.push_back(c);
}

void UiRenderer::filled_rect(const UiRect &rect, UiColor color)
{
    Vertex a{rect.x, rect.y, color.r, color.g, color.b, color.a};
    Vertex b{rect.x + rect.w, rect.y, color.r, color.g, color.b, color.a};
    Vertex c{rect.x + rect.w, rect.y + rect.h, color.r, color.g, color.b, color.a};
    Vertex d{rect.x, rect.y + rect.h, color.r, color.g, color.b, color.a};
    push_triangle(a, b, c);
    push_triangle(a, c, d);
}

void UiRenderer::outline_rect(const UiRect &rect, float thickness, UiColor color)
{
    filled_rect({rect.x, rect.y, rect.w, thickness}, color);
    filled_rect({rect.x, rect.y + rect.h - thickness, rect.w, thickness}, color);
    filled_rect({rect.x, rect.y + thickness, thickness, rect.h - (2.0f * thickness)}, color);
    filled_rect({rect.x + rect.w - thickness, rect.y + thickness, thickness, rect.h - (2.0f * thickness)}, color);
}

float UiRenderer::measure_text(std::string_view text, float scale)
{
    if (text.empty()) {
        return 0.0f;
    }
    return static_cast<float>((GLYPH_ADVANCE * static_cast<int>(text.size())) - 1) * scale;
}

float UiRenderer::line_height(float scale)
{
    return static_cast<float>(GLYPH_HEIGHT + 1) * scale;
}

void UiRenderer::text(float x, float y, std::string_view text, float scale, UiColor color)
{
    float cursor_x = x;
    for (char c : text) {
        const auto rows = glyph_rows(c);
        for (int row = 0; row < GLYPH_HEIGHT; ++row) {
            for (int col = 0; col < GLYPH_WIDTH; ++col) {
                const unsigned char bit = static_cast<unsigned char>(1u << (GLYPH_WIDTH - 1 - col));
                if ((rows[row] & bit) == 0) {
                    continue;
                }
                filled_rect({
                    cursor_x + static_cast<float>(col) * scale,
                    y + static_cast<float>(row) * scale,
                    scale,
                    scale,
                }, color);
            }
        }
        cursor_x += static_cast<float>(GLYPH_ADVANCE) * scale;
    }
}

void UiRenderer::flush()
{
    if (m_vertices.empty() || m_width <= 0 || m_height <= 0) {
        return;
    }

    const glm::mat4 projection = glm::ortho(0.0f,
                                            static_cast<float>(m_width),
                                            static_cast<float>(m_height),
                                            0.0f,
                                            -1.0f,
                                            1.0f);

    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glUseProgram(m_program);
    glUniformMatrix4fv(m_u_projection, 1, GL_FALSE, glm::value_ptr(projection));

    glBindVertexArray(m_vao);
    glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(m_vertices.size() * sizeof(Vertex)),
                 m_vertices.data(),
                 GL_DYNAMIC_DRAW);
    glDrawArrays(GL_TRIANGLES, 0, static_cast<GLsizei>(m_vertices.size()));
    glBindVertexArray(0);

    glEnable(GL_DEPTH_TEST);
}
