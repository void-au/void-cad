#include "OrbitCamera.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>

static constexpr float PI = 3.14159265358979f;

OrbitCamera::OrbitCamera(glm::vec3 target, float distance,
                         float azimuth, float elevation)
    : m_target(target)
    , m_distance(distance)
    , m_azimuth(azimuth)
    , m_elevation(elevation)
    , m_default_target(target)
    , m_default_distance(distance)
    , m_default_azimuth(azimuth)
    , m_default_elevation(elevation)
{}

glm::vec3 OrbitCamera::eye() const
{
    float az  = glm::radians(m_azimuth);
    float el  = glm::radians(m_elevation);
    float cos_el = std::cos(el);

    glm::vec3 offset(
        m_distance * cos_el * std::sin(az),
        m_distance * std::sin(el),
        m_distance * cos_el * std::cos(az)
    );
    return m_target + offset;
}

glm::mat4 OrbitCamera::view_matrix() const
{
    return glm::lookAt(eye(), m_target, glm::vec3(0.0f, 1.0f, 0.0f));
}

void OrbitCamera::orbit(float delta_az, float delta_el)
{
    m_azimuth   += delta_az;
    m_elevation  = std::clamp(m_elevation + delta_el, -89.0f, 89.0f);
}

void OrbitCamera::pan(float dx, float dy)
{
    // Build right and up vectors from the current view orientation
    glm::mat4 v    = view_matrix();
    glm::vec3 right(v[0][0], v[1][0], v[2][0]);
    glm::vec3 up   (v[0][1], v[1][1], v[2][1]);

    // Scale pan speed proportionally to distance so it feels consistent
    float speed = m_distance * 0.001f;
    m_target -= right * (dx * speed);
    m_target += up    * (dy * speed);
}

void OrbitCamera::zoom(float delta)
{
    m_distance = std::clamp(m_distance - delta, 0.5f, 500.0f);
}

void OrbitCamera::set_target(const glm::vec3 &target)
{
    m_target = target;
}

void OrbitCamera::reset()
{
    m_target = m_default_target;
    m_distance = m_default_distance;
    m_azimuth = m_default_azimuth;
    m_elevation = m_default_elevation;
}

void OrbitCamera::set_state(const glm::vec3 &target, float distance,
                           float azimuth, float elevation)
{
    m_target = target;
    m_distance = std::clamp(distance, 0.5f, 500.0f);
    m_azimuth = azimuth;
    m_elevation = std::clamp(elevation, -89.0f, 89.0f);
}

void OrbitCamera::set_angles(float azimuth, float elevation)
{
    m_azimuth = azimuth;
    m_elevation = std::clamp(elevation, -89.0f, 89.0f);
}

void OrbitCamera::set_distance(float distance)
{
    m_distance = std::clamp(distance, 0.5f, 500.0f);
}
