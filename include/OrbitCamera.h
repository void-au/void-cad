#pragma once
#include <glm/glm.hpp>

class OrbitCamera {
public:
    // target     – world-space point the camera orbits around
    // distance   – radius from target
    // azimuth    – horizontal angle in degrees (0 = +Z axis)
    // elevation  – vertical angle in degrees, clamped to [-89, 89]
    OrbitCamera(glm::vec3 target   = {0.0f, 0.0f, 0.0f},
                float     distance  = 4.0f,
                float     azimuth   = 45.0f,
                float     elevation = 30.0f);

    // Derive the view matrix from current spherical coordinates
    glm::mat4 view_matrix() const;

    // Returns the current eye position in world space
    glm::vec3 eye() const;

    // --- Mutation (called by mouse/scroll handlers) ---
    void orbit(float delta_azimuth, float delta_elevation);
    void pan(float delta_x, float delta_y);   // in view-plane units
    void zoom(float delta);                    // positive = closer
    void set_target(const glm::vec3 &target);
    void reset();

    // Accessors
    float     distance()  const { return m_distance; }
    glm::vec3 target()    const { return m_target; }

    // Set full camera state (target, distance, angles)
    void set_state(const glm::vec3 &target, float distance,
                   float azimuth, float elevation);

    // Set angles / distance independently
    void set_angles(float azimuth, float elevation);
    void set_distance(float distance);

private:
    glm::vec3 m_target;
    float     m_distance;
    float     m_azimuth;    // degrees
    float     m_elevation;  // degrees

    glm::vec3 m_default_target;
    float     m_default_distance;
    float     m_default_azimuth;
    float     m_default_elevation;
};
