#pragma once

#include <string>

class Renderer;

namespace loaders {

class ModelLoader {
public:
    virtual ~ModelLoader() = default;

    virtual const char *name() const = 0;
    virtual bool can_load(const std::string &path) const = 0;
    virtual bool import(Renderer &renderer,
                        const std::string &path,
                        std::string &error_message) const = 0;
};

} // namespace loaders
