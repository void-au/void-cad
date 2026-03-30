#pragma once

#include "loaders/ModelLoader.h"

namespace loaders {

class StepLoader final : public ModelLoader {
public:
    const char *name() const override;
    bool can_load(const std::string &path) const override;
    bool import(Renderer &renderer,
                const std::string &path,
                std::string &error_message) const override;
};

} // namespace loaders
