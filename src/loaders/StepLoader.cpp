#include "loaders/StepLoader.h"

#include <algorithm>
#include <cctype>

#include "Renderer.h"

namespace loaders {

namespace {

bool has_step_extension(const std::string &path)
{
    const std::size_t dot = path.find_last_of('.');
    if (dot == std::string::npos) {
        return false;
    }

    std::string ext = path.substr(dot);
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    return ext == ".step" || ext == ".stp";
}

} // namespace

const char *StepLoader::name() const
{
    return "STEP";
}

bool StepLoader::can_load(const std::string &path) const
{
    return has_step_extension(path);
}

bool StepLoader::import(Renderer &renderer,
                        const std::string &path,
                        std::string &error_message) const
{
    Renderer::PreparedImport prepared;
    if (!renderer.prepare_step_file_import(path, prepared, error_message, nullptr)) {
        return false;
    }
    return renderer.apply_prepared_step_import(std::move(prepared), error_message);
}

} // namespace loaders
