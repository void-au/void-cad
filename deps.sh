#!/usr/bin/env bash
# deps.sh — check and install required system libraries for Void-CAD
set -e

MISSING=()

check() {
    local pkg="$1"
    if pkg-config --exists "$pkg" 2>/dev/null; then
        echo "[ok] $pkg"
    else
        echo "[missing] $pkg"
        MISSING+=("$pkg")
    fi
}

echo "=== Checking dependencies ==="
check "glfw3"
check "epoxy"
check "freetype2"

# GLM is header-only; check the include path directly
if [ -f /usr/include/glm/glm.hpp ] || [ -f /usr/local/include/glm/glm.hpp ]; then
    echo "[ok] glm (header-only)"
else
    echo "[missing] glm"
    MISSING+=("libglm-dev")
fi

if [ ${#MISSING[@]} -eq 0 ]; then
    echo ""
    echo "All dependencies satisfied."
    exit 0
fi

echo ""
echo "=== Installing missing packages ==="

if command -v apt-get &>/dev/null; then
    # Map pkg-config names to apt package names
    APT_PKGS=()
    for pkg in "${MISSING[@]}"; do
        case "$pkg" in
            glfw3)      APT_PKGS+=("libglfw3-dev") ;;
            epoxy)      APT_PKGS+=("libepoxy-dev") ;;
            freetype2)  APT_PKGS+=("libfreetype-dev") ;;
            libglm-dev) APT_PKGS+=("libglm-dev") ;;
        esac
    done
    sudo apt-get update -qq
    sudo apt-get install -y "${APT_PKGS[@]}"
elif command -v pacman &>/dev/null; then
    PACMAN_PKGS=()
    for pkg in "${MISSING[@]}"; do
        case "$pkg" in
            glfw3)      PACMAN_PKGS+=("glfw") ;;
            epoxy)      PACMAN_PKGS+=("libepoxy") ;;
            freetype2)  PACMAN_PKGS+=("freetype2") ;;
            libglm-dev) PACMAN_PKGS+=("glm") ;;
        esac
    done
    sudo pacman -S --needed "${PACMAN_PKGS[@]}"
elif command -v dnf &>/dev/null; then
    DNF_PKGS=()
    for pkg in "${MISSING[@]}"; do
        case "$pkg" in
            glfw3)      DNF_PKGS+=("glfw-devel") ;;
            epoxy)      DNF_PKGS+=("libepoxy-devel") ;;
            freetype2)  DNF_PKGS+=("freetype-devel") ;;
            libglm-dev) DNF_PKGS+=("glm-devel") ;;
        esac
    done
    sudo dnf install -y "${DNF_PKGS[@]}"
else
    echo "ERROR: Unknown package manager. Please install the following manually:"
    printf '  %s\n' "${MISSING[@]}"
    exit 1
fi

echo ""
echo "Dependencies installed successfully."
