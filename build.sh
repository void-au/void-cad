#!/usr/bin/env bash
# build.sh — compile and link Void-CAD
set -e

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -Wall -Wextra -Iinclude"
OUTDIR="build"
TARGET="$OUTDIR/void-cad"

# Gather pkg-config flags
GLFW_CFLAGS=$(pkg-config --cflags glfw3)
GLFW_LIBS=$(pkg-config --libs glfw3)
EPOXY_CFLAGS=$(pkg-config --cflags epoxy)
EPOXY_LIBS=$(pkg-config --libs epoxy)
FREETYPE_CFLAGS=$(pkg-config --cflags freetype2)
FREETYPE_LIBS=$(pkg-config --libs freetype2)

CFLAGS_ALL="$CXXFLAGS $GLFW_CFLAGS $EPOXY_CFLAGS $FREETYPE_CFLAGS"
LIBS_ALL="$GLFW_LIBS $EPOXY_LIBS $FREETYPE_LIBS"

# Collect all .cpp sources under src/
SRCS=$(find src -name '*.cpp')

if [ -z "$SRCS" ]; then
    echo "ERROR: No .cpp files found in src/"
    exit 1
fi

mkdir -p "$OUTDIR"

echo "=== Building Void-CAD ==="
echo "Sources: $SRCS"

# shellcheck disable=SC2086
"$CXX" $CFLAGS_ALL $SRCS -o "$TARGET" $LIBS_ALL

echo ""
echo "Build successful → $TARGET"
