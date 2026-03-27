#!/usr/bin/env bash
# build.sh — compile and link Void-CAD
set -e

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -Wall -Wextra -Iinclude"
OUTDIR="build"
TARGET="$OUTDIR/void-cad"

# Gather pkg-config flags
GTK_CFLAGS=$(pkg-config --cflags gtk4)
GTK_LIBS=$(pkg-config --libs gtk4)
# epoxy handles GL function loading and is already a GTK4 dependency
EPOXY_CFLAGS=$(pkg-config --cflags epoxy)
EPOXY_LIBS=$(pkg-config --libs epoxy)

CFLAGS_ALL="$CXXFLAGS $GTK_CFLAGS $EPOXY_CFLAGS"
LIBS_ALL="$GTK_LIBS $EPOXY_LIBS"

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
