# void-cad

> A free, open-source 3D CAD modelling application built with C++, OpenGL, GLFW, and a fully custom UI layer.

---

## Overview

Void-CAD is a lightweight, cross-platform CAD modelling tool written in modern C++. It uses **GLFW** for the window and input layer, **OpenGL 3.3** for rendering, a **fully custom-drawn UI** for toolbars and menus, and **GLM** for vector/matrix math. The goal is a clean, focused CAD environment — no bloat, no proprietary lock-in.

---

## Milestones

- [x] 3D Viewer — orbit, pan, and zoom via mouse
- [x] Toolbar
- [] Face / edge selection
- [] Sketch mode


## Building

### Prerequisites

Run the dependency helper to install required system libraries (GLFW, Epoxy, FreeType, GLM, OpenGL):

```bash
./deps.sh
```

The custom UI renders text with **Ubuntu Mono** when available (`UbuntuMono-R.ttf`); it falls back to an internal bitmap font if the TTF is missing.

### Compile

```bash
./build.sh
```

The compiled binary is written to `build/void-cad`.

---


## License

Released under the [MIT License](LICENSE). Copyright © 2026 Void Labs.

---

## Credits

Created by **Void Labs**  
Author: **Matt Boan**
