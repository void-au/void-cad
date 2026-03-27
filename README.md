# void-cad

> A free, open-source 3D CAD modelling application built with C++, OpenGL, GLFW, and a fully custom UI layer.

---

## Overview

Void-CAD is a lightweight, cross-platform CAD modelling tool written in modern C++. It uses **GLFW** for the window and input layer, **OpenGL 3.3** for rendering, a **fully custom-drawn UI** for toolbars and menus, and **GLM** for vector/matrix math. The goal is a clean, focused CAD environment — no bloat, no proprietary lock-in.

---

## Milestones

- [x] 3D Viewer — orbit, pan, and zoom via mouse
- [x] Toolbar — custom top toolbar and viewport controls
- [x] Face / edge selection — hover, select, and pivot targeting
- [x] Sketch mode foundations — sketch plane, grid, line / rectangle / circle tools
- [x] Basic sketch constraints — horizontal / vertical locking and radius-aware circle workflow
- [ ] Persistent sketch editing — selection/editing of existing sketch entities
- [ ] Profile detection — identify closed sketch regions for solid features
- [ ] Extrude — turn closed sketch profiles into 3D solids
- [ ] Cut / subtract extrude — remove material from existing bodies
- [ ] Radius / fillet — round selected edges
- [ ] Chamfer — bevel selected edges
- [ ] Model tree / feature tree — hierarchical view of bodies, sketches, and operations
- [ ] Parametric history editing — update earlier features and regenerate downstream geometry
- [ ] Dimensions and advanced constraints — equal, coincident, concentric, tangent, numeric dimensions
- [ ] Multi-body workflow — support separate parts/bodies in a scene
- [ ] Boolean operations — union, difference, and intersection
- [ ] Save / load projects — persist scene, sketches, and feature history
- [ ] Import / export — STEP / STL / OBJ or similar interchange formats


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
