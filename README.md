# void-cad

> A free, open-source 3D CAD modelling application built with C++, GTK4, and OpenGL.

---

## Overview

Void-CAD is a lightweight, cross-platform CAD modelling tool written in modern C++. It uses **GTK4** for the application shell and **OpenGL 3.3** (core profile) for the 3D viewport, with **GLM** handling all vector/matrix math. The goal is a clean, focused CAD environment — no bloat, no proprietary lock-in.

---

## Milestones

- [x] 3D Viewer — orbit, pan, and zoom via mouse
- [ ] Toolbar
- [ ] Face / edge selection
- [ ] Sketch mode


## Building

### Prerequisites

Run the dependency helper to install required system libraries (GTK4, GLEW/GLAD, GLM, OpenGL):

```bash
./deps.sh
```

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
