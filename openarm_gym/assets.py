# Copyright 2026 Enactic, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Locate the scene files, and the arm meshes they pull in.

This repository owns the tray scene and nothing else about the robot. The arm
itself -- its meshes, its pedestal, its joint limits -- is Enactic's
``openarm_mujoco``, vendored as a git submodule under ``third_party/`` rather
than copied, so it stays identifiably theirs and stays updatable.

``scenes/tray_grasp_scene.xml`` therefore reaches across into the submodule for
both its mesh directory and the arm model it attaches, and MuJoCo resolves both
relative to the scene file itself. The consequence worth knowing: **the
submodule has to be checked out** or model loading fails with a missing-file
error that does not mention submodules at all. :func:`scene_path` raises a
clearer one first.
"""

from __future__ import annotations

from pathlib import Path

#: Repository root, three levels up from this file when installed in place.
ROOT = Path(__file__).resolve().parent.parent
SCENES = ROOT / "scenes"
SUBMODULE = ROOT / "third_party" / "openarm_mujoco"

#: The scene everything in this repository runs on.
TRAY_SCENE = "tray_grasp_scene.xml"


def scene_path(name: str = TRAY_SCENE) -> str:
    """Return an absolute path to a scene file, checking its assets exist.

    Args:
        name: file name inside ``scenes/``.

    Returns:
        The absolute path, as a string, which is what MuJoCo's loader takes.

    Raises:
        FileNotFoundError: if the scene is missing, or if the arm submodule has
            not been checked out -- the second is much the more likely, and
            MuJoCo's own error for it names a mesh file rather than the cause.

    """
    path = SCENES / name
    if not path.exists():
        raise FileNotFoundError(f"no scene named {name} in {SCENES}")
    if not (SUBMODULE / "v2" / "assets").is_dir():
        raise FileNotFoundError(
            f"the arm assets are missing from {SUBMODULE}. This repository "
            "vendors enactic/openarm_mujoco as a submodule; run\n"
            "    git submodule update --init --recursive"
        )
    return str(path)
