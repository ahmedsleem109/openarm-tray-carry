// Copyright 2026 Enactic, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// A minimal MuJoCo-to-three.js view: build one three mesh per model geom, then
// each frame copy geom_xpos/geom_xmat across. It is deliberately small -- it
// draws what the simulation has, and nothing else. Anything cleverer would be a
// second source of truth about where the arm is.
//
// MuJoCo is z-up and three.js is y-up, so the whole scene is rotated once at
// the root rather than per object.

import * as THREE from "three";

const GEOM = {
  PLANE: 0,
  SPHERE: 2,
  CAPSULE: 3,
  ELLIPSOID: 4,
  CYLINDER: 5,
  BOX: 6,
  MESH: 7,
};

// A three.js geometry for one MuJoCo geom. Sizes follow MuJoCo's convention:
// half-extents for a box, radius for a sphere, (radius, half-length) for a
// cylinder or capsule.
function buildGeometry(model, index, size) {
  const type = model.geom_type[index];
  switch (type) {
    case GEOM.PLANE:
      // A finite stand-in: MuJoCo's plane is infinite, which three cannot draw.
      // Kept small, because a floor large enough to look infinite just fills
      // the frame with grey.
      return new THREE.PlaneGeometry(3.0, 3.0);
    case GEOM.SPHERE:
      return new THREE.SphereGeometry(size[0], 24, 16);
    case GEOM.CAPSULE:
      return new THREE.CapsuleGeometry(size[0], 2 * size[1], 8, 16);
    case GEOM.ELLIPSOID: {
      const geometry = new THREE.SphereGeometry(1, 24, 16);
      geometry.scale(size[0], size[1], size[2]);
      return geometry;
    }
    case GEOM.CYLINDER:
      return new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], 24);
    case GEOM.BOX:
      return new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    case GEOM.MESH:
      return buildMesh(model, model.geom_dataid[index]);
    default:
      return null;
  }
}

// Copy a mesh out of the model's own vertex and face arrays. The arm's links
// are all of this type, so without it the page draws a robot made of nothing.
function buildMesh(model, meshId) {
  if (meshId < 0) return null;
  const vertStart = model.mesh_vertadr[meshId];
  const vertCount = model.mesh_vertnum[meshId];
  const faceStart = model.mesh_faceadr[meshId];
  const faceCount = model.mesh_facenum[meshId];

  const positions = new Float32Array(vertCount * 3);
  positions.set(model.mesh_vert.subarray(vertStart * 3, (vertStart + vertCount) * 3));
  const indices = new Uint32Array(faceCount * 3);
  indices.set(model.mesh_face.subarray(faceStart * 3, (faceStart + faceCount) * 3));

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setIndex(new THREE.BufferAttribute(indices, 1));
  geometry.computeVertexNormals();
  return geometry;
}

function geomColor(model, index) {
  const materialId = model.geom_matid[index];
  const source = materialId >= 0 ? model.mat_rgba : model.geom_rgba;
  const offset = (materialId >= 0 ? materialId : index) * 4;
  return {
    color: new THREE.Color(source[offset], source[offset + 1], source[offset + 2]),
    opacity: source[offset + 3],
  };
}

// Cylinders and capsules are built along three's +Y but MuJoCo's +Z, so their
// geometry carries a fixed quarter turn. Everything else is identity.
function axisFix(model, index) {
  const type = model.geom_type[index];
  if (type === GEOM.CYLINDER || type === GEOM.CAPSULE) {
    return new THREE.Quaternion().setFromEuler(new THREE.Euler(Math.PI / 2, 0, 0));
  }
  if (type === GEOM.PLANE) {
    return new THREE.Quaternion().setFromEuler(new THREE.Euler(-Math.PI / 2, 0, 0));
  }
  return new THREE.Quaternion();
}

export class SceneView {
  constructor(canvas, model) {
    this.model = model;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x11131a);

    // One rotation for the whole world instead of one per object.
    this.root = new THREE.Group();
    this.root.rotation.x = -Math.PI / 2;
    this.scene.add(this.root);

    this.camera = new THREE.PerspectiveCamera(45, 16 / 9, 0.05, 50);
    this.setView(0.9);

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x333344, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.5);
    key.position.set(1.5, 2.5, 2.0);
    this.scene.add(key);

    this.meshes = [];
    for (let i = 0; i < model.ngeom; i++) {
      // MuJoCo's own viewer shows geom groups 0-2 and hides 3 and up, which is
      // where collision geometry lives -- drawn in the convention's translucent
      // red. Without this the arm renders as its collision capsules smeared
      // over its visual meshes, which is why it came out pink and mottled.
      if (model.geom_group[i] >= 3) {
        this.meshes.push(null);
        continue;
      }
      const size = model.geom_size.subarray(i * 3, i * 3 + 3);
      const geometry = buildGeometry(model, i, size);
      if (!geometry) {
        this.meshes.push(null);
        continue;
      }
      const { color, opacity } = geomColor(model, i);
      // The floor's material is a checker texture this renderer does not load,
      // so its flat colour comes out bright white. Dark grey reads as a floor.
      if (model.geom_type[i] === GEOM.PLANE) color.setRGB(0.13, 0.14, 0.17);
      const material = new THREE.MeshStandardMaterial({
        color,
        opacity,
        transparent: opacity < 1,
        roughness: 0.7,
        metalness: 0.05,
      });
      const mesh = new THREE.Mesh(geometry, material);
      mesh.userData.fix = axisFix(model, i);
      this.meshes.push(mesh);
      this.root.add(mesh);
    }
    this._quat = new THREE.Quaternion();
    this._mat = new THREE.Matrix4();
  }

  // Orbit the camera around the workspace. Everything here is computed in
  // MuJoCo's z-up world and converted once at the end, by the same rule the
  // root group applies: (x, y, z) -> (x, z, -y). Mixing the two conventions
  // halfway through is what put one of the two arms outside the frame.
  setView(azimuth, elevation = 0.30, distance = 1.15) {
    const a = azimuth * Math.PI * 2;
    const target = [0.36, 0.0, 0.52]; // the tray, once it is lifted
    const eye = [
      target[0] + distance * Math.cos(elevation) * Math.cos(a),
      target[1] + distance * Math.cos(elevation) * Math.sin(a),
      target[2] + distance * Math.sin(elevation),
    ];
    this.camera.position.set(eye[0], eye[2], -eye[1]);
    this.camera.lookAt(target[0], target[2], -target[1]);
  }

  resize(width, height) {
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  // Copy every geom's world pose out of MjData. MuJoCo stores a row-major 3x3;
  // three wants a column-major 4x4, hence the explicit transpose in `set`.
  sync(data) {
    for (let i = 0; i < this.meshes.length; i++) {
      const mesh = this.meshes[i];
      if (!mesh) continue;
      const p = data.geom_xpos.subarray(i * 3, i * 3 + 3);
      const m = data.geom_xmat.subarray(i * 9, i * 9 + 9);
      mesh.position.set(p[0], p[1], p[2]);
      this._mat.set(
        m[0], m[1], m[2], 0,
        m[3], m[4], m[5], 0,
        m[6], m[7], m[8], 0,
        0, 0, 0, 1,
      );
      this._quat.setFromRotationMatrix(this._mat);
      mesh.quaternion.copy(this._quat).multiply(mesh.userData.fix);
    }
  }

  draw() {
    this.renderer.render(this.scene, this.camera);
  }
}
