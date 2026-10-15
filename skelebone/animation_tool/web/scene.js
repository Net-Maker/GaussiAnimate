import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { TransformControls } from "three/addons/controls/TransformControls.js";

const TEAL = new THREE.Color("#17856c");
const GRAY = new THREE.Color("#606b65");
const DEG = Math.PI / 180;

function geometry(values, size = 3) {
  const result = new THREE.BufferGeometry();
  result.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(values), size),
  );
  return result;
}

function pointsMaterial(size, color, overlay = false) {
  const material = new THREE.PointsMaterial({
    size,
    color: color || 0xffffff,
    sizeAttenuation: false,
    vertexColors: !color,
    depthTest: !overlay,
    depthWrite: !overlay,
  });
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      "void main() {",
      "void main() { if (distance(gl_PointCoord, vec2(0.5)) > 0.5) discard;",
    );
  };
  return material;
}

export class AssetViewport {
  constructor(canvas, { editable = false, onPick, onRotate } = {}) {
    this.canvas = canvas;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("white");
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.001, 100);
    this.orbit = new OrbitControls(this.camera, canvas);
    this.orbit.enableDamping = true;
    this.content = new THREE.Group();
    this.scene.add(this.content);
    this.upAxis = "z";
    this.showSkeleton = true;
    this.pointSize = 2;
    this.colorMode = "appearance";
    this.selected = null;
    this.onRotate = onRotate;
    this.fitHeight = 1;
    this.lastSize = "";
    if (editable) {
      this.gizmo = new TransformControls(this.camera, canvas);
      this.gizmo.setMode("rotate");
      this.gizmo.setSpace("local");
      this.gizmo.setSize(0.65);
      this.scene.add(this.gizmo.getHelper());
      this.gizmo.addEventListener("dragging-changed", (event) => {
        this.orbit.enabled = !event.value;
      });
      this.gizmo.addEventListener("objectChange", () => {
        const joint = this.selected;
        if (joint === null || !this.bones) return;
        const offset = this.baseLocal[joint]
          .clone()
          .invert()
          .multiply(this.bones[joint].quaternion);
        const euler = new THREE.Euler().setFromQuaternion(offset, "ZYX");
        this.preview();
        onRotate?.(joint, [euler.x / DEG, euler.y / DEG, euler.z / DEG]);
      });
      let down = null;
      canvas.addEventListener("pointerdown", (event) => {
        down = { x: event.clientX, y: event.clientY, gizmo: !!this.gizmo.axis };
      });
      canvas.addEventListener("pointerup", (event) => {
        if (
          down &&
          !down.gizmo &&
          Math.hypot(event.clientX - down.x, event.clientY - down.y) < 5
        ) {
          const picked = this.pick(event.clientX, event.clientY);
          if (picked !== null) onPick?.(picked);
        }
        down = null;
      });
    }
    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas);
  }

  clear() {
    this.gizmo?.detach();
    this.content.traverse((object) => {
      object.geometry?.dispose();
      object.material?.dispose();
    });
    this.content.clear();
    this.bones = null;
    this.cloud = null;
    this.selected = null;
  }

  install(meta, points, joints, base) {
    this.clear();
    this.meta = meta;
    this.parents = meta.parents;
    this.upAxis = meta.up_axis;
    this.colors = new Float32Array(meta.colors);
    this.skinningColors = new Float32Array(meta.skinning_colors);
    this.cloud = new THREE.Points(
      geometry(points),
      pointsMaterial(this.pointSize),
    );
    this.cloud.geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(this.colors.slice(), 3),
    );
    this.edges = [];
    this.parents.forEach((parent, child) => {
      if (parent >= 0) this.edges.push(parent, child);
    });
    this.lines = new THREE.LineSegments(
      geometry(this.edges.length * 3),
      new THREE.LineBasicMaterial({
        vertexColors: true,
        depthTest: false,
        depthWrite: false,
      }),
    );
    this.lines.geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(new Float32Array(this.edges.length * 3), 3),
    );
    this.joints = new THREE.Points(
      geometry(joints),
      pointsMaterial(5, "#65736b", true),
    );
    this.marker = new THREE.Points(
      geometry(3),
      pointsMaterial(11, "#168268", true),
    );
    this.marker.visible = false;
    for (const object of [this.cloud, this.lines, this.joints, this.marker]) {
      object.frustumCulled = false;
      this.content.add(object);
    }
    this.lines.renderOrder = 2;
    this.joints.renderOrder = 3;
    this.marker.renderOrder = 4;
    if (this.gizmo) this.installRig(meta.rig, base);
    this.setGeometry(points, joints);
    this.setColorMode(this.colorMode);
    this.setSkeletonVisible(this.showSkeleton);
    this.fit();
  }

  installRig(rig, base) {
    this.basePoints = base.slice();
    this.bones = this.parents.map(() => new THREE.Bone());
    this.baseLocal = rig.local_rotations.map(
      ([w, x, y, z]) => new THREE.Quaternion(x, y, z, w),
    );
    this.parents.forEach((parent, joint) => {
      const bone = this.bones[joint];
      bone.quaternion.copy(this.baseLocal[joint]);
      bone.position.fromArray(rig.rest_joints[joint]);
      if (parent < 0) {
        bone.position.add(new THREE.Vector3(...rig.root_translation));
        this.content.add(bone);
      } else {
        bone.position.sub(new THREE.Vector3(...rig.rest_joints[parent]));
        this.bones[parent].add(bone);
      }
    });
    this.content.updateMatrixWorld(true);
    this.inverseBind = this.bones.map((bone) =>
      bone.matrixWorld.clone().invert(),
    );
    // Sparse influence lists keep live CPU skinning proportional to actual weights.
    this.influences = this.bones.map(() => []);
    const count = this.bones.length;
    for (let point = 0; point < this.meta.points; point++) {
      for (let joint = 0; joint < count; joint++) {
        const weight = rig.preview_weights[point * count + joint];
        if (weight > 0) this.influences[joint].push(point, weight);
      }
    }
  }

  setOffsets(offsets, preview = true) {
    if (!this.bones) return;
    this.bones.forEach((bone, joint) => {
      const angles = offsets[joint] || [0, 0, 0];
      const offset = new THREE.Quaternion().setFromEuler(
        new THREE.Euler(...angles.map((a) => a * DEG), "ZYX"),
      );
      bone.quaternion.copy(this.baseLocal[joint]).multiply(offset);
    });
    this.content.updateMatrixWorld(true);
    if (preview) this.preview();
  }

  preview() {
    this.content.updateMatrixWorld(true);
    const output = new Float32Array(this.basePoints.length);
    const joints = new Float32Array(this.bones.length * 3);
    const delta = new THREE.Matrix4();
    this.bones.forEach((bone, joint) => {
      joints.set(bone.matrixWorld.elements.slice(12, 15), joint * 3);
      const m = delta.multiplyMatrices(
        bone.matrixWorld,
        this.inverseBind[joint],
      ).elements;
      const list = this.influences[joint];
      for (let k = 0; k < list.length; k += 2) {
        const index = list[k] * 3,
          w = list[k + 1];
        const x = this.basePoints[index],
          y = this.basePoints[index + 1],
          z = this.basePoints[index + 2];
        output[index] += w * (m[0] * x + m[4] * y + m[8] * z + m[12]);
        output[index + 1] += w * (m[1] * x + m[5] * y + m[9] * z + m[13]);
        output[index + 2] += w * (m[2] * x + m[6] * y + m[10] * z + m[14]);
      }
    });
    this.setGeometry(output, joints);
  }

  setGeometry(points, joints) {
    if (!this.cloud) return;
    this.cloud.geometry.attributes.position.array.set(points);
    this.cloud.geometry.attributes.position.needsUpdate = true;
    this.joints.geometry.attributes.position.array.set(joints);
    this.joints.geometry.attributes.position.needsUpdate = true;
    const positions = this.lines.geometry.attributes.position;
    this.edges.forEach((joint, index) =>
      positions.array.set(joints.subarray(joint * 3, joint * 3 + 3), index * 3),
    );
    positions.needsUpdate = true;
    this.updateSelection();
  }

  select(joint) {
    this.selected = joint;
    if (this.gizmo && this.bones) {
      this.gizmo.attach(this.bones[joint]);
      this.gizmo.enabled = this.showSkeleton;
      this.gizmo.getHelper().visible = this.showSkeleton;
    }
    this.updateSelection();
  }

  updateSelection() {
    if (!this.cloud) return;
    const colors = this.lines.geometry.attributes.color;
    const affected = new Set();
    for (let j = 0; j < this.parents.length; j++) {
      let ancestor = j;
      while (ancestor >= 0 && ancestor !== this.selected)
        ancestor = this.parents[ancestor];
      if (ancestor === this.selected) affected.add(j);
    }
    for (let i = 0; i < this.edges.length; i += 2) {
      const color = affected.has(this.edges[i]) ? TEAL : GRAY;
      color.toArray(colors.array, i * 3);
      color.toArray(colors.array, (i + 1) * 3);
    }
    colors.needsUpdate = true;
    this.marker.visible = this.selected !== null && this.showSkeleton;
    if (this.selected !== null) {
      const p = this.joints.geometry.attributes.position.array;
      this.marker.geometry.attributes.position.array.set(
        p.subarray(this.selected * 3, this.selected * 3 + 3),
      );
      this.marker.geometry.attributes.position.needsUpdate = true;
    }
  }

  setSkeletonVisible(value) {
    this.showSkeleton = value;
    if (!this.cloud) return;
    this.lines.visible = value;
    this.joints.visible = value;
    if (this.gizmo) {
      this.gizmo.enabled = value;
      this.gizmo.getHelper().visible = value;
    }
    this.updateSelection();
  }
  setColorMode(mode) {
    this.colorMode = mode;
    if (!this.cloud) return;
    this.cloud.geometry.attributes.color.array.set(
      mode === "skinning" ? this.skinningColors : this.colors,
    );
    this.cloud.geometry.attributes.color.needsUpdate = true;
  }
  setPointSize(value) {
    this.pointSize = value;
    if (this.cloud) this.cloud.material.size = value;
  }

  fit() {
    if (!this.cloud) return;
    const box = new THREE.Box3().setFromBufferAttribute(
      this.cloud.geometry.attributes.position,
    );
    const center = box.getCenter(new THREE.Vector3()),
      size = box.getSize(new THREE.Vector3());
    const radius = Math.max(size.length() / 2, 0.01);
    const direction =
      this.upAxis === "z"
        ? new THREE.Vector3(1, -0.55, 0.4)
        : new THREE.Vector3(0.7, 0.45, 1);
    this.camera.up.set(
      0,
      this.upAxis === "y" ? 1 : 0,
      this.upAxis === "z" ? 1 : 0,
    );
    this.camera.position
      .copy(center)
      .addScaledVector(direction.normalize(), radius * 4);
    this.camera.lookAt(center);
    this.camera.near = radius * 0.01;
    this.camera.far = radius * 10;
    this.camera.zoom = 1;
    this.fitExtents = size.multiplyScalar(0.5);
    this.orbit.target.copy(center);
    this.orbit.update();
    this.resize(true);
  }
  resize(force = false) {
    const width = this.canvas.clientWidth,
      height = this.canvas.clientHeight;
    if (!width || !height) return;
    const key = `${width}/${height}`;
    if (key === this.lastSize && !force) return;
    this.lastSize = key;
    this.renderer.setSize(width, height, false);
    const aspect = width / height;
    if (this.fitExtents) {
      const right = new THREE.Vector3(1, 0, 0).applyQuaternion(
        this.camera.quaternion,
      );
      const up = new THREE.Vector3(0, 1, 0).applyQuaternion(
        this.camera.quaternion,
      );
      const span = (v) =>
        Math.abs(v.x) * this.fitExtents.x +
        Math.abs(v.y) * this.fitExtents.y +
        Math.abs(v.z) * this.fitExtents.z;
      this.fitHeight = Math.max(span(up), span(right) / aspect, 0.01) * 1.18;
    }
    this.camera.left = -this.fitHeight * aspect;
    this.camera.right = this.fitHeight * aspect;
    this.camera.top = this.fitHeight;
    this.camera.bottom = -this.fitHeight;
    this.camera.updateProjectionMatrix();
  }
  pick(x, y) {
    if (!this.cloud || !this.showSkeleton) return null;
    const rect = this.canvas.getBoundingClientRect(),
      p = new THREE.Vector3();
    const positions = this.joints.geometry.attributes.position.array;
    let closest = null,
      distance = 14;
    for (let joint = 0; joint < this.parents.length; joint++) {
      p.fromArray(positions, joint * 3).project(this.camera);
      const d = Math.hypot(
        (p.x * 0.5 + 0.5) * rect.width + rect.left - x,
        (-p.y * 0.5 + 0.5) * rect.height + rect.top - y,
      );
      if (d < distance) {
        closest = joint;
        distance = d;
      }
    }
    return closest;
  }
  render() {
    this.resize();
    this.orbit.update();
    this.renderer.render(this.scene, this.camera);
  }
}
