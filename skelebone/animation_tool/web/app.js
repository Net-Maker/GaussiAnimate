import { AssetViewport } from "./scene.js";
import {
  createIcons,
  Play,
  Pause,
  Undo2,
  Download,
  Focus,
} from "https://cdn.jsdelivr.net/npm/lucide@0.468.0/+esm";

const $ = (id) => document.getElementById(id);
const icons = () =>
  createIcons({ icons: { Play, Pause, Undo2, Download, Focus } });
const state = {
  generation: 0,
  edit: 0,
  ready: false,
  scene: null,
  offsets: {},
  mode: "auto",
  busy: false,
  pending: null,
  timer: null,
  controller: null,
  meta: null,
  reference: null,
  playing: false,
  frame: 0,
  clock: 0,
};

const asset = new AssetViewport($("stage"), {
  editable: true,
  onPick: (joint) => selectJoint(joint),
  onRotate: (joint, angles) => {
    state.offsets[joint] = angles;
    showAngles(joint);
    queuePose();
  },
});
const reference = new AssetViewport($("reference-stage"));

function status(text, error = false) {
  $("status").textContent = text;
  $("status").classList.toggle("error", error);
}
function poseStatus(text, preview = false) {
  $("pose-status").textContent = text;
  $("pose-status").classList.toggle("preview", preview);
}
function showDriver(result) {
  const fallback = result.driver === "lbs" && result.mode === "auto";
  poseStatus(
    fallback ? "LBS fallback" : result.driver === "lbs" ? "LBS" : "PartMM",
  );
  $("pose-status").dataset.driver = result.driver;
  $("pose-status").dataset.reason = result.drive_reason;
  const angle = `${result.match_angle_degrees.toFixed(1)} / ${result.max_match_angle_degrees.toFixed(1)} deg`;
  $("pose-status").title = `Match / limit: ${angle}`;
  if (fallback) {
    status(
      result.drive_reason === "out_of_support"
        ? `LBS fallback | Match / limit: ${angle}`
        : "LBS fallback | Matching failed",
    );
  } else {
    const label = result.driver === "lbs" ? "LBS" : "PartMM";
    status(
      result.elapsed_ms == null
        ? `${label} ready`
        : `${label} pose (${Math.round(result.elapsed_ms)} ms)`,
    );
  }
}
function exportEnabled(enabled) {
  const link = $("export-link");
  link.setAttribute("aria-disabled", String(!enabled));
  if (enabled)
    link.href = `/api/scenes/${encodeURIComponent(state.scene)}/export.npz`;
  else link.removeAttribute("href");
}
function controlsEnabled(enabled) {
  document
    .querySelectorAll(
      "aside input,aside select,aside button,#play-btn,#timeline",
    )
    .forEach((el) => {
      el.disabled = !enabled;
    });
}
async function json(url, options = {}) {
  const response = await fetch(url, options),
    body = await response.json();
  if (!response.ok) throw new Error(body.error || `HTTP ${response.status}`);
  return body;
}
async function buffer(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok)
    throw new Error(`Buffer request failed: ${response.status}`);
  return new Float32Array(await response.arrayBuffer());
}

function selectJoint(joint) {
  if (!state.ready) return;
  $("joint-select").value = String(joint);
  asset.select(joint);
  showAngles(joint);
}
function showAngles(joint) {
  const angles = state.offsets[joint] || [0, 0, 0];
  ["x", "y", "z"].forEach((axis, index) => {
    $(`rot-${axis}`).value = angles[index];
    $(`rot-${axis}-num`).value = Number(angles[index].toFixed(2));
  });
}
function queuePose() {
  state.edit++;
  poseStatus("LBS preview", true);
  $("pose-status").dataset.driver = "preview";
  status("Resolving pose");
  exportEnabled(false);
  state.pending = {
    generation: state.generation,
    edit: state.edit,
    scene: state.scene,
    offsets: structuredClone(state.offsets),
    mode: state.mode,
  };
  clearTimeout(state.timer);
  state.timer = setTimeout(() => {
    state.timer = null;
    solvePending();
  }, 80);
}
async function solvePending() {
  if (state.busy || !state.pending) return;
  const request = state.pending;
  state.pending = null;
  state.busy = true;
  try {
    const result = await json(
      `/api/scenes/${encodeURIComponent(request.scene)}/animate`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ offsets: request.offsets, mode: request.mode }),
      },
    );
    if (request.generation !== state.generation || request.edit !== state.edit)
      return;
    asset.setGeometry(
      new Float32Array(result.points),
      new Float32Array(result.joint_positions),
    );
    showDriver(result);
    exportEnabled(true);
  } catch (error) {
    if (
      request.generation === state.generation &&
      request.edit === state.edit
    ) {
      status(error.message, true);
      poseStatus("Preview only", true);
    }
  } finally {
    state.busy = false;
    if (state.pending && !state.timer) solvePending();
  }
}

async function loadScene(name) {
  const generation = ++state.generation;
  state.edit++;
  clearTimeout(state.timer);
  state.timer = null;
  state.pending = null;
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  state.ready = false;
  state.scene = name;
  state.playing = false;
  state.frame = 0;
  state.clock = 0;
  updatePlay();
  controlsEnabled(false);
  exportEnabled(false);
  asset.clear();
  reference.clear();
  status(`Loading ${name}`);
  poseStatus("Loading");
  try {
    const meta = await json(`/api/scenes/${encodeURIComponent(name)}`, {
      signal: controller.signal,
    });
    const [points, joints, base, input, inputJoints] = await Promise.all(
      ["driven", "driven_joints", "base", "input", "input_joints"].map((key) =>
        buffer(meta.buffers[key], controller.signal),
      ),
    );
    if (generation !== state.generation) return;
    if (
      points.length !== meta.points * 3 ||
      joints.length !== meta.joints * 3 ||
      base.length !== points.length ||
      input.length !== meta.frames * meta.points * 3 ||
      inputJoints.length !== meta.frames * meta.joints * 3
    )
      throw new Error("Asset buffer dimensions do not match");
    state.meta = meta;
    state.offsets = structuredClone(meta.joint_offsets);
    state.mode = meta.mode;
    $("driver-mode").value = meta.mode;
    state.reference = { points: input, joints: inputJoints };
    asset.install(meta, points, joints, base);
    asset.setOffsets(state.offsets, false);
    reference.install(
      meta,
      input.subarray(0, meta.points * 3),
      inputJoints.subarray(0, meta.joints * 3),
    );
    $("up-axis").value = meta.up_axis;
    $("joint-select").replaceChildren(
      ...meta.names.map((name, index) => {
        const option = document.createElement("option");
        option.value = index;
        option.textContent = name;
        return option;
      }),
    );
    $("timeline").max = meta.frames - 1;
    state.ready = true;
    controlsEnabled(true);
    exportEnabled(true);
    selectJoint(meta.default_joint);
    setReferenceFrame(0);
    $("stats").textContent =
      `${meta.points.toLocaleString()} points / ${meta.joints} joints`;
    showDriver(meta);
  } catch (error) {
    if (generation === state.generation && error.name !== "AbortError") {
      status(error.message, true);
      poseStatus("Unavailable");
    }
  }
}

function updatePlay() {
  const label = state.playing ? "Pause reference" : "Play reference";
  $("play-btn").title = label;
  $("play-btn").setAttribute("aria-label", label);
  $("play-btn").innerHTML =
    `<i data-lucide="${state.playing ? "pause" : "play"}"></i>`;
  icons();
}
function setReferenceFrame(frame) {
  if (!state.ready) return;
  state.frame = Math.max(0, Math.min(state.meta.frames - 1, frame));
  const p = state.meta.points * 3,
    j = state.meta.joints * 3,
    f = state.frame;
  reference.setGeometry(
    state.reference.points.subarray(f * p, (f + 1) * p),
    state.reference.joints.subarray(f * j, (f + 1) * j),
  );
  $("timeline").value = f;
  $("frame-label").textContent = `${f + 1} / ${state.meta.frames}`;
}

$("scene-select").addEventListener("change", (event) =>
  loadScene(event.target.value),
);
$("joint-select").addEventListener("change", (event) =>
  selectJoint(Number(event.target.value)),
);
$("driver-mode").addEventListener("change", (event) => {
  state.mode = event.target.value;
  asset.setOffsets(state.offsets);
  queuePose();
});
for (const [index, axis] of ["x", "y", "z"].entries()) {
  for (const suffix of ["", "-num"]) {
    $(`rot-${axis}${suffix}`).addEventListener("input", (event) => {
      if (!state.ready || event.target.value === "") return;
      const value = Number(event.target.value);
      if (!Number.isFinite(value)) return;
      const joint = Number($("joint-select").value);
      const angles = [...(state.offsets[joint] || [0, 0, 0])];
      angles[index] = Math.max(-180, Math.min(180, value));
      state.offsets[joint] = angles;
      $(`rot-${axis}${suffix ? "" : "-num"}`).value = angles[index];
      asset.setOffsets(state.offsets);
      queuePose();
    });
  }
}
$("reset-pose-btn").addEventListener("click", () => {
  state.offsets = {};
  asset.setOffsets({});
  showAngles(Number($("joint-select").value));
  queuePose();
});
$("reset-view-btn").addEventListener("click", () => asset.fit());
$("skeleton-check").addEventListener("change", (event) => {
  asset.setSkeletonVisible(event.target.checked);
  reference.setSkeletonVisible(event.target.checked);
});
$("point-size").addEventListener("input", (event) => {
  const size = Number(event.target.value);
  asset.setPointSize(size);
  reference.setPointSize(size);
  $("point-size-out").textContent = size;
});
$("color-mode").addEventListener("change", (event) => {
  asset.setColorMode(event.target.value);
  reference.setColorMode(event.target.value);
});
$("up-axis").addEventListener("change", (event) => {
  asset.upAxis = reference.upAxis = event.target.value;
  asset.fit();
  reference.fit();
});
$("reference-check").addEventListener("change", (event) => {
  $("reference-panel").hidden = !event.target.checked;
  if (event.target.checked) reference.fit();
  else {
    state.playing = false;
    updatePlay();
  }
});
$("play-btn").addEventListener("click", () => {
  state.playing = !state.playing;
  state.clock = 0;
  updatePlay();
});
$("timeline").addEventListener("input", (event) => {
  state.playing = false;
  updatePlay();
  setReferenceFrame(Number(event.target.value));
});

let last = performance.now();
function tick(now) {
  const delta = Math.min((now - last) / 1000, 0.1);
  last = now;
  if (state.ready && state.playing) {
    state.clock += delta * state.meta.fps;
    if (state.clock >= 1) {
      const step = Math.floor(state.clock);
      state.clock -= step;
      setReferenceFrame((state.frame + step) % state.meta.frames);
    }
  }
  asset.render();
  if (!$("reference-panel").hidden) reference.render();
  requestAnimationFrame(tick);
}
icons();
controlsEnabled(false);
requestAnimationFrame(tick);
try {
  const catalogue = await json("/api/scenes");
  $("scene-select").replaceChildren(
    ...catalogue.scenes.map((scene) => {
      const option = document.createElement("option");
      option.value = scene.id;
      option.textContent = scene.label;
      return option;
    }),
  );
  $("scene-select").value = catalogue.default_scene;
  await loadScene(catalogue.default_scene);
} catch (error) {
  status(error.message, true);
}
