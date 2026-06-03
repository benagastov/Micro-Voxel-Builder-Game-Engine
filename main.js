'use strict';

import {
  CH, WX, WY, WZ, WORLD_VX, WORLD_VY, WORLD_VZ, WORLD_VOXELS,
  WORDS_PER_CHUNK, BYTES_PER_CHUNK, BYTES_PER_COLOR_CHUNK, MAX_RESIDENT, CELLS,
  UNKNOWN, EMPTY, FULL, UPLOAD_BUDGET, UNLOAD_MARGIN
} from "./config.js";
import { SHADER_SRC, BLIT_SRC, EDGE_SRC, WORKER_SRC } from "./shaders.js";
import {
  BUILD, BUILD_MATS, circleCursorXZ, circleParams, genCircleTower,
  squareCursorXZ, squareParams, genSquareRoom, genGhosts,
  thicken, raiseColumns
} from "./building.js";


// ===========================================================================
// Streaming voxel world — WebGPU
//   • 256 x 8 x 256 chunks of 32^3  =>  8192 x 256 x 8192 = 17.18 billion voxels.
//   • Two-level DDA (Amanatides & Woo): outer stride over chunks, inner over
//     voxels. Empty/unknown chunks cost one step; full chunks hit at the face.
//   • Direct flat chunk-index grid (no ring buffer, no modulo remap):
//       cellIndex = (cz*WY + cy)*WX + cx
//     holds UNKNOWN / EMPTY / FULL sentinels, or a pool slot index.
//   • Parallel web workers generate chunks (heightmap + classification);
//     mixed chunks go to a GPU slot pool, air/solid chunks cost zero memory.
// ===========================================================================

// ---------------------------------------------------------------------------
// WGSL compute shader — two-level DDA
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Blit shader — fullscreen triangle copying the compute output to the canvas.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Chunk-generation worker (built from a Blob so the page stays single-file).
// Heightmap terrain: solid where worldY <= height(x,z). Classifies each chunk
// as EMPTY / FULL / MIXED so air & solid chunks cost zero GPU memory.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// App state
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const canvas = $('c');

function showError(title, detail) {
  const e = document.createElement('div');
  e.className = 'error';
  e.innerHTML = `<h2>${title}</h2><div>${detail}</div>`;
  document.body.appendChild(e);
}
if (!navigator.gpu) {
  showError("WebGPU not supported",
    `This needs WebGPU (Chrome/Edge 113+, recent Firefox/Safari). See
     <a href="https://caniuse.com/webgpu" target="_blank">caniuse.com/webgpu</a>.`);
  throw new Error("no webgpu");
}

let device, queue, ctx, canvasFormat;
let computePipeline, blitPipeline, ghostPipeline = null, edgePipeline = null;
let outTexture = null, normDepthTexture = null, edgeTexture = null;
let computeBG = null, blitBG = null, ghostBG = null, edgeBG = null;
let poolBuf, colorPoolBuf, gridBuf, uniformBuf, ghostUniBuf = null, ghostInstBuf = null, ghostDepthTex = null, ghostCubeBuf = null;
let ghostCount = 0;
const GHOST_MAX = 40000;
let renderW = 1, renderH = 1;
let renderScale = 0.65, renderDist = 14, flySpeed = 120;

// Pre-allocated zero buffer for clearing a color pool slot (1024 u32s = 4096 bytes)
const _zeroColorChunk = new Uint32Array(BYTES_PER_COLOR_CHUNK / 4);

// World bookkeeping (CPU mirror of the GPU index grid).
const gridCPU = new Uint32Array(CELLS);
const pending = new Uint8Array(CELLS);
const freeSlots = [];
const slotCell = new Int32Array(MAX_RESIDENT).fill(-1);
let seed = (Math.random() * 1e6) | 0;

// Build/edit overlay. Values: {s:1,r,g,b} = forced solid with color, {s:0} = forced empty.
const EDITS = new Map();
const SAVE_VERSION = 2;
const editKey = (x,y,z) => `${x},${y},${z}`;
function parseEditKey(k) { return k.split(',').map(Number); }

// Chunk-indexed edits: chunkCell → Map(localIndex → {x,y,z,v}). Lets editsForChunk()
// read only the edits in/around a chunk instead of scanning the whole EDITS map on
// every dispatch — which was O(EDITS) per streamed chunk and stalled the main thread
// once a large area had been terraformed.
const chunkEdits = new Map();   // cell → Map(li → {x,y,z,v})
function editChunkCell(x,y,z){ return (Math.floor(z/CH)*WY + Math.floor(y/CH))*WX + Math.floor(x/CH); }
function setEdit(x,y,z,v){
  EDITS.set(editKey(x,y,z), v);
  const cell = editChunkCell(x,y,z);
  let m = chunkEdits.get(cell);
  if (!m) { m = new Map(); chunkEdits.set(cell, m); }
  const lx = x - Math.floor(x/CH)*CH, ly = y - Math.floor(y/CH)*CH, lz = z - Math.floor(z/CH)*CH;
  m.set(lx + CH*(ly + CH*lz), {x,y,z,v});
}
function clearEdits(){ EDITS.clear(); chunkEdits.clear(); }

// Streaming metrics
let cumVoxels = 0, chunksGen = 0, residentMixed = 0, residentFull = 0;
let genMsAccum = 0, genMsCount = 0;
let chunkRateWindow = [];     // timestamps of recent completions

// Workers
let workers = [];
let workerBusy = [];
let inFlight = 0;

// Camera (free fly, world-voxel coordinates)
const cam = {
  pos: [WORLD_VX / 2, 95, WORLD_VZ / 2],
  yaw: 0.6, pitch: -0.45, fov: 70,
};
const keys = {};
let pointerLocked = false, autoFly = false, hintFaded = false;

const scratchU32 = new Uint32Array(1);

// ---------------------------------------------------------------------------
// Ghost block rendering (translucent instanced cubes)
// ---------------------------------------------------------------------------
const GHOST_SHADER = /* wgsl */`
struct GU { vp : mat4x4<f32>, tint : vec4<f32> };
@group(0) @binding(0) var<uniform> G : GU;
struct VO { @builtin(position) pos : vec4<f32>, @location(0) nrm : vec3<f32> };
@vertex fn vs(
  @location(0) corner : vec3<f32>,
  @location(1) nor    : vec3<f32>,
  @location(2) inst   : vec4<f32>
) -> VO {
  var o : VO;
  let wp = inst.xyz + corner;
  o.pos = G.vp * vec4<f32>(wp, 1.0);
  o.nrm = nor;
  return o;
}
@fragment fn fs(i : VO) -> @location(0) vec4<f32> {
  let l = clamp(dot(normalize(i.nrm), normalize(vec3<f32>(0.5, 0.78, 0.38))) * 0.5 + 0.5, 0.0, 1.0);
  return vec4<f32>(G.tint.rgb * (0.55 + 0.5 * l), G.tint.a);
}
`;

function mat4Perspective(fovy, aspect, near, far) {
  // WebGPU NDC depth range is [0, 1]
  const f = 1 / Math.tan(fovy / 2), nf = 1 / (near - far);
  return new Float32Array([
    f / aspect, 0,  0,               0,
    0,          f,  0,               0,
    0,          0,  far * nf,       -1,
    0,          0,  near * far * nf, 0,
  ]);
}
function mat4LookAt(e, c, u) {
  let zx = e[0]-c[0], zy = e[1]-c[1], zz = e[2]-c[2];
  let zl = Math.hypot(zx,zy,zz)||1; zx/=zl; zy/=zl; zz/=zl;
  let xx = u[1]*zz-u[2]*zy, xy = u[2]*zx-u[0]*zz, xz = u[0]*zy-u[1]*zx;
  let xl = Math.hypot(xx,xy,xz)||1; xx/=xl; xy/=xl; xz/=xl;
  const yx = zy*xz-zz*xy, yy = zz*xx-zx*xz, yz = zx*xy-zy*xx;
  return new Float32Array([
    xx, yx, zx, 0,  xy, yy, zy, 0,  xz, yz, zz, 0,
    -(xx*e[0]+xy*e[1]+xz*e[2]), -(yx*e[0]+yy*e[1]+yz*e[2]), -(zx*e[0]+zy*e[1]+zz*e[2]), 1
  ]);
}
function mat4Mul(a, b) {
  const o = new Float32Array(16);
  for (let c = 0; c < 4; c++) for (let r = 0; r < 4; r++) {
    let s = 0; for (let k = 0; k < 4; k++) s += a[k*4+r] * b[c*4+k]; o[c*4+r] = s;
  }
  return o;
}

function setupGhostPipeline() {
  // unit cube: 36 vertices (corner xyz + face normal xyz)
  const faces = [
    [[0,0,0],[1,0,0],[1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,0,-1]],
    [[0,0,1],[1,1,1],[1,0,1],[0,0,1],[0,1,1],[1,1,1],[0,0,1]],
    [[0,0,0],[0,1,0],[0,1,1],[0,0,0],[0,1,1],[0,0,1],[-1,0,0]],
    [[1,0,0],[1,1,1],[1,1,0],[1,0,0],[1,0,1],[1,1,1],[1,0,0]],
    [[0,0,0],[1,0,1],[1,0,0],[0,0,0],[0,0,1],[1,0,1],[0,-1,0]],
    [[0,1,0],[1,1,0],[1,1,1],[0,1,0],[1,1,1],[0,1,1],[0,1,0]],
  ];
  const data = [];
  for (const f of faces) { const n = f[6]; for (let i = 0; i < 6; i++) { const v = f[i]; data.push(v[0],v[1],v[2], n[0],n[1],n[2]); } }
  const arr = new Float32Array(data);
  ghostCubeBuf = device.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  queue.writeBuffer(ghostCubeBuf, 0, arr);

  ghostInstBuf = device.createBuffer({ size: GHOST_MAX * 16, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
  ghostUniBuf  = device.createBuffer({ size: 80, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const gm = device.createShaderModule({ code: GHOST_SHADER });
  ghostPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: gm, entryPoint: 'vs', buffers: [
      { arrayStride: 24, stepMode: 'vertex',   attributes: [
        { shaderLocation: 0, offset: 0,  format: 'float32x3' },
        { shaderLocation: 1, offset: 12, format: 'float32x3' },
      ]},
      { arrayStride: 16, stepMode: 'instance', attributes: [
        { shaderLocation: 2, offset: 0, format: 'float32x4' },
      ]},
    ]},
    fragment: { module: gm, entryPoint: 'fs', targets: [{ format: canvasFormat, blend: {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one',       dstFactor: 'one-minus-src-alpha', operation: 'add' },
    }}]},
    primitive: { topology: 'triangle-list', cullMode: 'none' },
    depthStencil: { format: 'depth24plus', depthWriteEnabled: true, depthCompare: 'less' },
  });
  ghostBG = device.createBindGroup({
    layout: ghostPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: ghostUniBuf } }],
  });
}

function uploadGhosts() {
  const subMode = BUILD.subMode || 'native';
  let g;
  if (subMode === 'planner') {
    // pre-anchor: update ghost to follow cursor
    if (!BUILD.apAnchored) {
      const rc = BUILD._lastRC;
      if (rc) { BUILD.apGhost = [rc.place[0], rc.place[1], rc.place[2]]; }
    }
    g = genPlannerGhosts();
  } else if (subMode === 'terraform') {
    g = genTerraformGhosts();
    if (BUILD.tfPainting) applyTerraform();
  } else {
    g = genGhosts();
  }
  BUILD.ghosts = g;
  const n = Math.min(g.length, GHOST_MAX);
  if (n > 0) {
    const a = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) { a[i*4] = g[i][0]; a[i*4+1] = g[i][1]; a[i*4+2] = g[i][2]; a[i*4+3] = 0; }
    queue.writeBuffer(ghostInstBuf, 0, a);
  }
  ghostCount = n;
}

function rebuildGhostDepth() {
  if (ghostDepthTex) ghostDepthTex.destroy();
  ghostDepthTex = device.createTexture({
    size: [canvas.width, canvas.height],
    format: 'depth24plus',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------
async function init() {
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) { showError("No WebGPU adapter", "Your GPU/driver may be unsupported."); throw new Error("no adapter"); }
  device = await adapter.requestDevice();
  queue = device.queue;
  canvasFormat = navigator.gpu.getPreferredCanvasFormat();
  device.addEventListener?.('uncapturederror', e => console.error('WebGPU:', e.error));

  const cm = device.createShaderModule({ code: SHADER_SRC });
  cm.getCompilationInfo().then(info => {
    const errs = (info.messages || []).filter(m => m.type === 'error');
    if (errs.length) showError("Compute shader error",
      `<pre>${errs.map(m => `${m.message} (${m.lineNum}:${m.linePos})`).join('\n')}</pre>`);
  });
  computePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: cm, entryPoint: 'main' } });

  const bm = device.createShaderModule({ code: BLIT_SRC });
  blitPipeline = device.createRenderPipeline({
    layout: 'auto',
    vertex: { module: bm, entryPoint: 'vs' },
    fragment: { module: bm, entryPoint: 'fs', targets: [{ format: canvasFormat }] },
    primitive: { topology: 'triangle-list' },
  });

  const em = device.createShaderModule({ code: EDGE_SRC });
  em.getCompilationInfo().then(info => {
    const errs = (info.messages || []).filter(m => m.type === 'error');
    if (errs.length) showError("Edge shader error",
      `<pre>${errs.map(m => `${m.message} (${m.lineNum}:${m.linePos})`).join('\n')}</pre>`);
  });
  edgePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: em, entryPoint: 'main' } });

  poolBuf      = device.createBuffer({ size: MAX_RESIDENT * BYTES_PER_CHUNK,       usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  colorPoolBuf = device.createBuffer({ size: MAX_RESIDENT * BYTES_PER_COLOR_CHUNK, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  gridBuf      = device.createBuffer({ size: CELLS * 4,                            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  uniformBuf   = device.createBuffer({ size: 96,                                   usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  ctx = canvas.getContext('webgpu');
  ctx.configure({ device, format: canvasFormat, alphaMode: 'opaque' });
  setupGhostPipeline();

  $('worldsub').textContent = `${WORLD_VOXELS.toLocaleString()}-voxel world · ${WX}×${WY}×${WZ} chunks`;

  resize();
  window.addEventListener('resize', resize);
  spawnWorkers();
  setupUI();
  setupInput();
  resetWorld();
}

function spawnWorkers() {
  const url = URL.createObjectURL(new Blob([WORKER_SRC], { type: 'application/javascript' }));
  const n = Math.max(2, Math.min(12, navigator.hardwareConcurrency || 4));
  for (let i = 0; i < n; i++) {
    const w = new Worker(url);
    const wi = i;
    w.onmessage = (e) => onChunk(e, wi);
    workers.push(w);
    workerBusy.push(false);
  }
}

function resetWorld() {
  _heightCache.clear();
  gridCPU.fill(UNKNOWN);
  pending.fill(0);
  freeSlots.length = 0;
  for (let i = MAX_RESIDENT - 1; i >= 0; i--) freeSlots.push(i);
  slotCell.fill(-1);
  cumVoxels = 0; chunksGen = 0; residentMixed = 0; residentFull = 0;
  genMsAccum = 0; genMsCount = 0; chunkRateWindow.length = 0;
  if (typeof pendingUploads !== 'undefined') pendingUploads.length = 0;
  dirtyChunks.clear();
  cam.pos = [WORLD_VX / 2, 188, WORLD_VZ / 2];
  cam.yaw = 0.6; cam.pitch = -0.45;
  queue.writeBuffer(gridBuf, 0, gridCPU);     // upload full UNKNOWN grid once
}

// ---------------------------------------------------------------------------
// Worker dispatch + result handling
// ---------------------------------------------------------------------------
function editsForChunk(cx, cy, cz) {
  const out = [];       // interior edits: [localIndex, solid, r, g, b]
  const apron = [];     // edits in the 1-voxel border of neighbor chunks: [wx, wy, wz, solid]
  const cell = (cz * WY + cy) * WX + cx;

  // Interior edits: O(this chunk's edits), straight from the index.
  const mine = chunkEdits.get(cell);
  if (mine) for (const { x, y, z, v } of mine.values()) {
    const li = (x - cx*CH) + CH * ((y - cy*CH) + CH * (z - cz*CH));
    out.push([ li, v.s, v.r ?? 0, v.g ?? 0, v.b ?? 0 ]);
  }

  // Apron: edits one voxel outside this chunk on a shared face. Only the 6 face-
  // neighbor chunks can hold them, and only their voxels touching the shared plane
  // matter. Scanning those buckets is far cheaper than the whole EDITS map.
  const x0 = cx*CH, y0 = cy*CH, z0 = cz*CH, x1 = x0+CH-1, y1 = y0+CH-1, z1 = z0+CH-1;
  const addApronFrom = (ncx, ncy, ncz) => {
    if (ncx<0||ncy<0||ncz<0||ncx>=WX||ncy>=WY||ncz>=WZ) return;
    const nm = chunkEdits.get((ncz*WY+ncy)*WX+ncx);
    if (!nm) return;
    for (const { x, y, z, v } of nm.values()) {
      // Keep only edits within the 1-voxel apron shell of this chunk.
      if (x>=x0-1 && x<=x1+1 && y>=y0-1 && y<=y1+1 && z>=z0-1 && z<=z1+1) apron.push([ x, y, z, v.s ]);
    }
  };
  addApronFrom(cx-1,cy,cz); addApronFrom(cx+1,cy,cz);
  addApronFrom(cx,cy-1,cz); addApronFrom(cx,cy+1,cz);
  addApronFrom(cx,cy,cz-1); addApronFrom(cx,cy,cz+1);
  return { interior: out, apron };
}
function dispatchChunk(cell, cx, cy, cz) {
  const wi = workerBusy.indexOf(false);
  if (wi < 0) return false;
  workerBusy[wi] = true;
  pending[cell] = 1;
  // NOTE: we deliberately do NOT blank gridCPU[cell] to UNKNOWN here. A regenerating
  // chunk keeps its existing data on the GPU (old slot / sentinel) so it stays visible
  // until the new data lands and is swapped in place — no flash to sky on edits.
  inFlight++;
  const { interior, apron } = editsForChunk(cx, cy, cz);
  workers[wi].postMessage({ cell, cx, cy, cz, seed, edits: interior, apronEdits: apron });
  return true;
}

let pendingUploads = [];   // mixed chunks awaiting GPU write (budgeted per frame)

function onChunk(e, wi) {
  const m = e.data;
  workerBusy[wi] = false;
  inFlight--;
  chunksGen++;
  cumVoxels += m.count;
  genMsAccum += m.genMs; genMsCount++;
  chunkRateWindow.push(performance.now());

  if (m.kind === 0) {                 // EMPTY
    releaseOldSlotIfAny(m.cell);      // free a prior MIXED slot (rebuild that emptied)
    pending[m.cell] = 0;
    setGrid(m.cell, EMPTY);
  } else if (m.kind === 1) {          // FULL
    releaseOldSlotIfAny(m.cell);
    pending[m.cell] = 0;
    setGrid(m.cell, FULL);
    residentFull++;
  } else {                            // MIXED — stays pending until its GPU upload lands
    pendingUploads.push({
      cell: m.cell,
      words:  new Uint32Array(m.buf),
      colors: m.cbuf ? new Uint32Array(m.cbuf) : null,
    });
  }
}

function setGrid(cell, value) {
  gridCPU[cell] = value;
  scratchU32[0] = value;
  queue.writeBuffer(gridBuf, cell * 4, scratchU32);
}

// Free a chunk's resident slot/FULL bookkeeping if it had one, WITHOUT touching the
// grid sentinel — the caller writes the chunk's new state immediately afterward.
function releaseOldSlotIfAny(cell) {
  const state = gridCPU[cell];
  if (state < FULL) {
    const slot = state;
    if (slotCell[slot] === cell) { slotCell[slot] = -1; freeSlots.push(slot); residentMixed = Math.max(0, residentMixed - 1); }
  } else if (state === FULL) {
    residentFull = Math.max(0, residentFull - 1);
  }
}

function allocSlot(cell) {
  if (freeSlots.length === 0) return -1;
  const slot = freeSlots.pop();
  slotCell[slot] = cell;
  residentMixed++;
  return slot;
}
function freeSlot(slot) {
  const cell = slotCell[slot];
  if (cell >= 0) { gridCPU[cell] = UNKNOWN; scratchU32[0] = UNKNOWN; queue.writeBuffer(gridBuf, cell * 4, scratchU32); }
  slotCell[slot] = -1;
  freeSlots.push(slot);
  residentMixed--;
}

// Chunks whose voxel data changed (edits) and need regeneration. We keep the OLD
// data rendering until the new data lands, so edits never flash the chunk to sky.
const dirtyChunks = new Set();   // cell indices awaiting a (re)generation dispatch

function markDirty(cx, cy, cz) {
  if (cx < 0 || cy < 0 || cz < 0 || cx >= WX || cy >= WY || cz >= WZ) return false;
  dirtyChunks.add((cz * WY + cy) * WX + cx);
  return true;
}

function rebuildTouchedVoxels(list) {
  const cells = new Set();
  for (const [x,y,z] of list) {
    const cx = Math.floor(x/CH), cy = Math.floor(y/CH), cz = Math.floor(z/CH);
    // Always regen the edited chunk AND its 6 face-neighbors. An edit can change the
    // air-touching exposure of any voxel within ±1 cell, and that voxel may live in an
    // adjacent chunk. Regenerating all 6 neighbors unconditionally is robust (no missed
    // walls); the dirty-set dedups and the edits-first dispatch keeps painting cheap.
    cells.add(`${cx},${cy},${cz}`);
    cells.add(`${cx-1},${cy},${cz}`); cells.add(`${cx+1},${cy},${cz}`);
    cells.add(`${cx},${cy-1},${cz}`); cells.add(`${cx},${cy+1},${cz}`);
    cells.add(`${cx},${cy},${cz-1}`); cells.add(`${cx},${cy},${cz+1}`);
  }
  let n = 0;
  for (const key of cells) {
    const [cx,cy,cz] = key.split(',').map(Number);
    if (markDirty(cx,cy,cz)) n++;
  }
  return n;
}

// ---------------------------------------------------------------------------
// Streaming: request nearest unknown chunks, upload, and unload distant ones.
// ---------------------------------------------------------------------------
function streamingUpdate() {
  const ccx = Math.floor(cam.pos[0] / CH);
  const ccy = Math.floor(cam.pos[1] / CH);
  const ccz = Math.floor(cam.pos[2] / CH);
  const R = renderDist;
  const R2 = R * R;

  // 1) Upload budgeted mixed chunks.
  let uploaded = 0;
  while (pendingUploads.length && uploaded < UPLOAD_BUDGET) {
    const job = pendingUploads.shift();
    const prev = gridCPU[job.cell];
    // Reuse the chunk's existing slot if it already had one (in-place rebuild after an
    // edit) — no free/alloc churn, and the chunk never blanks to sky. Otherwise grab a
    // free slot. A chunk regenerated while it was streamed-out (UNKNOWN) just allocates.
    let slot;
    if (prev < FULL && slotCell[prev] === job.cell) {
      slot = prev;                       // overwrite in place
    } else {
      slot = allocSlot(job.cell);
      if (slot < 0) { evictFarthest(ccx, ccy, ccz); slot = allocSlot(job.cell); }
      if (slot < 0) { pending[job.cell] = 0; continue; }                    // pool full: retry later
    }
    queue.writeBuffer(poolBuf, slot * BYTES_PER_CHUNK, job.words);
    if (job.colors) {
      queue.writeBuffer(colorPoolBuf, slot * BYTES_PER_COLOR_CHUNK, job.colors);
    } else {
      // Zero out stale color data from previous occupant (zeroes = terrain color)
      queue.writeBuffer(colorPoolBuf, slot * BYTES_PER_COLOR_CHUNK, _zeroColorChunk);
    }
    setGrid(job.cell, slot);
    pending[job.cell] = 0;
    uploaded++;
  }

  // 1b) Dispatch dirty (edited) chunks FIRST, ahead of streaming new terrain, so edits
  // show up immediately. Coalesced via the Set: painting the same spot re-marks the
  // same cells instead of piling up. A chunk already in flight is left for next frame.
  if (dirtyChunks.size) {
    for (const cell of dirtyChunks) {
      if (workerBusy.indexOf(false) < 0) break;   // all workers busy — rest wait next frame
      // Leave a chunk that's still regenerating in the set WITHOUT dispatching: its
      // in-flight worker holds an older edit snapshot, so once it lands we must re-run
      // it with the latest edits. We only remove a chunk on a successful dispatch, and
      // a chunk is only (re)added by markDirty() on a real edit — so no spin-loop.
      if (pending[cell]) continue;
      const cx = cell % WX;
      const cy = ((cell / WX) | 0) % WY;
      const cz = (cell / (WX * WY)) | 0;
      if (dispatchChunk(cell, cx, cy, cz)) dirtyChunks.delete(cell);
      else break;
    }
  }

  // 2) Unload mixed chunks beyond render distance (with hysteresis).
  const UR = R + UNLOAD_MARGIN, UR2 = UR * UR;
  for (let s = 0; s < MAX_RESIDENT; s++) {
    const cell = slotCell[s];
    if (cell < 0) continue;
    const cx = cell % WX;
    const cy = ((cell / WX) | 0) % WY;
    const cz = (cell / (WX * WY)) | 0;
    const dx = cx - ccx, dy = cy - ccy, dz = cz - ccz;
    if (dx * dx + dz * dz > UR2 || Math.abs(dy) > UR) freeSlot(s);
  }

  // 3) Request nearest UNKNOWN chunks within range, nearest-first.
  const idle = workerBusy.reduce((a, b) => a + (b ? 0 : 1), 0);
  if (idle === 0) return;
  const cands = [];
  const yLo = Math.max(0, ccy - R), yHi = Math.min(WY - 1, ccy + R);
  const xLo = Math.max(0, ccx - R), xHi = Math.min(WX - 1, ccx + R);
  const zLo = Math.max(0, ccz - R), zHi = Math.min(WZ - 1, ccz + R);
  for (let cz = zLo; cz <= zHi; cz++) {
    const dz = cz - ccz;
    for (let cx = xLo; cx <= xHi; cx++) {
      const dx = cx - ccx;
      if (dx * dx + dz * dz > R2) continue;
      for (let cy = yLo; cy <= yHi; cy++) {
        const cell = (cz * WY + cy) * WX + cx;
        if (gridCPU[cell] !== UNKNOWN || pending[cell]) continue;
        const dy = cy - ccy;
        cands.push([dx * dx + dy * dy + dz * dz, cell, cx, cy, cz]);
      }
    }
  }
  if (!cands.length) return;
  cands.sort((a, b) => a[0] - b[0]);
  const take = Math.min(idle, cands.length);
  for (let i = 0; i < take; i++) {
    const c = cands[i];
    if (!dispatchChunk(c[1], c[2], c[3], c[4])) break;
  }
}

function evictFarthest(ccx, ccy, ccz) {
  let worst = -1, worstD = -1;
  for (let s = 0; s < MAX_RESIDENT; s++) {
    const cell = slotCell[s];
    if (cell < 0) continue;
    const cx = cell % WX;
    const cy = ((cell / WX) | 0) % WY;
    const cz = (cell / (WX * WY)) | 0;
    const dx = cx - ccx, dy = cy - ccy, dz = cz - ccz;
    const d = dx * dx + dy * dy + dz * dz;
    if (d > worstD) { worstD = d; worst = s; }
  }
  if (worst >= 0) freeSlot(worst);
}

// ---------------------------------------------------------------------------
// Camera matrix (orbit-free pinhole; col3 = eye, M*(px,py,1,0) = ray dir)
// ---------------------------------------------------------------------------
function rayMatrix() {
  const W = renderW, H = renderH, aspect = W / H;
  const tanFov = Math.tan(cam.fov * Math.PI / 180 * 0.5);
  const cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw);
  // forward from yaw/pitch
  let fx = cp * sy, fy = sp, fz = cp * cy;
  // right = normalize(cross(forward, up)), up = (0,1,0)
  let rx = -fz, ry = 0, rz = fx;
  let rl = Math.hypot(rx, ry, rz) || 1; rx /= rl; rz /= rl;
  // up = cross(right, forward)
  const ux = ry * fz - rz * fy, uy = rz * fx - rx * fz, uz = rx * fy - ry * fx;
  const sx = 2 * aspect * tanFov / W, syc = -2 * tanFov / H;
  const M = new Float32Array(16);
  M[0] = sx * rx; M[1] = sx * ry; M[2] = sx * rz; M[3] = 0;
  M[4] = syc * ux; M[5] = syc * uy; M[6] = syc * uz; M[7] = 0;
  M[8]  = -aspect * tanFov * rx + tanFov * ux + fx;
  M[9]  = -aspect * tanFov * ry + tanFov * uy + fy;
  M[10] = -aspect * tanFov * rz + tanFov * uz + fz;
  M[11] = 0;
  M[12] = cam.pos[0]; M[13] = cam.pos[1]; M[14] = cam.pos[2]; M[15] = 1;
  return M;
}

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------
let lastT = performance.now(), frames = 0, fps = 0, fpsT = performance.now();

function frame() {
  const now = performance.now();
  const dt = Math.min(0.05, (now - lastT) / 1000);
  lastT = now;

  updateCamera(dt);
  if (BUILD.on) { refreshBuildHover(); uploadGhosts(); } else { ghostCount = 0; }
  streamingUpdate();

  if (!outTexture || outTexture.width !== renderW || outTexture.height !== renderH) buildTargets();

  // Uniforms
  const M = rayMatrix();
  const buf = new ArrayBuffer(96);
  const f = new Float32Array(buf), u = new Uint32Array(buf);
  f.set(M, 0);
  u[16] = WX; u[17] = WY; u[18] = WZ;
  f[19] = 0.9 / (renderDist * CH);                 // fogDensity ~ render distance
  const sd = normalize3([0.5, 0.78, 0.38]);
  f[20] = sd[0]; f[21] = sd[1]; f[22] = sd[2];
  f[23] = now / 1000;
  queue.writeBuffer(uniformBuf, 0, buf);

  const enc = device.createCommandEncoder();
  const cp = enc.beginComputePass();
  cp.setPipeline(computePipeline);
  cp.setBindGroup(0, computeBG);
  cp.dispatchWorkgroups(Math.ceil(renderW / 8), Math.ceil(renderH / 8), 1);
  cp.end();

  // Edge-outline post-pass: reads color + normal/depth G-buffer, writes edgeTexture.
  const ep = enc.beginComputePass();
  ep.setPipeline(edgePipeline);
  ep.setBindGroup(0, edgeBG);
  ep.dispatchWorkgroups(Math.ceil(renderW / 8), Math.ceil(renderH / 8), 1);
  ep.end();

  const canvasView = ctx.getCurrentTexture().createView();
  const rp = enc.beginRenderPass({
    colorAttachments: [{ view: canvasView, clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: 'clear', storeOp: 'store' }],
  });
  rp.setPipeline(blitPipeline);
  rp.setBindGroup(0, blitBG);
  rp.draw(3);
  rp.end();

  // Ghost block preview pass (translucent instanced cubes)
  if (ghostDepthTex && (ghostDepthTex.width !== canvas.width || ghostDepthTex.height !== canvas.height)) rebuildGhostDepth();
  if (BUILD.on && ghostCount > 0 && ghostPipeline && ghostDepthTex) {
    const {fwd} = camBasis();
    const eye = cam.pos;
    const aspect = canvas.width / canvas.height;
    const proj = mat4Perspective(cam.fov * Math.PI / 180, aspect, 0.5, WORLD_VX * 2.2);
    const view = mat4LookAt(eye, [eye[0]+fwd[0], eye[1]+fwd[1], eye[2]+fwd[2]], [0,1,0]);
    const vp   = mat4Mul(proj, view);
    const tint = BUILD.subMode==='terraform'
      ? (BUILD.tfTool==='erase'   ? [1.0,0.42,0.42,0.5]
       : BUILD.tfTool==='lower'   ? [1.0,0.72,0.3, 0.5]
       : BUILD.tfTool==='flatten' ? [0.9,0.9, 0.3, 0.5]
       :                            [0.4,1.0, 0.55,0.5])
      : (BUILD.tool==='remove'    ? [1.0,0.42,0.42,0.5] : [0.92,0.97,1.0,0.42]);
    const ub = new Float32Array(20); ub.set(vp, 0); ub.set(tint, 16);
    queue.writeBuffer(ghostUniBuf, 0, ub);

    const gp = enc.beginRenderPass({
      colorAttachments: [{ view: canvasView, loadOp: 'load', storeOp: 'store' }],
      depthStencilAttachment: { view: ghostDepthTex.createView(), depthLoadOp: 'clear', depthStoreOp: 'store', depthClearValue: 1.0 },
    });
    gp.setPipeline(ghostPipeline);
    gp.setBindGroup(0, ghostBG);
    gp.setVertexBuffer(0, ghostCubeBuf);
    gp.setVertexBuffer(1, ghostInstBuf);
    gp.draw(36, ghostCount);
    gp.end();
  }

  queue.submit([enc.finish()]);  // encoder finished after all passes above

  // HUD
  frames++;
  if (now - fpsT >= 400) {
    fps = Math.round(frames * 1000 / (now - fpsT));
    frames = 0; fpsT = now;
    updateHUD();
  }
  requestAnimationFrame(frame);
}

function buildTargets() {
  if (outTexture) outTexture.destroy();
  if (normDepthTexture) normDepthTexture.destroy();
  if (edgeTexture) edgeTexture.destroy();
  outTexture = device.createTexture({
    size: [renderW, renderH], format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  // G-buffer (packed normal + depth) written by the trace, read by the edge pass.
  normDepthTexture = device.createTexture({
    size: [renderW, renderH], format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  // Final outlined color, written by the edge pass and blitted to the canvas.
  edgeTexture = device.createTexture({
    size: [renderW, renderH], format: 'rgba8unorm',
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
  });
  computeBG = device.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: outTexture.createView() },
      { binding: 1, resource: { buffer: poolBuf } },
      { binding: 2, resource: { buffer: gridBuf } },
      { binding: 3, resource: { buffer: uniformBuf } },
      { binding: 4, resource: { buffer: colorPoolBuf } },
      { binding: 5, resource: normDepthTexture.createView() },
    ],
  });
  edgeBG = device.createBindGroup({
    layout: edgePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: outTexture.createView() },
      { binding: 1, resource: normDepthTexture.createView() },
      { binding: 2, resource: edgeTexture.createView() },
    ],
  });
  blitBG = device.createBindGroup({
    layout: blitPipeline.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: edgeTexture.createView() }],
  });
}

function updateCamera(dt) {
  if (autoFly) {
    cam.yaw += dt * 0.08;
    const {fwd} = camBasis();
    const fh = [fwd[0], 0, fwd[2]], fhl = Math.hypot(fh[0], fh[2]) || 1;
    cam.pos[0] += (fh[0]/fhl) * flySpeed * dt;
    cam.pos[2] += (fh[2]/fhl) * flySpeed * dt;
  }
  if (pointerLocked || BUILD.on) {
    const {fwd, r} = camBasis();
    const sp = (keys['shift'] ? 3.2 : 1) * flySpeed * dt;
    // Horizontal movement (ground-plane) matching old_sparse
    const fh = [fwd[0], 0, fwd[2]], fhl = Math.hypot(fh[0], fh[2]) || 1;
    if (keys['w']) { cam.pos[0] += (fh[0]/fhl)*sp; cam.pos[2] += (fh[2]/fhl)*sp; }
    if (keys['s']) { cam.pos[0] -= (fh[0]/fhl)*sp; cam.pos[2] -= (fh[2]/fhl)*sp; }
    if (keys['d']) { cam.pos[0] += r[0]*sp; cam.pos[2] += r[2]*sp; }
    if (keys['a']) { cam.pos[0] -= r[0]*sp; cam.pos[2] -= r[2]*sp; }
    if (keys[' '])       cam.pos[1] += sp;
    if (keys['control']) cam.pos[1] -= sp;
  }
  cam.pos[0] = clamp(cam.pos[0], 1, WORLD_VX - 1);
  cam.pos[1] = clamp(cam.pos[1], 1, WORLD_VY - 1);
  cam.pos[2] = clamp(cam.pos[2], 1, WORLD_VZ - 1);
}

function updateHUD() {
  const now = performance.now();
  while (chunkRateWindow.length && now - chunkRateWindow[0] > 1000) chunkRateWindow.shift();
  const cps = chunkRateWindow.length;
  const avgMs = genMsCount ? (genMsAccum / genMsCount) : 0;
  const busy = workerBusy.reduce((a, b) => a + (b ? 1 : 0), 0);
  $('stats').innerHTML =
    `FPS <b>${fps}</b> &nbsp; workers <b>${busy}/${workers.length}</b> busy<br>` +
    `Voxels generated: <span class="big">${cumVoxels.toLocaleString()}</span><br>` +
    `World volume: <b>${WORLD_VOXELS.toLocaleString()}</b><br>` +
    `Chunks generated: <b>${chunksGen.toLocaleString()}</b><br>` +
    `Loading rate: <b>${cps}</b> chunks/s · gen <b>${avgMs.toFixed(2)}</b> ms avg<br>` +
    `Resident: <b>${residentMixed}</b>/${MAX_RESIDENT} slots · queue <b>${(inFlight + pendingUploads.length)}</b><br>` +
    `Pos: <b>${cam.pos.map(v => v | 0).join(', ')}</b>`;
}

// ---------------------------------------------------------------------------
// Input + UI
// ---------------------------------------------------------------------------
function setupInput() {
  canvas.addEventListener('mousedown', e => {
    if (BUILD.on) {
      e.preventDefault();
      if (e.button === 0) {
        if (BUILD.subMode === 'terraform') {
          BUILD.tfPainting = true;
          // Lock the Flatten target height to the first click, so holding/dragging
          // carves a flat bed (e.g. a river) at one consistent level instead of
          // chasing the cursor's height every frame.
          if (BUILD.tfTool === 'flatten') { const rc = raycastVoxel(); BUILD.tfFlattenY = rc ? rc.hit[1] : null; }
          applyTerraform();
        }
        else buildClick();
      } else if (e.button === 2) { BUILD.dragLook = true; }
    } else if (e.button === 0 && !autoFly) { canvas.requestPointerLock(); }
  });
  window.addEventListener('mouseup', e => {
    if (e.button === 2) BUILD.dragLook = false;
    if (e.button === 0) { BUILD.tfPainting = false; BUILD.tfFlattenY = null; }
  });
  canvas.addEventListener('contextmenu', e => { if (BUILD.on) e.preventDefault(); });
  document.addEventListener('pointerlockchange', () => {
    pointerLocked = (document.pointerLockElement === canvas);
    $('cross').style.display = (pointerLocked || BUILD.on) ? 'block' : 'none';
    if (pointerLocked) fadeHint();
  });
  document.addEventListener('mousemove', e => {
    if (BUILD.on) {
      if (BUILD.dragLook) {
        cam.yaw   -= e.movementX * 0.0026;
        cam.pitch -= e.movementY * 0.0026;
        cam.pitch  = Math.max(-1.5, Math.min(1.5, cam.pitch));
      } else {
        // Update aim cursor in NDC so raycast goes through mouse position
        const r = canvas.getBoundingClientRect();
        BUILD.aim.x = ((e.clientX - r.left) / r.width)  * 2 - 1;
        BUILD.aim.y = 1 - ((e.clientY - r.top)  / r.height) * 2;
        if (BUILD.subMode === 'terraform' && BUILD.tfPainting) applyTerraform();
      }
      refreshBuildHover();
      return;
    }
    if (!pointerLocked) return;
    cam.yaw   -= e.movementX * 0.0022;
    cam.pitch -= e.movementY * 0.0022;
    const L = Math.PI / 2 - 0.02;
    cam.pitch = Math.max(-L, Math.min(L, cam.pitch));
  });
  window.addEventListener('keydown', e => {
    const k = e.key.toLowerCase();
    if (k === 'b') { setBuildMode(!BUILD.on); e.preventDefault(); return; }
    if (k === 'g') { clearEdits(); seed = (Math.random() * 1e6) | 0; resetWorld(); e.preventDefault(); return; }
    if (k === 'o') { exportWorld(); e.preventDefault(); return; }
    if (k === 'i' && !BUILD.on) { importWorldFile(); e.preventDefault(); return; }
    if (BUILD.on && k === 'escape') { buildCancel(); e.preventDefault(); return; }
    // Architecture Planner ghost nudge (only when anchored)
    if (BUILD.on && BUILD.subMode === 'planner' && BUILD.apAnchored) {
      if (k === 'arrowleft'  || k === 'j') { apMoveGhost(-1, 0,  0); e.preventDefault(); return; }
      if (k === 'arrowright' || k === 'l') { apMoveGhost( 1, 0,  0); e.preventDefault(); return; }
      if (k === 'arrowup'    || k === 'i') { apMoveGhost( 0, 0, -1); e.preventDefault(); return; }
      if (k === 'arrowdown'  || k === 'k') { apMoveGhost( 0, 0,  1); e.preventDefault(); return; }
      if (k === 'u') { apMoveGhost(0,  1, 0); e.preventDefault(); return; }
      if (k === 'o') { apMoveGhost(0, -1, 0); e.preventDefault(); return; }
    }
    keys[k] = true; if (e.key === ' ') e.preventDefault();
  });
  window.addEventListener('keyup',   e => { keys[e.key.toLowerCase()] = false; });
}

function setupUI() {
  const dist = $('dist'), scale = $('scale'), speed = $('speed');
  dist.addEventListener('input', () => { renderDist = +dist.value; $('distVal').textContent = renderDist; });
  scale.addEventListener('input', () => { renderScale = +scale.value / 100; $('scaleVal').textContent = scale.value + '%'; resize(); });
  speed.addEventListener('input', () => { flySpeed = +speed.value; $('speedVal').textContent = flySpeed; });
  $('autofly').addEventListener('click', () => {
    autoFly = !autoFly;
    $('autofly').classList.toggle('on', autoFly);
    if (autoFly && pointerLocked) document.exitPointerLock();
    fadeHint();
  });
  $('regen').addEventListener('click', () => { clearEdits(); seed = (Math.random() * 1e6) | 0; resetWorld(); });
  // Controls hints popup: dismiss with the ✕, reopen via the small tab it leaves behind.
  const keyhints = $('keyhints'), khReopen = $('keyhints-reopen');
  $('kh-close').addEventListener('click', () => { keyhintsDismissed = true; keyhints.classList.add('hide'); khReopen.style.display = 'block'; });
  khReopen.addEventListener('click', () => { keyhintsDismissed = false; keyhints.classList.remove('hide'); khReopen.style.display = 'none'; });
  setupBuildUI();
}

function fadeHint() { if (!hintFaded) { hintFaded = true; $('hint').classList.add('fade'); } }

function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  renderW = Math.max(1, Math.floor(window.innerWidth  * dpr * renderScale));
  renderH = Math.max(1, Math.floor(window.innerHeight * dpr * renderScale));
  canvas.width = renderW; canvas.height = renderH;
  if (device && ctx) ctx.configure({ device, format: canvasFormat, alphaMode: 'opaque' });
  // Drop the render targets; buildTargets() recreates all three on the next frame
  // (the !outTexture guard re-triggers it after this null).
  if (outTexture) { outTexture.destroy(); outTexture = null; }
  if (normDepthTexture) { normDepthTexture.destroy(); normDepthTexture = null; }
  if (edgeTexture) { edgeTexture.destroy(); edgeTexture = null; }
  if (device && ghostPipeline) rebuildGhostDepth();
}


const _heightCache = new Map();
function terrainHeightCPU(wx,wz){
  const k = (wx|0) + ',' + (wz|0);
  const c = _heightCache.get(k);
  if (c !== undefined) return c;
  const ih=(x,z,sd)=>{let h=Math.imul(x|0,374761393)^Math.imul(z|0,668265263)^Math.imul(sd>>>0,2246822519); h=Math.imul(h^(h>>>15),2246822519); h^=h>>>13; h=Math.imul(h,3266489917); h^=h>>>16; return (h>>>0)/4294967296;};
  const vn=(x,z,sd)=>{const xi=Math.floor(x),zi=Math.floor(z),xf=x-xi,zf=z-zi,u=xf*xf*(3-2*xf),v=zf*zf*(3-2*zf); const a=ih(xi,zi,sd),b=ih(xi+1,zi,sd),c=ih(xi,zi+1,sd),d=ih(xi+1,zi+1,sd); return a*(1-u)*(1-v)+b*u*(1-v)+c*(1-u)*v+d*u*v;};
  const fbm=(x,z,sd,oct=5)=>{let s=0,amp=1,f=1,norm=0; for(let o=0;o<oct;o++){s+=amp*vn(x*f,z*f,sd+o*1311);norm+=amp;amp*=0.5;f*=2;} return s/norm;};
  const rid=(x,z,sd,oct=5)=>{let s=0,amp=1,f=1,norm=0; for(let o=0;o<oct;o++){let n=vn(x*f,z*f,sd+o*1777); n=1-Math.abs(2*n-1); n*=n; s+=amp*n; norm+=amp; amp*=0.5; f*=2;} return s/norm;};
  // Must match the worker's height() exactly. plainsBase≈40 (chair-scale reference),
  // mountainTop=380 for towering Minecraft-scale peaks; just under the 416 (WY*CH) ceiling.
  const plainsBase=Math.round(128*0.31), mountainTop=380;
  const raw=fbm(wx*0.0022+3.7,wz*0.0022+1.9,seed,3), keepFade=clamp((Math.hypot(wx-WORLD_VX/2,wz-WORLD_VZ/2)-120)/96,0,1), b=raw*keepFade;
  const blend=clamp((b-0.62)/0.10,0,1), blendS=blend*blend*(3-2*blend);
  let result;
  if(blendS<0.001){ result=plainsBase; }
  else {
    // Cliffy plateau (mesa) shaping — MUST match the worker's height() in shaders.js.
    // FBM blobs + low exponent = broad massifs; terracing quantizes the slope into flat
    // shelves with vertical cliff faces between them (large flat tops, cliffy step-ups).
    const shape=fbm(wx*0.006+1.3,wz*0.006+2.7,seed,5)*0.72+fbm(wx*0.015+5.1,wz*0.015+3.9,seed,4)*0.28;
    const elev=Math.pow(Math.max(0,shape-0.12)/0.88,0.85);
    let mtnH=plainsBase+(mountainTop-plainsBase)*elev;
    const STEP=26, lvl=Math.floor(mtnH/STEP), frac=mtnH/STEP-lvl, cliff=Math.pow(frac,6);
    mtnH=(lvl+cliff)*STEP + fbm(wx*0.06+7.1,wz*0.06+4.3,seed,3)*3-1.5;
    result=plainsBase*(1-blendS)+mtnH*blendS;
  }
  if(_heightCache.size > 65536) _heightCache.clear(); // limit cache size
  _heightCache.set(k, result);
  return result;
}
function solidAtEdited(x,y,z){ if(x<0||y<0||z<0||x>=WORLD_VX||y>=WORLD_VY||z>=WORLD_VZ) return false; const e=EDITS.get(editKey(x,y,z)); if(e!==undefined) return e.s===1; return y<=Math.floor(terrainHeightCPU(x,z)); }
function camBasis(){
  const cp=Math.cos(cam.pitch), sp=Math.sin(cam.pitch);
  const cy=Math.cos(cam.yaw),   sy=Math.sin(cam.yaw);
  const fwd=[cp*sy, sp, cp*cy];
  let rx=-fwd[2], ry=0, rz=fwd[0];
  const rl=Math.hypot(rx,ry,rz)||1; rx/=rl; rz/=rl;
  const r=[rx,ry,rz];
  const up=[ry*fwd[2]-rz*fwd[1], rz*fwd[0]-rx*fwd[2], rx*fwd[1]-ry*fwd[0]];
  return {fwd, r, up};
}
function aimRay(){
  // Ray through the cursor position — exactly matches the GPU compute shader ray-gen.
  // GPU: d = M * [px+0.5, py+0.5, 1, 0]  where px,py are pixel coords.
  // CPU: cursor NDC (aim.x, aim.y) → pixel (px,py) = ((aim.x+1)/2*W, (1-aim.y)/2*H)
  // Simplifies to: d = fwd + aim.x*aspect*tanFov*right + aim.y*tanFov*up  (same as old_sparse)
  const {fwd,r,up}=camBasis();
  const aspect=renderW/renderH, th=Math.tan(cam.fov*Math.PI/360);
  const nx=BUILD.on?BUILD.aim.x:0, ny=BUILD.on?BUILD.aim.y:0;
  const dx=fwd[0]+nx*aspect*th*r[0]+ny*th*up[0];
  const dy=fwd[1]+nx*aspect*th*r[1]+ny*th*up[1];
  const dz=fwd[2]+nx*aspect*th*r[2]+ny*th*up[2];
  const L=Math.hypot(dx,dy,dz)||1;
  return { ro: cam.pos.slice(), rd: [dx/L, dy/L, dz/L] };
}
function raycastVoxel(){
  const {ro,rd}=aimRay();
  let px=ro[0],py=ro[1],pz=ro[2];
  // step slightly forward so we don't immediately hit the voxel the camera is inside
  px+=rd[0]*0.01; py+=rd[1]*0.01; pz+=rd[2]*0.01;
  let ix=Math.floor(px),iy=Math.floor(py),iz=Math.floor(pz);
  const sx=rd[0]>=0?1:-1,sy=rd[1]>=0?1:-1,sz=rd[2]>=0?1:-1;
  const tdx=Math.abs(1/(rd[0]||1e-9)),tdy=Math.abs(1/(rd[1]||1e-9)),tdz=Math.abs(1/(rd[2]||1e-9));
  let txm=((sx>0?ix+1-px:px-ix))*tdx,tym=((sy>0?iy+1-py:py-iy))*tdy,tzm=((sz>0?iz+1-pz:pz-iz))*tdz;
  let nfx=0,nfy=0,nfz=0,tCur=0;
  for(let i=0;i<900;i++){
    if(solidAtEdited(ix,iy,iz)) return {hit:[ix,iy,iz],place:[ix-nfx,iy-nfy,iz-nfz],face:[nfx,nfy,nfz],pos:[px+rd[0]*tCur,py+rd[1]*tCur,pz+rd[2]*tCur]};
    if(txm<tym&&txm<tzm){tCur=txm;ix+=sx;txm+=tdx;nfx=sx;nfy=0;nfz=0;}
    else if(tym<tzm){tCur=tym;iy+=sy;tym+=tdy;nfx=0;nfy=sy;nfz=0;}
    else{tCur=tzm;iz+=sz;tzm+=tdz;nfx=0;nfy=0;nfz=sz;}
  }
  return null;
}
function refreshBuildHover(){
  if(!BUILD.on) return;
  const rc=raycastVoxel(); BUILD._lastRC=rc; BUILD._lastHit=rc?rc.hit:null; BUILD.hover=rc?rc.place:null;
  const ht=$('bd-hover'); if(ht) ht.textContent=BUILD.hover?BUILD.hover.join(', '):'—';
}
function applyVoxels(list, remove){
  const mat = BUILD_MATS[BUILD.matIndex] || BUILD_MATS[0];
  const touched=[];
  for(const v of list){
    const [x,y,z]=v.map(n=>Math.floor(n));
    if(x<0||y<0||z<0||x>=WORLD_VX||y>=WORLD_VY||z>=WORLD_VZ) continue;
    if(remove) setEdit(x,y,z, {s:0,r:0,g:0,b:0});
    else       setEdit(x,y,z, {s:1,r:mat.r,g:mat.g,b:mat.b});
    touched.push([x,y,z]);
  }
  if(touched.length) rebuildTouchedVoxels(touched);
  return touched.length;
}
function buildClick(){
  refreshBuildHover();
  const subMode = BUILD.subMode || 'native';
  if (subMode === 'planner') { apBuildClick(); return; }
  if (subMode === 'terraform') { applyTerraform(); return; }
  // native mode
  const rc=BUILD._lastRC; if(!rc) return; const T=BUILD.tool;
  if(T==='block'){applyVoxels([rc.place],false); return;}
  if(T==='remove'){applyVoxels([rc.hit],true); return;}
  if(T==='circle'){ if(!BUILD.circle){BUILD.circle={step:1,cx:rc.place[0]+0.5,cz:rc.place[2]+0.5,baseY:rc.place[1],radius:null,thick:null,height:null}; return;} const C=BUILD.circle,cur=circleCursorXZ(rc.place); if(C.step===1){C.radius=Math.max(1,Math.hypot(cur[0]-C.cx,cur[1]-C.cz)); C.step=2; return;} if(C.step===2){const d=Math.hypot(cur[0]-C.cx,cur[1]-C.cz); C.thick=Math.max(1,Math.min(C.radius,C.radius-d)); C.step=3; return;} C.height=Math.max(1,Math.round(Math.hypot(cur[0]-C.cx,cur[1]-C.cz))); applyVoxels(genCircleTower(circleParams(rc.place)),false); BUILD.circle=null; return;}
  if(T==='square'){ if(!BUILD.square){BUILD.square={step:1,x0:rc.place[0],z0:rc.place[2],x1:null,z1:null,baseY:rc.place[1],thick:null,height:null}; return;} const S=BUILD.square,cur=squareCursorXZ(rc.place),ox=Math.round(cur[0]-0.5),oz=Math.round(cur[1]-0.5); if(S.step===1){S.x1=ox;S.z1=oz;S.step=2;return;} if(S.step===2){const p=squareParams(rc.place); S.thick=p?p.thick:1; S.step=3; return;} applyVoxels(genSquareRoom(squareParams(rc.place)),false); BUILD.square=null; return;}
  if(T==='arc'){ if(!BUILD.anchor) BUILD.anchor=rc.place; else if(!BUILD.mid) BUILD.mid=rc.place; else {applyVoxels(genGhosts(),false); BUILD.anchor=null; BUILD.mid=null;} return;}
  if(!BUILD.anchor) BUILD.anchor=rc.place; else {applyVoxels(genGhosts(),false); BUILD.anchor=null;}
}
function buildCancel(){ BUILD.anchor=null; BUILD.mid=null; BUILD.circle=null; BUILD.square=null; BUILD.apGhost=null; BUILD.apAnchored=false; BUILD.tfPainting=false; BUILD.tfFlattenY=null; BUILD.dragLook=false; refreshBuildHover(); }
function setBuildMode(on){
  BUILD.on=on; buildCancel();
  BUILD.aim.x=0; BUILD.aim.y=0;
  if (on) { document.exitPointerLock(); for (const k in keys) keys[k] = false; }
  $('buildbar').style.display=on?'block':'none'; $('bd-badge').style.display=on?'block':'none';
  $('cross').style.display=on?'block':(pointerLocked?'block':'none');
  canvas.classList.toggle('build',on); refreshBuildHover();
}
function selectTool(t){
  BUILD.tool=t; buildCancel();
  document.querySelectorAll('#buildbar .tool').forEach(el=>el.classList.toggle('active',el.dataset.tool===t));
  const showH=t!=='block'&&t!=='remove', showT=t==='wall'||t==='arc'||t==='circle'||t==='square', showR=t==='disc'||t==='circle'||t==='square'||t==='rect';
  $('row-h').style.display=showH?'':'none'; $('row-t').style.display=showT?'':'none'; $('row-r').style.display=showR?'':'none';
  $('bd-tool').textContent=t.toUpperCase();
}
function selectMat(i){ BUILD.matIndex=i; document.querySelectorAll('#buildbar .swatch').forEach((el,idx)=>el.classList.toggle('active',idx===i)); $('bd-mat').textContent=BUILD_MATS[i].name; }

/* ---- Architecture Planner sub-mode ---- */
BUILD.subMode    = 'native';
BUILD.apGhost    = null;   // [x,y,z] ghost voxel position
BUILD.apAnchored = false;  // false = follows cursor; true = keyboard-driven

function setBuildSubMode(mode){
  BUILD.subMode = mode; buildCancel();
  $('tab-native').classList.toggle('active', mode==='native');
  $('tab-planner').classList.toggle('active', mode==='planner');
  $('tab-terraform').classList.toggle('active', mode==='terraform');
  $('native-tools').style.display   = mode==='native'    ? '' : 'none';
  $('planner-tools').style.display  = mode==='planner'   ? '' : 'none';
  $('terraform-tools').style.display= mode==='terraform' ? '' : 'none';
  $('ap-pos').style.display         = mode==='planner'   ? '' : 'none';
  $('native-hint').style.display    = mode==='native'    ? '' : 'none';
  $('planner-hint').style.display   = mode==='planner'   ? '' : 'none';
  $('terraform-hint').style.display = mode==='terraform' ? '' : 'none';
  if(mode==='terraform'){
    // Terraform: raise/lower work one voxel per stroke (like old_sparse); only Radius
    // (brush size) matters — hide Height & Thickness.
    $('row-h').style.display='none'; $('row-t').style.display='none'; $('row-r').style.display='';
    if(!BUILD.tfTool) BUILD.tfTool='raise'; selectTfTool(BUILD.tfTool);
  } else { selectTool(BUILD.tool||'wall'); }
}

function genPlannerGhosts(){
  if(!BUILD.apGhost) return [];
  const [gx,gy,gz]=BUILD.apGhost, T=BUILD.tool, hgt=BUILD.height, th=BUILD.thick, rad=BUILD.radius;
  if(T==='block') return [[gx,gy,gz]];
  if(T==='remove'){ const rc=BUILD._lastRC; return rc?[rc.hit]:[]; }
  if(T==='wall') return raiseColumns(thicken([[gx,gz]],th),gy,hgt);
  if(T==='rect'){ const hw=Math.floor(rad/2), cols=[]; for(let z=gz-hw;z<=gz+hw;z++) for(let x=gx-hw;x<=gx+hw;x++) cols.push([x,z]); return raiseColumns(cols,gy,hgt); }
  if(T==='disc'){ const cols=[]; for(let z=-rad;z<=rad;z++) for(let x=-rad;x<=rad;x++) if(Math.hypot(x,z)<=rad+0.5) cols.push([gx+x,gz+z]); return raiseColumns(cols,gy,hgt); }
  if(T==='circle') return genCircleTower({cx:gx+0.5,cz:gz+0.5,baseY:gy,radius:rad,thick:th,height:hgt});
  if(T==='square'){ const hw=Math.floor(rad/2); return genSquareRoom({x0:gx-hw,x1:gx+hw,z0:gz-hw,z1:gz+hw,baseY:gy,thick:th,height:hgt}); }
  return [];
}
function apBuildClick(){
  if(!BUILD.apAnchored){
    // 1st click: anchor ghost at current raycast hit
    const rc=BUILD._lastRC; if(!rc) return;
    BUILD.apGhost=[rc.place[0],rc.place[1],rc.place[2]]; BUILD.apAnchored=true;
    const el=$('ap-xyz'); if(el) el.textContent=BUILD.apGhost.join(', ');
  } else {
    // 2nd click: build at anchored ghost
    const voxels=genPlannerGhosts();
    if(voxels.length) applyVoxels(voxels,false);
    BUILD.apGhost=null; BUILD.apAnchored=false;
  }
}
function apMoveGhost(dx,dy,dz){
  if(!BUILD.apAnchored||!BUILD.apGhost) return;
  BUILD.apGhost[0]=clamp(BUILD.apGhost[0]+dx,0,WORLD_VX-1);
  BUILD.apGhost[1]=clamp(BUILD.apGhost[1]+dy,0,WORLD_VY-1);
  BUILD.apGhost[2]=clamp(BUILD.apGhost[2]+dz,0,WORLD_VZ-1);
  const el=$('ap-xyz'); if(el) el.textContent=BUILD.apGhost.join(', ');
}

/* ---- Terraform sub-mode ---- */
BUILD.tfTool     = 'raise';
BUILD.tfPainting = false;
BUILD.tfFlattenY = null;   // Flatten target height locked for the duration of a drag

function selectTfTool(t){
  BUILD.tfTool=t;
  document.querySelectorAll('#terraform-tools .tf-tool').forEach(el=>el.classList.toggle('active',el.dataset.tftool===t));
  $('bd-tool').textContent=t.toUpperCase();
}
function tfTopSolid(x,z){
  // Find the true top of the column, including a tall stack of raised edits.
  //
  // The naive "scan from WORLD_VY-1 down to 0" version cost ~WORLD_VY solidAtEdited
  // calls per column, every frame while painting — brutal once the ceiling grew, and
  // it forced a noise eval on the first (uncached) column. Instead, start just above
  // the natural terrain height and only climb higher when a genuinely tall stack of
  // edits is actually present. The common case scans ~64 rows instead of the full
  // ceiling, while Raise can still build arbitrarily high.
  const SCAN_MARGIN = 64;
  let start = Math.floor(terrainHeightCPU(x,z)) + SCAN_MARGIN;
  if(start >= WORLD_VY) start = WORLD_VY - 1;
  // If the start row is itself solid, raised edits may extend above it — climb until
  // we clear them so the returned top is the real top, not the scan window's edge.
  while(start < WORLD_VY-1 && solidAtEdited(x,start,z)) start++;
  for(let y=start; y>=0; y--) if(solidAtEdited(x,y,z)) return y;
  return -1;
}
function tfBrushXZ(cx,cz){ const R=BUILD.radius,cols=[]; for(let dz=-R;dz<=R;dz++) for(let dx=-R;dx<=R;dx++) if(Math.hypot(dx,dz)<=R+0.5) cols.push([cx+dx,cz+dz]); return cols; }
function genTerraformGhosts(){
  const rc=BUILD._lastRC; if(!rc) return [];
  const [cx,cy,cz]=rc.hit;
  if(BUILD.tfTool==='erase'){
    const R=BUILD.radius,ghosts=[];
    for(let dz=-R;dz<=R;dz++) for(let dy=-R;dy<=R;dy++) for(let dx=-R;dx<=R;dx++){
      const d=Math.hypot(dx,dy,dz); if(d>R+0.5||d<R-0.5) continue;
      const y=cy+dy; if(y>=0&&y<WORLD_VY) ghosts.push([cx+dx,y,cz+dz]);
    }
    return ghosts;
  }
  const cols=tfBrushXZ(cx,cz), ghosts=[];
  for(const [x,z] of cols){ const top=tfTopSolid(x,z); if(top>=0) ghosts.push([x,top+1,z]); }
  return ghosts;
}
function applyTerraform(){
  const rc=raycastVoxel(); if(!rc) return;
  const [hx,hy,hz]=rc.hit; const cols=tfBrushXZ(hx,hz); const T=BUILD.tfTool;
  if(T==='raise'){ const list=[]; for(const [x,z] of cols){ const top=tfTopSolid(x,z); if(top>=0&&top+1<WORLD_VY) list.push([x,top+1,z]); } if(list.length) applyVoxels(list,false); }
  else if(T==='lower'){ const list=[]; for(const [x,z] of cols){ const top=tfTopSolid(x,z); if(top>=0) list.push([x,top,z]); } if(list.length) applyVoxels(list,true); }
  else if(T==='flatten'){
    // Use the height locked at mousedown while dragging, so the whole stroke flattens
    // to one level; fall back to the current hover height for a fresh single click.
    const tY=(BUILD.tfFlattenY!=null)?BUILD.tfFlattenY:hy; const toAdd=[],toRem=[];
    for(const [x,z] of cols){ const top=tfTopSolid(x,z); if(top<0) continue; if(top<tY) for(let y=top+1;y<=tY;y++) toAdd.push([x,y,z]); else if(top>tY) for(let y=tY+1;y<=top;y++) toRem.push([x,y,z]); }
    if(toAdd.length) applyVoxels(toAdd,false); if(toRem.length) applyVoxels(toRem,true);
  } else if(T==='erase'){
    const R=BUILD.radius,list=[];
    for(let dz=-R;dz<=R;dz++) for(let dy=-R;dy<=R;dy++) for(let dx=-R;dx<=R;dx++){
      if(Math.hypot(dx,dy,dz)>R+0.5) continue; const y=hy+dy; if(y<0||y>=WORLD_VY) continue;
      if(solidAtEdited(hx+dx,y,hz+dz)) list.push([hx+dx,y,hz+dz]);
    }
    if(list.length) applyVoxels(list,true);
  }
}

function setupBuildUI(){
  document.querySelectorAll('#buildbar .bb-tab').forEach(el=>el.addEventListener('click',()=>setBuildSubMode(el.dataset.submode)));
  document.querySelectorAll('#buildbar .tool').forEach(el=>el.addEventListener('click',()=>selectTool(el.dataset.tool)));
  document.querySelectorAll('#terraform-tools .tf-tool').forEach(el=>el.addEventListener('click',()=>selectTfTool(el.dataset.tftool)));
  const sw=$('swatches'); if(sw&&!sw.childElementCount){ BUILD_MATS.forEach((m,i)=>{const d=document.createElement('div'); d.className='swatch'; d.title=m.name; d.style.background=`rgb(${m.r},${m.g},${m.b})`; d.addEventListener('click',()=>selectMat(i)); sw.appendChild(d);}); }
  $('bd-hh').addEventListener('input',e=>{BUILD.height=+e.target.value;$('bd-hv').textContent=BUILD.height;});
  $('bd-tt').addEventListener('input',e=>{BUILD.thick=+e.target.value;$('bd-tv').textContent=BUILD.thick;});
  $('bd-r').addEventListener('input',e=>{BUILD.radius=+e.target.value;$('bd-rv').textContent=BUILD.radius;});
  // Scroll wheel: adjust height/radius in build mode, FOV in fly mode
  canvas.addEventListener('wheel', e => {
    if (BUILD.on) {
      const dir = -Math.sign(e.deltaY);
      if (BUILD.subMode === 'terraform') {
        BUILD.radius = Math.max(1, Math.min(64, BUILD.radius + dir));
        const s = $('bd-r'); if (s) { s.value = BUILD.radius; $('bd-rv').textContent = BUILD.radius; }
      } else if (BUILD.subMode === 'planner') {
        if (e.shiftKey) {
          BUILD.thick = Math.max(1, Math.min(6, BUILD.thick + dir));
          const s = $('bd-tt'); if (s) { s.value = BUILD.thick; $('bd-tv').textContent = BUILD.thick; }
        } else if (BUILD.tool==='disc'||BUILD.tool==='circle'||BUILD.tool==='square'||BUILD.tool==='rect') {
          BUILD.radius = Math.max(1, Math.min(64, BUILD.radius + dir));
          const s = $('bd-r'); if (s) { s.value = BUILD.radius; $('bd-rv').textContent = BUILD.radius; }
        } else {
          BUILD.height = Math.max(1, Math.min(48, BUILD.height + dir));
          const s = $('bd-hh'); if (s) { s.value = BUILD.height; $('bd-hv').textContent = BUILD.height; }
        }
      } else {
        BUILD.height = Math.max(1, Math.min(48, BUILD.height + dir));
        const s = $('bd-hh'); if (s) { s.value = BUILD.height; $('bd-hv').textContent = BUILD.height; }
      }
      refreshBuildHover();
      e.preventDefault();
      return;
    }
    cam.fov = Math.max(40, Math.min(95, cam.fov + Math.sign(e.deltaY) * 2));
  }, { passive: false });
  selectTool(BUILD.tool); selectMat(0);
}
function serializeWorld(){
  const edits=[];
  for(const [k,v] of EDITS){ const [x,y,z]=parseEditKey(k); edits.push([x,y,z,v.s,v.r??0,v.g??0,v.b??0]); }
  return {version:SAVE_VERSION,seed:seed>>>0,player:{pos:cam.pos,yaw:cam.yaw,pitch:cam.pitch},edits};
}
function exportWorld(){ const data=serializeWorld(), blob=new Blob([JSON.stringify(data)],{type:'application/json'}), url=URL.createObjectURL(blob), a=document.createElement('a'); a.href=url; a.download=`fast-voxel-${data.seed}-${data.edits.length}edits.json`; document.body.appendChild(a); a.click(); a.remove(); setTimeout(()=>URL.revokeObjectURL(url),1000); }
async function loadWorld(save){
  if(!save||typeof save.seed!=='number') return;
  clearEdits(); seed=save.seed>>>0;
  for(const row of save.edits||[]){
    if(row.length>=4){
      const s=row[3]?1:0, r=row[4]??0, g=row[5]??0, b=row[6]??0;
      setEdit(row[0],row[1],row[2], {s,r,g,b});
    }
  }
  resetWorld();
  if(save.player){ cam.pos=save.player.pos||cam.pos; cam.yaw=save.player.yaw??cam.yaw; cam.pitch=save.player.pitch??cam.pitch; }
}
function importWorldFile(){ const inp=document.createElement('input'); inp.type='file'; inp.accept='application/json,.json'; inp.addEventListener('change',()=>{const f=inp.files&&inp.files[0]; if(!f)return; const rd=new FileReader(); rd.onload=()=>{try{loadWorld(JSON.parse(rd.result));}catch(e){console.error(e);}}; rd.readAsText(f);}); inp.click(); }

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------
function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }
function normalize3(v) { const l = Math.hypot(v[0], v[1], v[2]) || 1; return [v[0] / l, v[1] / l, v[2] / l]; }

init().then(() => requestAnimationFrame(frame)).catch(err => { console.error('init failed:', err); });
