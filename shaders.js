"use strict";

export const SHADER_SRC = /* wgsl */`
struct U {
  rayMat      : mat4x4<f32>,
  worldChunks : vec3<u32>,
  fogDensity  : f32,
  sunDir      : vec3<f32>,
  time        : f32,
};

@group(0) @binding(0) var outImage   : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read> pool      : array<u32>;
@group(0) @binding(2) var<storage, read> grid      : array<u32>;
@group(0) @binding(3) var<uniform>       uni       : U;
@group(0) @binding(4) var<storage, read> colorPool : array<u32>;
// G-buffer: packed surface normal (xyz, 0..1) + normalized depth (a). 1.0 = sky.
@group(0) @binding(5) var normDepth  : texture_storage_2d<rgba8unorm, write>;

const CH    : f32 = 32.0;
const CHI   : u32 = 32u;
const FULLC : u32 = 0xfffffffdu;     // states >= FULLC are sentinels, else slot

fn cellIndex(c : vec3<i32>) -> u32 {
  let wc = uni.worldChunks;
  return (u32(c.z) * wc.y + u32(c.y)) * wc.x + u32(c.x);
}

fn readLocal(slot : u32, l : vec3<i32>) -> bool {
  let w = slot * 1024u + u32(l.z) * 32u + u32(l.y);
  let bit = u32(l.x);
  return ((pool[w] >> bit) & 1u) != 0u;
}

fn readColor(slot : u32, l : vec3<i32>) -> u32 {
  // colorPool layout mirrors bitmask: slot * 1024 + lz*32 + ly
  // One color per (lz,ly) word — shared by all 32 voxels along X in that word.
  let idx = slot * 1024u + u32(l.z) * 32u + u32(l.y);
  return colorPool[idx];  // 0 = terrain color, else top byte 0xFF + 0x00RRGGBB
}

struct Hit { hit : bool, t : f32, n : vec3<f32>, wp : vec3<f32>, editColor : u32 };

fn skyColor(dir : vec3<f32>) -> vec3<f32> {
  let up = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
  let horizon = vec3<f32>(0.72, 0.81, 0.92);
  let zenith  = vec3<f32>(0.26, 0.46, 0.86);
  var c = mix(horizon, zenith, up);
  let s = max(dot(dir, uni.sunDir), 0.0);
  c += vec3<f32>(1.0, 0.92, 0.72) * pow(s, 64.0) * 0.9;
  return c;
}

fn terrainColor(wy : f32, n : vec3<f32>) -> vec3<f32> {
  let snow  = vec3<f32>(0.94, 0.96, 1.00);
  let rock  = vec3<f32>(0.40, 0.39, 0.37);
  let grass = vec3<f32>(0.38, 0.56, 0.26);
  let dirt  = vec3<f32>(0.47, 0.34, 0.22);
  // Bands scaled to the towering peaks (plainsBase≈40, mountainTop=380): snow caps the
  // high peaks, rock below, grass on the lower slopes/plains. Thresholds placed
  // proportionally across the 340-voxel plains→peak span (grass/rock/snow at ~21/45/69%).
  var c = grass;
  if (wy > 275.0)      { c = snow; }
  else if (wy > 193.0) { c = mix(rock, snow, clamp((wy - 193.0) / 82.0, 0.0, 1.0)); }
  else if (wy > 111.0) { c = mix(grass, rock, clamp((wy - 111.0) / 82.0, 0.0, 1.0)); }
  if (n.y < 0.5) { c = mix(c, dirt, select(0.35, 0.62, wy < 115.0)); }
  return c;
}

fn innerDDA(o : vec3<f32>, dir : vec3<f32>, inv : vec3<f32>, stepi : vec3<i32>,
            cell : vec3<i32>, slot : u32, tEntry : f32, entryAxis : i32) -> Hit {
  var miss : Hit; miss.hit = false; miss.editColor = 0u; miss.wp = vec3<f32>(0.0);
  let base = vec3<f32>(cell) * CH;
  let pe = o + dir * (tEntry + 1e-3) - base;
  var l = vec3<i32>(clamp(floor(pe), vec3<f32>(0.0), vec3<f32>(31.0)));
  let vb = base + vec3<f32>(l) + select(vec3<f32>(0.0), vec3<f32>(1.0), dir > vec3<f32>(0.0));
  var tMax = (vb - o) * inv;
  let tDelta = abs(inv);
  var axis = entryAxis;
  var t = tEntry;
  // If the ray started inside this chunk (entryAxis == -1), skip the first voxel
  // so we don't render the terrain block the camera is embedded in.
  var skipFirst = (entryAxis < 0);
  for (var k : u32 = 0u; k < 128u; k = k + 1u) {
    if (l.x < 0 || l.y < 0 || l.z < 0 || l.x > 31 || l.y > 31 || l.z > 31) { return miss; }
    if (skipFirst) { skipFirst = false; }
    else if (readLocal(slot, l)) {
      var h : Hit; h.hit = true; h.t = t;
      var n = vec3<f32>(0.0);
      if (axis == 0) { n.x = -f32(stepi.x); }
      else if (axis == 1) { n.y = -f32(stepi.y); }
      else { n.z = -f32(stepi.z); }
      h.n = n;
      h.wp = base + vec3<f32>(l);
      h.editColor = readColor(slot, l);
      return h;
    }
    if (tMax.x < tMax.y) {
      if (tMax.x < tMax.z) { l.x += stepi.x; t = tMax.x; tMax.x += tDelta.x; axis = 0; }
      else                 { l.z += stepi.z; t = tMax.z; tMax.z += tDelta.z; axis = 2; }
    } else {
      if (tMax.y < tMax.z) { l.y += stepi.y; t = tMax.y; tMax.y += tDelta.y; axis = 1; }
      else                 { l.z += stepi.z; t = tMax.z; tMax.z += tDelta.z; axis = 2; }
    }
  }
  return miss;
}

fn trace(o : vec3<f32>, dIn : vec3<f32>) -> Hit {
  var miss : Hit; miss.hit = false; miss.editColor = 0u; miss.wp = vec3<f32>(0.0);
  let dir = select(dIn, vec3<f32>(1e-6), dIn == vec3<f32>(0.0));
  let inv = 1.0 / dir;
  let wc = uni.worldChunks;
  let WV = vec3<f32>(f32(wc.x) * CH, f32(wc.y) * CH, f32(wc.z) * CH);

  let t0 = (vec3<f32>(0.0) - o) * inv;
  let t1 = (WV - o) * inv;
  let tlo = min(t0, t1);
  let thi = max(t0, t1);
  var tenter = max(max(tlo.x, tlo.y), max(tlo.z, 0.0));
  let texit  = min(min(thi.x, thi.y), thi.z);
  if (tenter > texit) { return miss; }

  let stepi = vec3<i32>(sign(dir));
  var lastAxis : i32 = -1;
  if (tenter > 0.0) {
    if (tlo.x >= tlo.y && tlo.x >= tlo.z) { lastAxis = 0; }
    else if (tlo.y >= tlo.z) { lastAxis = 1; }
    else { lastAxis = 2; }
  } else {
    tenter = 0.0;
  }
  var t = tenter;
  let p = o + dir * (t + 1e-3);
  var cell = clamp(vec3<i32>(floor(p / CH)),
                   vec3<i32>(0),
                   vec3<i32>(i32(wc.x) - 1, i32(wc.y) - 1, i32(wc.z) - 1));

  let nb = (vec3<f32>(cell) + select(vec3<f32>(0.0), vec3<f32>(1.0), dir > vec3<f32>(0.0))) * CH;
  var tMax = (nb - o) * inv;
  let tDelta = abs(CH * inv);

  for (var it : u32 = 0u; it < 800u; it = it + 1u) {
    if (cell.x < 0 || cell.y < 0 || cell.z < 0 ||
        cell.x >= i32(wc.x) || cell.y >= i32(wc.y) || cell.z >= i32(wc.z)) { break; }

    let state = grid[cellIndex(cell)];
    if (state >= FULLC) {
      if (state == FULLC) {                 // full solid chunk: hit at entry face
        // lastAxis == -1 means we started inside this chunk (camera inside solid).
        // Skip it so we don't render a false "big sphere" around the camera.
        if (lastAxis < 0) {
          // fall through to step to the next chunk
        } else {
          var h : Hit; h.hit = true; h.t = t; h.editColor = 0u;
          var n = vec3<f32>(0.0);
          if (lastAxis == 0) { n.x = -f32(stepi.x); }
          else if (lastAxis == 1) { n.y = -f32(stepi.y); }
          else { n.z = -f32(stepi.z); }
          h.n = n;
          h.wp = o + dir * t;
          return h;
        }
      }
      // EMPTY, UNKNOWN, or skipped FULL: fall through and step to the next chunk.
    } else {
      let inn = innerDDA(o, dir, inv, stepi, cell, state, t, lastAxis);
      if (inn.hit) { return inn; }
    }

    if (tMax.x < tMax.y) {
      if (tMax.x < tMax.z) { cell.x += stepi.x; t = tMax.x; tMax.x += tDelta.x; lastAxis = 0; }
      else                 { cell.z += stepi.z; t = tMax.z; tMax.z += tDelta.z; lastAxis = 2; }
    } else {
      if (tMax.y < tMax.z) { cell.y += stepi.y; t = tMax.y; tMax.y += tDelta.y; lastAxis = 1; }
      else                 { cell.z += stepi.z; t = tMax.z; tMax.z += tDelta.z; lastAxis = 2; }
    }
  }
  return miss;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let dims = textureDimensions(outImage);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let pixel = vec2<i32>(gid.xy);

  let o = uni.rayMat[3].xyz;
  let d = (uni.rayMat * vec4<f32>(vec2<f32>(pixel) + vec2<f32>(0.5), 1.0, 0.0)).xyz;
  let dn = normalize(d);

  let h = trace(o, d);
  var col : vec3<f32>;
  let isEditColor = h.hit && ((h.editColor >> 24u) == 0xFFu);
  // A ray only stops at a solid voxel it reached *through air* (the DDA arrives
  // from an empty cell), so any hit is, by construction, a face adjacent to air —
  // exactly the "neighbor is air" face-culling rule the mesh renderer in src/ uses.
  // We therefore draw every hit. Looking down from above, the ray stops at the top
  // surface and never sees the dirt below it; flying underground, the dug-out/cave
  // walls around you are exactly the air-adjacent faces, so they stay visible.
  let isSurface   = h.hit;

  var gn = vec3<f32>(0.0);   // surface normal for the G-buffer (0 = sky)
  var gd = 1.0;              // normalized depth for the G-buffer (1 = sky/far)
  if (isSurface) {
    var base : vec3<f32>;
    if (isEditColor) {
      let r = f32((h.editColor >> 16u) & 255u) / 255.0;
      let g = f32((h.editColor >>  8u) & 255u) / 255.0;
      let b = f32( h.editColor         & 255u) / 255.0;
      base = vec3<f32>(r, g, b);
    } else {
      base = terrainColor(h.wp.y, h.n);
    }
    let lambert = max(dot(h.n, uni.sunDir), 0.0) * 0.75 + 0.28;
    col = base * lambert;
    let fog = clamp(1.0 - exp(-h.t * uni.fogDensity), 0.0, 1.0);
    col = mix(col, skyColor(dn), fog);
    gn = h.n;
    // Normalize hit distance against world width so edge thresholds are scale-stable.
    let WX = f32(uni.worldChunks.x) * CH;
    gd = clamp(h.t / (WX * 1.5), 0.0, 0.999);
  } else {
    // No hit, or hit is underground/buried — show sky (void/emptiness)
    col = skyColor(dn);
  }
  // mild tonemap / gamma
  col = pow(clamp(col, vec3<f32>(0.0), vec3<f32>(1.0)), vec3<f32>(0.85));
  textureStore(outImage, pixel, vec4<f32>(col, 1.0));
  textureStore(normDepth, pixel, vec4<f32>(gn * 0.5 + 0.5, gd));
}
`;

export const BLIT_SRC = /* wgsl */`
@group(0) @binding(0) var srcTex : texture_2d<f32>;
struct VSOut { @builtin(position) pos : vec4<f32> };
@vertex fn vs(@builtin(vertex_index) vi : u32) -> VSOut {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
  var o : VSOut; o.pos = vec4<f32>(p[vi], 0.0, 1.0); return o;
}
@fragment fn fs(@builtin(position) fp : vec4<f32>) -> @location(0) vec4<f32> {
  return textureLoad(srcTex, vec2<i32>(i32(fp.x), i32(fp.y)), 0);
}
`;

// Screen-space edge outline pass, ported from old_sparse/SH_EDGE. Darkens depth
// discontinuities (silhouettes/steps) and lightens normal creases, which makes
// each voxel step read crisply instead of merging into smooth slabs.
export const EDGE_SRC = /* wgsl */`
@group(0) @binding(0) var colorTex     : texture_2d<f32>;
@group(0) @binding(1) var normDepthTex : texture_2d<f32>;
@group(0) @binding(2) var outTex       : texture_storage_2d<rgba8unorm, write>;

fn getDepth(pos:vec2<i32>, sz:vec2<i32>) -> f32 {
  return textureLoad(normDepthTex, clamp(pos, vec2<i32>(0), sz-1), 0).a;
}
fn getNormal(pos:vec2<i32>, sz:vec2<i32>) -> vec3<f32> {
  return textureLoad(normDepthTex, clamp(pos, vec2<i32>(0), sz-1), 0).rgb * 2.0 - 1.0;
}

fn depthEdgeIndicator(pos:vec2<i32>, sz:vec2<i32>) -> f32 {
  let d = getDepth(pos, sz);
  var diff = 0.0;
  diff += clamp(getDepth(pos+vec2<i32>( 1, 0), sz) - d, 0.0, 1.0);
  diff += clamp(getDepth(pos+vec2<i32>(-1, 0), sz) - d, 0.0, 1.0);
  diff += clamp(getDepth(pos+vec2<i32>( 0, 1), sz) - d, 0.0, 1.0);
  diff += clamp(getDepth(pos+vec2<i32>( 0,-1), sz) - d, 0.0, 1.0);
  return floor(smoothstep(0.002, 0.008, diff) * 2.0) / 2.0;
}

fn neighborNormalEdge(pos:vec2<i32>, off:vec2<i32>, sz:vec2<i32>, depth:f32, normal:vec3<f32>) -> f32 {
  let npos = pos + off;
  let depthDiff = getDepth(npos, sz) - depth;
  let nNormal = getNormal(npos, sz);
  let normalDiff = dot(normal - nNormal, vec3<f32>(1.0));
  let normalIndicator = clamp(smoothstep(-0.01, 0.01, normalDiff), 0.0, 1.0);
  let depthIndicator  = clamp(sign(depthDiff * 0.25 + 0.0025), 0.0, 1.0);
  return distance(normal, nNormal) * depthIndicator * normalIndicator;
}

fn normalEdgeIndicator(pos:vec2<i32>, sz:vec2<i32>) -> f32 {
  let d = getDepth(pos, sz);
  let n = getNormal(pos, sz);
  var ind = 0.0;
  ind += neighborNormalEdge(pos, vec2<i32>( 0,-1), sz, d, n);
  ind += neighborNormalEdge(pos, vec2<i32>( 0, 1), sz, d, n);
  ind += neighborNormalEdge(pos, vec2<i32>(-1, 0), sz, d, n);
  ind += neighborNormalEdge(pos, vec2<i32>( 1, 0), sz, d, n);
  return step(0.1, ind);
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let sz  = vec2<i32>(textureDimensions(colorTex));
  let pos = vec2<i32>(gid.xy);
  if (pos.x >= sz.x || pos.y >= sz.y) { return; }

  let dei = depthEdgeIndicator(pos, sz);
  let nei = normalEdgeIndicator(pos, sz);

  let coefficient = select(1.0 + 0.3 * nei, 1.0 - 0.7 * dei, dei > 0.0);
  let src = textureLoad(colorTex, pos, 0);
  textureStore(outTex, pos, vec4<f32>(clamp(src.rgb * coefficient, vec3<f32>(0.0), vec3<f32>(1.0)), 1.0));
}
`;

export const WORKER_SRC = `
const CH = 32;
function ih(x, z, seed) {
  let h = Math.imul(x|0,374761393) ^ Math.imul(z|0,668265263) ^ Math.imul(seed>>>0,2246822519);
  h = Math.imul(h ^ (h >>> 15), 2246822519); h ^= h >>> 13; h = Math.imul(h, 3266489917); h ^= h >>> 16;
  return (h >>> 0) / 4294967296;
}
function smooth(t) { return t * t * (3 - 2 * t); }
function vnoise(x, z, seed) {
  const xi = Math.floor(x), zi = Math.floor(z), xf = x - xi, zf = z - zi;
  const a = ih(xi, zi, seed), b = ih(xi + 1, zi, seed), c = ih(xi, zi + 1, seed), d = ih(xi + 1, zi + 1, seed);
  const u = smooth(xf), v = smooth(zf);
  return a * (1 - u) * (1 - v) + b * u * (1 - v) + c * (1 - u) * v + d * u * v;
}
function fbm(x, z, seed, oct = 5) { let s = 0, amp = 1, f = 1, norm = 0; for (let o = 0; o < oct; o++) { s += amp * vnoise(x * f, z * f, seed + o * 1311); norm += amp; amp *= 0.5; f *= 2; } return s / norm; }
function ridged(x, z, seed, oct = 5) { let s = 0, amp = 1, f = 1, norm = 0; for (let o = 0; o < oct; o++) { let n = vnoise(x * f, z * f, seed + o * 1777); n = 1 - Math.abs(2 * n - 1); n *= n; s += amp * n; norm += amp; amp *= 0.5; f *= 2; } return s / norm; }
function height(wx, wz, seed) {
  // Terrain scales off the 128-voxel world height (WY*CH = 4*32), matching old_sparse
  // (VY=128): plainsBase≈40, mountainTop≈107 — fits under the 128 ceiling.
  const VY = 128;
  const plainsBase = Math.round(VY * 0.31);       // old_sparse: flat buildable plains (~40)
  const mountainTop = 380;                          // towering Minecraft-scale peaks (~340 above plains);
                                                    // just under the 416 (WY*CH=13*32) ceiling
  const tcx = 8192 * 0.5, tcz = 8192 * 0.5, clearR = 120;
  const raw = fbm(wx * 0.0022 + 3.7, wz * 0.0022 + 1.9, seed, 3);
  const keepFade = Math.min(1, Math.max(0, (Math.hypot(wx - tcx, wz - tcz) - clearR) / 96));
  const b = raw * keepFade;
  const blend = Math.min(1, Math.max(0, (b - 0.62) / 0.10));
  const blendS = blend * blend * (3 - 2 * blend);
  if (blendS < 0.001) return plainsBase;
  // Cliffy plateau (mesa/table-mountain) shaping. FBM (rounded blobs) instead of
  // ridged (sharp needles) gives broad massifs; a low exponent keeps mid-elevations
  // wide rather than pinching them into peaks. The continuous height is then TERRACED
  // (quantized to discrete levels) so smooth slopes become flat shelves separated by
  // vertical cliff faces — large flat tops with cliffy step-ups between them.
  const m1 = fbm(wx * 0.006 + 1.3, wz * 0.006 + 2.7, seed, 5);
  const m2 = fbm(wx * 0.015 + 5.1, wz * 0.015 + 3.9, seed, 4);
  const shape = m1 * 0.72 + m2 * 0.28;
  const elev = Math.pow(Math.max(0, shape - 0.12) / 0.88, 0.85);  // broad plateaus
  let mtnH = plainsBase + (mountainTop - plainsBase) * elev;
  // Terrace into flat shelves. STEP = shelf height ≈ cliff height between levels.
  const STEP = 26;
  const lvl = Math.floor(mtnH / STEP);
  const frac = mtnH / STEP - lvl;                       // 0..1 within the current shelf
  // Sharpen the transition: snap most of each shelf flat, then a quick cliff face near
  // the top of the band. pow(frac, 6) stays ~0 across the flat top and shoots up only
  // in the last sliver, so risers read as near-vertical cliffs, treads as flat tops.
  const cliff = Math.pow(frac, 6);
  mtnH = (lvl + cliff) * STEP;
  // A little sub-voxel grit so cliff edges and tops aren't machine-perfect.
  mtnH += fbm(wx * 0.06 + 7.1, wz * 0.06 + 4.3, seed, 3) * 3 - 1.5;
  return plainsBase * (1 - blendS) + mtnH * blendS;
}
self.onmessage = (e) => {
  const m = e.data;
  const { cell, cx, cy, cz, seed, edits, apronEdits } = m;
  const t0 = performance.now();
  const wx0 = cx * CH, wy0 = cy * CH, wz0 = cz * CH;
  // editMap: localIndex → [solid, r, g, b]  (edits inside this chunk)
  const editMap = edits && edits.length
    ? new Map(edits.map(e => [e[0], {s:e[1], r:e[2]??0, g:e[3]??0, b:e[4]??0}]))
    : null;
  // apronMap: "wx,wy,wz" → solid  (edits one voxel outside this chunk, so the
  // air-touching test sees holes dug just across a chunk border).
  const apronMap = apronEdits && apronEdits.length
    ? new Map(apronEdits.map(e => [e[0] + ',' + e[1] + ',' + e[2], e[3]]))
    : null;
  // Any edit (interior or apron) means terrain exposure may have changed, so the
  // full 6-neighbor air-touching test must run for this chunk.
  const hasEdits = !!editMap || !!apronMap;
  // Sample the heightmap one voxel wider on each side (a 34x34 padded grid) so we
  // know the terrain top of every horizontal neighbor, including those that fall in
  // the adjacent chunk. With neighbor tops we can tell whether a solid voxel is
  // exposed to air (a surface/cliff voxel) or fully buried (all 6 neighbors solid).
  const PAD = CH + 2;
  const hmP = new Float32Array(PAD * PAD);
  let minH = 1e9, maxH = -1e9;     // over interior columns (for the all-air test)
  let minHpad = 1e9;               // over interior + 1-voxel apron (for the buried test)
  for (let pz = 0; pz < PAD; pz++) {
    for (let px = 0; px < PAD; px++) {
      const h = height(wx0 + px - 1, wz0 + pz - 1, seed);
      hmP[pz * PAD + px] = h;
      if (h < minHpad) minHpad = h;
      // Track interior min/max over the actual 32x32 chunk columns only.
      if (px >= 1 && px <= CH && pz >= 1 && pz <= CH) {
        if (h < minH) minH = h;
        if (h > maxH) maxH = h;
      }
    }
  }
  const hmCol = (lx, lz) => hmP[(lz + 1) * PAD + (lx + 1)];  // interior accessor, lx/lz in 0..31
  // top(x,z) and the min of the 4 horizontal neighbor tops, per column.
  const colTop = new Int32Array(CH * CH);
  const colNbrMin = new Int32Array(CH * CH);  // min floor() of N/E/S/W neighbor tops
  for (let lz = 0; lz < CH; lz++) {
    for (let lx = 0; lx < CH; lx++) {
      const t  = Math.floor(hmCol(lx, lz));
      const tn = Math.min(
        Math.floor(hmCol(lx - 1, lz)), Math.floor(hmCol(lx + 1, lz)),
        Math.floor(hmCol(lx, lz - 1)), Math.floor(hmCol(lx, lz + 1)),
      );
      colTop[lz * CH + lx] = t;
      colNbrMin[lz * CH + lx] = tn;
    }
  }
  const topVox = wy0 + CH - 1;
  let kind, count = 0, buf = null, cbuf = null;
  if (!hasEdits && wy0 > maxH) {
    // Entire chunk is above all terrain → empty (unchanged).
    kind = 0;
  } else if (!hasEdits && topVox < minHpad) {
    // Even this chunk's highest voxel sits below the lowest terrain surface of any
    // adjacent column. Every voxel is buried on all sides. Keep it FULL (solid) — NOT
    // empty: a FULL chunk stops the ray instantly at its entry face (one DDA step),
    // exactly like old_sparse relies on solid chunks to block rays. The face is
    // underground so it's never reached from above (the surface shell blocks first),
    // and marking it empty instead would force rays to march across the whole hollow
    // interior — the cause of the framerate collapse. FULL stores zero voxel data.
    kind = 1; count = CH * CH * CH;
  } else {
    kind = 2;
    const words  = new Uint32Array(CH * CH);  // bitmask: 1024 u32s
    const colors = new Uint32Array(CH * CH);  // word-level color: 1024 u32s, one per (lz,ly)
    let hasColor = false;
    // True world-space solidity, accounting for edits. Used for the 6-neighbor
    // air-touching test so digging a hole re-exposes the surrounding buried voxels.
    //   x,z are world coords; the heightmap is sampled directly (works for any cell,
    //   even outside this chunk). Edits only exist inside this chunk's local range.
    const solidWorld = (x, y, z) => {
      if (y < 0) return false;
      if (editMap) {
        const elx = x - wx0, ely = y - wy0, elz = z - wz0;
        if (elx >= 0 && elx < CH && ely >= 0 && ely < CH && elz >= 0 && elz < CH) {
          const e = editMap.get(elx + CH * (ely + CH * elz));
          if (e) return e.s === 1;
        }
      }
      if (apronMap) {
        const a = apronMap.get(x + ',' + y + ',' + z);
        if (a !== undefined) return a === 1;
      }
      // Reuse the precomputed padded heightmap when the column is in range
      // (it covers wx0-1..wx0+CH, wz0-1..wz0+CH — exactly the 6-neighbor reach),
      // avoiding a fresh 5-octave FBM eval per neighbor. Fall back to height() only
      // for the rare out-of-apron column.
      const px = x - wx0 + 1, pz = z - wz0 + 1;
      const h = (px >= 0 && px < PAD && pz >= 0 && pz < PAD)
        ? hmP[pz * PAD + px]
        : height(x, z, seed);
      return y <= Math.floor(h);
    };
    for (let lz = 0; lz < CH; lz++) {
      for (let lx = 0; lx < CH; lx++) {
        const top    = colTop[lz * CH + lx];
        const nbrMin = colNbrMin[lz * CH + lx];
        for (let ly = 0; ly < CH; ly++) {
          const li = lx + CH * (ly + CH * lz);
          const wy = wy0 + ly, wx = wx0 + lx, wz = wz0 + lz;
          // We render like src/: keep a voxel solid only if it touches air on some side.
          // Three cases below — explicit edit, edited-chunk (authoritative 6-neighbor
          // test), or unedited chunk (fast heightmap-only test).
          const isTerrain  = wy <= top;
          let solid = false;
          let cr = 0, cg = 0, cb = 0, hasEditColor = false;
          const ed = editMap ? editMap.get(li) : undefined;
          if (ed) {
            // Explicit edit wins outright.
            solid = ed.s === 1;
            if (solid && (ed.r || ed.g || ed.b)) {
              cr = ed.r; cg = ed.g; cb = ed.b; hasEditColor = true;
            }
          } else if (hasEdits) {
            // This chunk (or a neighbor) was edited, so the cheap heightmap-only
            // air-touching test can be wrong in either direction — a dug pit exposes a
            // previously-buried voxel, or a raised mound buries a previously-surface one.
            // Decide authoritatively: a terrain voxel is solid iff it's terrain AND at
            // least one of its 6 neighbors is air (per true, edit-aware solidity).
            if (isTerrain) {
              solid =
                !solidWorld(wx - 1, wy, wz) || !solidWorld(wx + 1, wy, wz) ||
                !solidWorld(wx, wy - 1, wz) || !solidWorld(wx, wy + 1, wz) ||
                !solidWorld(wx, wy, wz - 1) || !solidWorld(wx, wy, wz + 1);
            }
          } else {
            // Unedited chunk: fast heightmap-only air-touching test.
            //   • topmost voxel of the column (air directly above), or
            //   • a horizontal neighbor column's terrain is lower (exposed cliff side).
            solid = isTerrain && (wy === top || wy > nbrMin);
          }
          if (solid) {
            words[lz * CH + ly] |= (1 << lx);
            count++;
            if (hasEditColor) {
              // Store in word-level slot: (lz, ly) — all X voxels in this word share the color
              colors[lz * CH + ly] = (0xFF << 24) | (cr << 16) | (cg << 8) | cb;
              hasColor = true;
            }
          }
        }
      }
    }
    if (count === 0) kind = 0;
    else if (count === CH * CH * CH) kind = 1;
    else {
      buf = words.buffer;
      if (hasColor) cbuf = colors.buffer;
    }
  }
  const genMs = performance.now() - t0;
  const msg = { cell, cx, cy, cz, kind, count, genMs };
  const transfers = [];
  if (buf)  { msg.buf  = buf;  transfers.push(buf); }
  if (cbuf) { msg.cbuf = cbuf; transfers.push(cbuf); }
  self.postMessage(msg, transfers);
};
`;
