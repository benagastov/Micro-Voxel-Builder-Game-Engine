"use strict";

/* ====================  W G S L   S H A D E R S  ==========================  */

export const SH_SHARED = /* wgsl */`
const EPS = 0.0001;
const INV255 = 0.00392156862;
const INV65535 = 0.0000152590219;
const CS = 8.0;        // chunk size (voxels)
const INVCS = 0.125;   // 1/8
const HINVCS = 0.0625; // 1/16
const CHUNK_U32 = 24u;

struct Uniforms {
  camPos:vec4<f32>, camFwd:vec4<f32>, camRight:vec4<f32>, camUp:vec4<f32>,
  sunDir:vec4<f32>, sunStrength:vec4<f32>, ambient:vec4<f32>,
  skyBot:vec4<f32>, skyTop:vec4<f32>,
  mapSize:vec4<u32>,
  p0:vec4<u32>,   // viewMode, frameNum, split, reqCap
  p1:vec4<u32>,   // numDiffuse, maxDiffuse, diffuseBounce, specBounce
  p2:vec4<f32>,   // time, shadowSoft, resX, resY
};

@group(0) @binding(0) var<storage, read_write> mapFlags : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read>       mapVoxIdx: array<u32>;
@group(0) @binding(2) var<storage, read>       chunks   : array<u32>;
@group(0) @binding(3) var<storage, read_write> voxels   : array<u32>;
@group(0) @binding(4) var<storage, read>       materials: array<u32>;
@group(0) @binding(5) var<uniform>             U        : Uniforms;

struct Voxel { normal:vec3<f32>, material:u32, albedo:vec3<f32>, spec:vec3<f32>, diffuse:vec3<f32> };
struct Mat   { emissive:u32, opacity:f32, refractIndex:f32, specular:f32, reflectType:u32, shininess:u32 };
struct Hit   { hit:bool, rayPos:vec3<f32>, normal:vec3<f32>, voxel:Voxel, colorAdd:vec3<f32>, colorMult:f32 };

fn decode4(v:u32) -> vec4<u32> {
  return vec4<u32>((v>>24u)&255u,(v>>16u)&255u,(v>>8u)&255u,v&255u);
}
fn encode4(v:vec4<u32>) -> u32 {
  let c = v & vec4<u32>(255u);
  return (c.x<<24u)|(c.y<<16u)|(c.z<<8u)|c.w;
}

fn getMat(i:u32) -> Mat {
  let b = i*8u;
  var m:Mat;
  m.emissive     = materials[b+0u];
  m.opacity      = bitcast<f32>(materials[b+1u]);
  m.refractIndex = bitcast<f32>(materials[b+2u]);
  m.specular     = bitcast<f32>(materials[b+3u]);
  m.reflectType  = materials[b+4u];
  m.shininess    = materials[b+5u];
  return m;
}

fn skyColor(dir:vec3<f32>) -> vec3<f32> {
  return mix(U.skyBot.xyz, U.skyTop.xyz, dir.y*0.5+1.0);
}

// WGSL has no bool->numeric vector conversion; use select()
fn fmask(m:vec3<bool>) -> vec3<f32> { return select(vec3<f32>(0.0), vec3<f32>(1.0), m); }
fn imask(m:vec3<bool>) -> vec3<i32> { return select(vec3<i32>(0),   vec3<i32>(1),   m); }

fn inMap(p:vec3<i32>) -> bool {
  return p.x>=0 && p.y>=0 && p.z>=0 &&
         p.x<i32(U.mapSize.x) && p.y<i32(U.mapSize.y) && p.z<i32(U.mapSize.z);
}
fn inChunk(p:vec3<i32>) -> bool {
  return p.x>=0 && p.y>=0 && p.z>=0 && p.x<8 && p.y<8 && p.z<8;
}
fn mapIndex(p:vec3<i32>) -> u32 {
  let x = u32(p.x);
  let y = u32(p.y);
  let z = u32(p.z);
  return x + U.mapSize.x*(y + U.mapSize.y*z);
}

fn voxExists(cell:u32, cp:vec3<i32>) -> bool {
  let idx = u32(cp.x) + 8u*(u32(cp.y)+8u*u32(cp.z));
  return ((chunks[cell*CHUNK_U32 + 7u + (idx>>5u)] >> (idx & 31u)) & 1u) != 0u;
}

// popcount lookup of a voxel's storage index inside the sparse chunk
fn voxStorageIndex(cell:u32, cp:vec3<i32>) -> u32 {
  let li  = u32(cp.x) + 8u*(u32(cp.y)+8u*u32(cp.z));
  let bmi = li >> 5u;
  let base = cell*CHUNK_U32;
  var voxNum:u32 = 0u;
  if (bmi > 3u) { voxNum = chunks[base + 4u + ((bmi>>2u)-1u)]; }
  var i = bmi & ~3u;
  loop {
    if (i > bmi) { break; }
    var bits = chunks[base + 7u + i];
    if (i == bmi) { bits = bits & ((1u << (li & 31u)) - 1u); }
    voxNum = voxNum + countOneBits(bits);
    i = i + 1u;
  }
  return mapVoxIdx[cell] + voxNum;
}

// inverse: which (x,y,z) is the n-th stored voxel of a chunk
fn voxPosition(cell:u32, voxNumIn:u32) -> vec3<i32> {
  let base = cell*CHUNK_U32;
  var count:u32 = 0u; var pos:u32 = 0u; var startI:u32 = 0u;
  for (var i=0u; i<3u; i=i+1u) {
    if (chunks[base+4u+i] < voxNumIn) { count = chunks[base+4u+i]; pos = pos+128u; startI = startI+4u; }
    else { break; }
  }
  let voxNum = voxNumIn + 1u;
  var i = startI;
  loop {
    if (i >= 16u) { break; }
    if (count == voxNum) { break; }
    let bc = countOneBits(chunks[base+7u+i]);
    if (count + bc >= voxNum) {
      var bitNum = 0u;
      loop {
        if (count >= voxNum) { break; }
        count = count + ((chunks[base+7u+i] >> bitNum) & 1u);
        bitNum = bitNum + 1u; pos = pos + 1u;
        if (bitNum > 32u) { break; }
      }
    } else { count = count + bc; pos = pos + 32u; }
    i = i + 1u;
  }
  if (pos > 0u) { pos = pos - 1u; }
  if (count < voxNum) { return vec3<i32>(-1,-1,-1); }
  return vec3<i32>(i32(pos%8u), i32((pos/8u)%8u), i32(pos/64u));
}

fn decompress(vIndex:u32) -> Voxel {
  let b = vIndex*4u;
  let nR = decode4(voxels[b+0u]);
  let aR = decode4(voxels[b+1u]);
  let sR = decode4(voxels[b+2u]);
  let dR = decode4(voxels[b+3u]);
  let dUp = vec3<u32>(sR.z, dR.x, dR.z) << vec3<u32>(8u);
  let dLo = vec3<u32>(sR.w, dR.y, dR.w);
  let dif = vec3<f32>(dUp | dLo);
  var v:Voxel;
  v.normal   = (vec3<f32>(nR.yzw)*INV255 - 0.5)*2.0;
  v.material = nR.x;
  v.albedo   = vec3<f32>(aR.xyz)*INV255;
  v.diffuse  = dif*INV65535;
  v.spec     = vec3<f32>(f32(aR.w), f32(sR.x), f32(sR.y))*INV255;
  return v;
}

// ---- two-level DDA: chunk grid (skip empties) then voxels inside hit chunk --
fn stepMap(rayDirIn:vec3<f32>, rayPosIn:vec3<f32>, ignoreFirstIn:bool,
           maxDepth:f32, normalIn:vec3<f32>, orgPos:vec3<f32>, propagateVis:bool) -> Hit {
  var H:Hit;
  H.hit = false; H.colorAdd = vec3<f32>(0.0); H.colorMult = 1.0; H.normal = normalIn;
  H.rayPos = rayPosIn;
  let rayDir = rayDirIn;
  let invDir = 1.0 / rayDir;
  var rayPos = rayPosIn;
  var ignoreFirst = ignoreFirstIn;

  // outer DDA setup
  var pos = vec3<i32>(floor(rayPos));
  let dDist = abs(invDir);
  let rStep = vec3<i32>(sign(rayDir));
  var sDist = (sign(rayDir)*(vec3<f32>(pos)-rayPos) + sign(rayDir)*0.5 + 0.5) * dDist;
  var lastS = vec3<f32>(0.0);

  var guard = 0;
  loop {
    if (!inMap(pos) || guard > 4096) { break; }
    guard = guard + 1;
    let cell = mapIndex(pos);
    let flags = atomicLoad(&mapFlags[cell]);
    if ((flags & 3u) == 2u) {
      let m = min(min(lastS.x,lastS.y),lastS.z);
      let upd = rayPos + rayDir*(m - EPS);
      var cRay = (upd - vec3<f32>(pos))*CS;
      cRay = clamp(cRay, vec3<f32>(EPS), vec3<f32>(CS-EPS));

      // ---- inner voxel DDA ----
      var ip = vec3<i32>(floor(cRay));
      let idDist = dDist;            // same direction
      var isDist = (sign(rayDir)*(vec3<f32>(ip)-cRay) + sign(rayDir)*0.5 + 0.5) * idDist;
      var ilast  = vec3<f32>(0.0);
      var lastVoxID:u32 = 0xffffffffu;
      var iguard = 0;
      loop {
        if (!inChunk(ip) || iguard > 64) { break; }
        iguard = iguard + 1;
        if (voxExists(cell, ip) && !ignoreFirst) {
          let vIndex = voxStorageIndex(cell, ip);
          let vx = decompress(vIndex);
          let mat = getMat(vx.material);
          if (mat.opacity >= 1.0) {
            let mm = min(min(ilast.x,ilast.y),ilast.z);
            cRay = cRay + rayDir*(mm + EPS);
            H.hit = true; H.voxel = vx;
            H.rayPos = vec3<f32>(pos) + cRay*INVCS;
            if (propagateVis) { atomicOr(&mapFlags[cell], 4u); }
            return H;
          } else {
            let rawA = voxels[vIndex*4u+1u];
            let rawN = voxels[vIndex*4u+0u];
            let thisID = (rawA & 0xffffff00u) | (rawN >> 24u);
            if (lastVoxID != thisID) {
              let mm = min(min(ilast.x,ilast.y),ilast.z);
              var cur = cRay + rayDir*mm;
              cur = vec3<f32>(pos) + cur*INVCS;
              let toCur = cur - orgPos;
              if (maxDepth < 0.0 || dot(toCur,toCur) < maxDepth*maxDepth) {
                H.colorAdd = H.colorAdd + H.colorMult*mat.opacity*vx.albedo*U.sunStrength.xyz;
                H.colorMult = H.colorMult*(1.0 - mat.opacity);
              }
              lastVoxID = thisID;
            }
          }
        }
        // iterate inner
        let mask = isDist.xyz <= min(isDist.yzx, isDist.zxy);
        ilast = isDist;
        isDist = isDist + fmask(mask)*idDist;
        ip = ip + imask(mask)*rStep;
        H.normal = fmask(mask)*(-vec3<f32>(rStep));
        ignoreFirst = false;
      }
    }
    // iterate outer
    let mask = sDist.xyz <= min(sDist.yzx, sDist.zxy);
    lastS = sDist;
    sDist = sDist + fmask(mask)*dDist;
    pos = pos + imask(mask)*rStep;
    H.normal = fmask(mask)*(-vec3<f32>(rStep));
    ignoreFirst = false;
  }
  return H;
}
`;

export const SH_DRAW = /* wgsl */`
const GAMMA = 2.2;
const INV_GAMMA = 0.4545;
@group(0) @binding(6) var outTex : texture_storage_2d<rgba8unorm, write>;

// ---- procedural sky (only for primary background rays) -------------------
fn hash21(p:vec2<f32>) -> f32 {
  let h = dot(p, vec2<f32>(127.1, 311.7));
  return fract(sin(h)*43758.5453);
}
fn vn2(p:vec2<f32>) -> f32 {
  let i = floor(p); let f = fract(p);
  let u = f*f*(3.0-2.0*f);
  let a = hash21(i);
  let b = hash21(i+vec2<f32>(1.0,0.0));
  let c = hash21(i+vec2<f32>(0.0,1.0));
  let d = hash21(i+vec2<f32>(1.0,1.0));
  return mix(mix(a,b,u.x), mix(c,d,u.x), u.y);
}
fn fbm2(p:vec2<f32>) -> f32 {
  var s=0.0; var amp=0.5; var q=p;
  for (var i=0; i<5; i=i+1) { s = s + amp*vn2(q); q = q*2.0; amp = amp*0.5; }
  return s;
}
fn skySky(dirIn:vec3<f32>) -> vec3<f32> {
  let dir = normalize(dirIn);
  let up  = clamp(dir.y, -1.0, 1.0);
  // vertical gradient: pale warm horizon → deep blue zenith
  let zenith  = vec3<f32>(0.18, 0.36, 0.66);
  let horizon = vec3<f32>(0.80, 0.87, 0.94);
  var col = mix(horizon, zenith, pow(clamp(up,0.0,1.0), 0.55));
  // soft haze just below the horizon line
  let ground = vec3<f32>(0.55, 0.58, 0.58);
  col = mix(col, ground, smoothstep(0.0, -0.16, up));
  // sun disc + warm halo
  let sd   = max(dot(dir, normalize(U.sunDir.xyz)), 0.0);
  let glow = pow(sd, 6.0)*0.35 + pow(sd, 110.0)*0.55;
  let disc = pow(sd, 2200.0);
  col = col + vec3<f32>(1.00, 0.82, 0.55)*glow;
  col = col + vec3<f32>(1.00, 0.96, 0.88)*disc*3.0;
  // drifting cumulus, projected on a sky dome, faded near horizon
  if (up > 0.015) {
    let pl = dir.xz / (dir.y + 0.16);
    let t  = U.p2.x * 0.004;
    var c  = fbm2(pl*1.3 + vec2<f32>(t, t*0.6));
    c = smoothstep(0.50, 0.92, c);
    let edge = smoothstep(0.015, 0.20, up);
    let lit  = clamp(dot(normalize(vec3<f32>(pl.x,1.0,pl.y)), normalize(U.sunDir.xyz)), 0.0, 1.0);
    let cloudCol = mix(vec3<f32>(0.80,0.83,0.88), vec3<f32>(1.0,0.99,0.96), lit);
    col = mix(col, cloudCol, c*edge*0.85);
  }
  return col;
}

fn intersectAABB(invDir:vec3<f32>, ro:vec3<f32>, bmin:vec3<f32>, bmax:vec3<f32>) -> vec2<f32> {
  let t1 = (bmin-ro)*invDir; let t2 = (bmax-ro)*invDir;
  let a = min(t1,t2); let b = max(t1,t2);
  return vec2<f32>(max(max(a.x,a.y),a.z), min(min(b.x,b.y),b.z));
}
fn normalAABB(p:vec3<f32>, bmin:vec3<f32>, bmax:vec3<f32>) -> vec3<f32> {
  let c=(bmin+bmax)*0.5; let d=(bmax-bmin)*0.5;
  return normalize(trunc((p-c)/d*(1.0+EPS)));
}
fn voxelColor(v:Voxel, colorAdd:vec3<f32>, colorMult:f32, hitN:vec3<f32>) -> vec3<f32> {
  let m = getMat(v.material);
  var spec = v.spec * m.specular;
  var dif  = v.diffuse * (1.0 - m.specular);
  let vm = U.p0.x;
  if (vm == 0u) {
    var solid:vec3<f32>;
    if (m.emissive != 0u) { solid = v.albedo; }
    else { solid = dif*v.albedo + spec; }
    return solid*colorMult + colorAdd;
  } else if (vm == 1u) { return v.albedo; }
  else if (vm == 2u)  { if (m.emissive!=0u){return v.albedo;} return dif; }
  else if (vm == 3u)  { if (m.emissive!=0u){return v.albedo;} return spec; }
  else if (vm == 4u)  { return abs(v.normal); }
  else                { return abs(hitN); }
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let res = vec2<u32>(u32(U.p2.z), u32(U.p2.w));
  if (gid.x >= res.x || gid.y >= res.y) { return; }
  let coords = vec2<i32>(i32(gid.x), i32(gid.y));

  // ray from camera basis
  let ndc = vec2<f32>(
     (f32(gid.x)+0.5)/f32(res.x)*2.0 - 1.0,
     1.0 - (f32(gid.y)+0.5)/f32(res.y)*2.0 );
  let ro = U.camPos.xyz;
  let rd = normalize(U.camFwd.xyz + ndc.x*U.camRight.xyz + ndc.y*U.camUp.xyz) + vec3<f32>(EPS);
  let invRd = 1.0/rd;

  var color:vec3<f32>;
  let ms = vec3<f32>(f32(U.mapSize.x),f32(U.mapSize.y),f32(U.mapSize.z));
  let it = intersectAABB(invRd, ro, vec3<f32>(0.0), ms);

  if (it.x > it.y || it.y < 0.0) {
    color = pow(skySky(rd), vec3<f32>(GAMMA));
  } else {
    var rp = ro;
    if (it.x > 0.0) { rp = rp + rd*(it.x+EPS); }
    var n = normalAABB(rp, vec3<f32>(0.0), ms);
    let H = stepMap(rd, rp, false, -1.0, n, ro, true);
    if (H.hit) {
      color = voxelColor(H.voxel, H.colorAdd, H.colorMult, H.normal);
    } else {
      color = pow(skySky(rd), vec3<f32>(GAMMA))*H.colorMult + H.colorAdd;
    }
  }
  color = pow(color, vec3<f32>(INV_GAMMA));
  textureStore(outTex, coords, vec4<f32>(color, 1.0));
}
`;

export const SH_LIGHT = /* wgsl */`
@group(0) @binding(6) var<storage, read>        requests   : array<u32>;
@group(0) @binding(7) var<storage, read_write>  sampleCount: array<atomic<u32>>;

var<private> SP : array<vec3<f32>,15> = array<vec3<f32>,15>(
  vec3<f32>(0.0,1.0,0.0), vec3<f32>(-0.379803,0.857143,0.347931), vec3<f32>(0.061185,0.714286,-0.697174),
  vec3<f32>(0.499316,0.571429,0.651270), vec3<f32>(-0.889696,0.428571,-0.157375), vec3<f32>(0.808584,0.285714,-0.514354),
  vec3<f32>(-0.256942,0.142857,0.955810), vec3<f32>(-0.460906,0.0,-0.887449), vec3<f32>(0.929687,-0.142857,0.339521),
  vec3<f32>(-0.885815,-0.285714,0.365650), vec3<f32>(0.382949,-0.428571,-0.818338), vec3<f32>(0.245607,-0.571429,0.783037),
  vec3<f32>(-0.605521,-0.714286,-0.350913), vec3<f32>(0.503065,-0.857143,-0.110596), vec3<f32>(0.0,-1.0,0.0));

fn rnd(s:f32) -> f32 { return fract(sin(s)*43758.5453)*2.0 - 1.0; }
fn rnd3(s:f32) -> vec3<f32> { return vec3<f32>(rnd(s),rnd(s*2.0),rnd(s*3.0)); }
fn rndSphere(seed:f32) -> vec3<f32> {
  var s = seed;
  for (var k=0; k<12; k=k+1) { let p = rnd3(s); s = s + 1.0; if (dot(p,p) < 1.0) { return p; } }
  return normalize(rnd3(s));
}

var<private> firstSample:bool;

fn shadowRay(rayPos:vec3<f32>, seed:f32, color:ptr<function,vec3<f32>>) {
  var sdir:vec3<f32>;
  if (firstSample) { sdir = U.sunDir.xyz + vec3<f32>(EPS); }
  else { sdir = normalize(U.sunDir.xyz*U.p2.y + rndSphere(seed)) + vec3<f32>(EPS); }
  let H = stepMap(sdir, rayPos, true, -1.0, vec3<f32>(0.0), rayPos, false);
  if (!H.hit) { *color = *color + U.sunStrength.xyz*H.colorMult + H.colorAdd; }
}

fn specularRay(mapPos:vec3<i32>, rayPosIn:vec3<f32>, rayDirIn:vec3<f32>,
               albedo:vec3<f32>, color:ptr<function,vec3<f32>>) {
  var rayPos = rayPosIn; var rayDir = rayDirIn;
  var lastPos = rayPos; var mult = albedo;
  let limit = i32(U.p1.w);
  for (var i=0; i<limit; i=i+1) {
    let H = stepMap(rayDir, rayPos, true, -1.0, vec3<f32>(0.0), rayPos, true);
    rayPos = H.rayPos;
    if (H.hit) {
      let d = abs(floor(rayPos*CS) - floor(lastPos*CS));
      if (dot(d,d) <= 1.0) { return; }
      let hm = getMat(H.voxel.material);
      let hdif = H.voxel.diffuse*(1.0 - hm.specular);
      if (hm.emissive != 0u) {
        *color = *color + (H.voxel.albedo*H.colorMult + H.colorAdd)*mult*albedo; return;
      } else {
        let hc = hdif*H.voxel.albedo;
        *color = *color + (hc*H.colorMult + H.colorAdd)*mult;
        if (hm.specular == 0.0) { return; }
        mult = mult*H.voxel.albedo*H.colorMult*hm.specular;
        lastPos = rayPos;
        rayDir = reflect(rayDir, H.voxel.normal);
      }
    } else if (dot(rayDir, U.sunDir.xyz) > 0.99) {
      *color = *color + (U.sunStrength.xyz*H.colorMult + H.colorAdd); return;
    } else {
      *color = *color + (skyColor(rayDir)*H.colorMult + H.colorAdd)*mult; return;
    }
  }
}

fn diffuseRay(normal:vec3<f32>, rayPosIn:vec3<f32>, initial:Voxel, seed:f32, color:ptr<function,vec3<f32>>) {
  var hitVox = initial; var hitN = normal; var hm:Mat = getMat(initial.material);
  var newColor = vec3<f32>(1.0);
  var rayPos = rayPosIn; var lastPos = rayPos; var lastDir = vec3<f32>(0.0);
  let limit = i32(U.p1.z);
  for (var i=0; i<limit; i=i+1) {
    var dir:vec3<f32>;
    if (i>0 && (rnd(seed + f32(limit) + f32(i))+1.0)*0.5 < hm.specular) {
      dir = normalize(reflect(lastDir, hitN)*f32(hm.shininess) + rndSphere(U.p2.x + f32(i)));
    } else if (firstSample) {
      dir = normalize(hitN) + vec3<f32>(EPS);
    } else {
      dir = normalize(hitN + rndSphere(seed + f32(i))) + vec3<f32>(EPS);
    }
    let H = stepMap(dir, rayPos, true, -1.0, vec3<f32>(0.0), rayPos, false);
    let nrp = H.rayPos;
    hitVox = H.voxel; hitN = hitVox.normal; hm = getMat(hitVox.material);
    if (H.hit) {
      rayPos = nrp;
      let d = abs(floor(lastPos*CS) - floor(rayPos*CS));
      if (dot(d,d) < 1.0) { return; }
      if (hm.emissive != 0u) {
        *color = *color + newColor*(hitVox.albedo*H.colorMult + H.colorAdd); return;
      } else {
        newColor = newColor*(hitVox.albedo*H.colorMult + H.colorAdd);
      }
    } else {
      *color = *color + (newColor*max(dot(dir,U.sunDir.xyz),0.0)*U.sunStrength.xyz*H.colorMult + H.colorAdd);
      return;
    }
    lastDir = dir;
  }
}

@compute @workgroup_size(32,1,1)
fn main(@builtin(workgroup_id) wid:vec3<u32>, @builtin(local_invocation_id) lid:vec3<u32>) {
  let req = requests[wid.x];
  let cell = req >> 4u;
  let group = req & 15u;
  let cb = cell*CHUNK_U32;
  let mapPos = vec3<i32>(bitcast<i32>(chunks[cb+0u]), bitcast<i32>(chunks[cb+1u]), bitcast<i32>(chunks[cb+2u]));
  let voxNum = lid.x + group*32u;
  let cp = voxPosition(cell, voxNum);
  if (!inChunk(cp)) { return; }

  let vIndex = mapVoxIdx[cell] + voxNum;
  let vx = decompress(vIndex);
  let mat = getMat(vx.material);
  let stored = atomicLoad(&sampleCount[cell]);
  let indirectSamples = f32(min(stored, U.p1.y));
  firstSample = (indirectSamples == 0.0);

  var rayPos = INVCS*vec3<f32>(cp) + vec3<f32>(mapPos) + vec3<f32>(HINVCS);
  rayPos = rayPos + (HINVCS - EPS)*vx.normal;

  var specLight = vec3<f32>(0.0);
  var diffuseLight = vec3<f32>(0.0);

  let viewDir = rayPos - U.camPos.xyz;
  if (mat.specular > 0.0 && dot(viewDir, vx.normal) < 0.0 && mat.reflectType <= 1u) {
    let refl = reflect(normalize(viewDir), vx.normal);
    for (var i=0; i<15; i=i+1) {
      let sd = normalize(refl*f32(mat.shininess) + SP[i]) + vec3<f32>(EPS);
      specularRay(mapPos, rayPos, sd, vx.albedo, &specLight);
    }
    specLight = specLight/15.0;
  }

  if (mat.specular < 1.0) {
    let nd = i32(U.p1.x);
    for (var i=0; i<nd; i=i+1) {
      diffuseLight = diffuseLight + U.ambient.xyz;
      diffuseRay(vx.normal+vec3<f32>(EPS), rayPos, vx, U.p2.x*f32(i+1), &diffuseLight);
      shadowRay(rayPos, U.p2.x*f32(i+1+nd), &diffuseLight);
    }
    diffuseLight = (vx.diffuse*indirectSamples + diffuseLight)/(indirectSamples + f32(nd));
  }

  specLight = clamp(specLight, vec3<f32>(0.0), vec3<f32>(1.0));
  diffuseLight = clamp(diffuseLight, vec3<f32>(0.0), vec3<f32>(1.0));

  let wd = vec3<u32>(round(diffuseLight*65535.0));
  let wLo = wd & vec3<u32>(255u);
  let wUp = (wd >> vec3<u32>(8u)) & vec3<u32>(255u);

  let b = vIndex*4u;
  voxels[b+1u] = encode4(vec4<u32>(vec3<u32>(round(vx.albedo*255.0)), u32(round(specLight.x*255.0))));
  voxels[b+2u] = encode4(vec4<u32>(vec2<u32>(round(specLight.yz*255.0)), wUp.x, wLo.x));
  voxels[b+3u] = encode4(vec4<u32>(wUp.y, wLo.y, wUp.z, wLo.z));

  if (lid.x == 0u && group == 0u) { atomicAdd(&sampleCount[cell], 1u); }
}
`;

export const SH_BUILD = /* wgsl */`
struct U2 { mapSize:vec4<u32>, p0:vec4<u32> };
@group(0) @binding(0) var<storage, read_write> mapFlags : array<atomic<u32>>;
@group(0) @binding(1) var<storage, read>       chunks   : array<u32>;
@group(0) @binding(2) var<storage, read_write> requests : array<u32>;
@group(0) @binding(3) var<storage, read_write> counter  : atomic<u32>;
@group(0) @binding(4) var<uniform>             U        : U2;

@compute @workgroup_size(64,1,1)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
  let cell = gid.x;
  let total = U.mapSize.x*U.mapSize.y*U.mapSize.z;
  if (cell >= total) { return; }
  let flags = atomicLoad(&mapFlags[cell]);
  let loaded = (flags & 3u) == 2u;
  let visible = (flags & 4u) != 0u;
  // consume the visible bit every frame
  atomicAnd(&mapFlags[cell], ~4u);
  if (!loaded || !visible) { return; }
  let split = U.p0.z; let frame = U.p0.y;
  if (cell % split != frame) { return; }
  let numVox = chunks[cell*24u + 3u];
  let cap = U.p0.w;
  var g = 0u;
  loop {
    if (g*32u >= numVox) { break; }
    let idx = atomicAdd(&counter, 1u);
    if (idx < cap) { requests[idx] = (cell << 4u) | g; }
    g = g + 1u;
  }
}
`;

export const SH_ARGS = /* wgsl */`
@group(0) @binding(0) var<storage, read>       counter : u32;
@group(0) @binding(1) var<storage, read_write> args    : array<u32>;
@group(0) @binding(2) var<uniform>             cap     : u32;
@compute @workgroup_size(1,1,1)
fn main() {
  args[0] = min(counter, cap);
  args[1] = 1u;
  args[2] = 1u;
}
`;

export const SH_BLIT = /* wgsl */`
@group(0) @binding(0) var tex : texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;
struct VO { @builtin(position) pos:vec4<f32>, @location(0) uv:vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i:u32) -> VO {
  var p = array<vec2<f32>,3>(vec2<f32>(-1.0,-1.0), vec2<f32>(3.0,-1.0), vec2<f32>(-1.0,3.0));
  var o:VO; let xy = p[i]; o.pos = vec4<f32>(xy,0.0,1.0);
  o.uv = vec2<f32>((xy.x+1.0)*0.5, 1.0-(xy.y+1.0)*0.5); return o;
}
@fragment fn fs(in:VO) -> @location(0) vec4<f32> {
  return textureSampleLevel(tex, samp, in.uv, 0.0);
}
`;

