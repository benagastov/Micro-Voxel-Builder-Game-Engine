"use strict";

/* ---- tunables ------------------------------------------------------------ */
export const CFG = {
  MAP: [64, 16, 64],     // initial loaded world size in 8³ chunks → 512 x 128 x 512 voxels
  RES_SCALE: 0.85,       // internal render resolution scale
  LIGHT_SPLIT: 4,        // re-light 1/N of visible chunks each frame
  DIFFUSE_SAMPLES: 1,    // diffuse+shadow rays per voxel per update
  MAX_DIFFUSE_SAMPLES: 900,
  DIFFUSE_BOUNCES: 2,
  SPEC_BOUNCES: 2,
  SHADOW_SOFT: 12.0,
  REQ_CAP: 65535,        // max lighting-request groups/frame (must be <= max workgroups/dim)
  STREAM_TEST: true,     // 3×3 sliding window — center shifts to player sector; neighbours stream via Web Workers
  STREAM_RADIUS: 1,      // render radius in sectors — R=1 → 3×3 window (8 neighbours rendered at once)
  STREAM_LOAD_WORKERS: 4,
  STREAM_UNLOAD_WORKERS: 4,
};
export const CHUNK_U32 = 24, VOX_U32 = 4, MAT_U32 = 8;
export const LIGHT_WG = 32;     // voxels per lighting workgroup (4-bit group offset → 16 groups)

/* ============================  M O U N T A I N S  ===========================
   The world is TOTALLY FLAT everywhere except where these "mountain" entities
   reach.  Each entity is plain JSON in world-voxel XYZ:

       { "x": <0–511>, "y": <summit height>, "z": <0–511>, "r": <radius>, "cliff": <0–1> }

     • x , z  — where the summit sits on the 512×512 map
     • y      — how tall the summit is in voxels (flats sit at ~40; snow caps
                appear above ~92; world ceiling is 124)
     • r      — base radius in voxels (optional → defaults to (y-40)·1.5)
     • cliff  — character, 0..1 (optional, default 0.45):
                  0   = smooth, fully ramped/walkable hill (gentle slopes)
                  1   = sheer terraced cliffs (mostly walls)
                Values between MIX broken cliff faces with slopes.
     • ramp     — optional ramp heading in degrees (default: seeded per entity)
     • windings — optional spiral turns of the ramp (default 1.25)

   EVERY mountain gets at least one guaranteed walkable spiral ramp from base to
   summit; cliffiness only controls the faces between the ramp.  The seed varies
   ridge detail; placement, ramp & character stay put.                         */
export const MOUNTAINS = [
  { "x": 400, "y": 116, "z": 400, "r": 86, "cliff": 0.70 },   // NE hero — snow-capped, behind the keep
  { "x": 110, "y": 100, "z": 400, "r": 80, "cliff": 0.45 },   // NW massif — balanced cliffs & ramps
  { "x": 404, "y":  94, "z": 110, "r": 78, "cliff": 0.55 },   // SE ridge
  { "x": 110, "y":  86, "z": 110, "r": 70, "cliff": 0.30 },   // SW — gentler, mostly ramps
  { "x": 256, "y":  84, "z": 442, "r": 58, "cliff": 0.18 }    // N foothill — smooth & walkable
];

