"use strict";

/* ---- fast streaming world tunables --------------------------------------- */
export const CH = 32;                       // chunk edge (voxels)
export const WX = 256, WY = 13, WZ = 256;   // world size (chunks) — WY*CH = 416 voxels
                                            // tall. Terrain still sits at ~40–107 (see
                                            // height() VY=128); the extra height above is
                                            // flyable airspace, so the camera can climb to
                                            // ~415 (well past the requested 400).
export const WORLD_VX = WX * CH, WORLD_VY = WY * CH, WORLD_VZ = WZ * CH;
export const WORLD_VOXELS = WORLD_VX * WORLD_VY * WORLD_VZ;
export const WORDS_PER_CHUNK = (CH / 32) * CH * CH;  // bitmask words per chunk (1024 u32s)
export const BYTES_PER_CHUNK = WORDS_PER_CHUNK * 4;
// Edit color pool: same layout as bitmask pool — 1 u32 per bitmask word.
// colorPool[slot * WORDS_PER_CHUNK + lz*CH + ly] = 0xFF_RR_GG_BB for that (lz,ly) column.
// 0x00000000 = use terrain color for all 32 voxels in this word.
// All 32 voxels along X in a word share the same edit color (fine for buildings).
export const BYTES_PER_COLOR_CHUNK = WORDS_PER_CHUNK * 4;  // 4096 bytes = 20 MB total
export const MAX_RESIDENT = 5120;           // GPU slot pool size (~20 MB)
export const CELLS = WX * WY * WZ;

// Index-grid sentinels (must match WGSL; slots are anything < FULL).
export const UNKNOWN = 0xffffffff >>> 0;
export const EMPTY   = 0xfffffffe >>> 0;
export const FULL    = 0xfffffffd >>> 0;

export const UPLOAD_BUDGET = 64;            // max mixed-chunk GPU writes/frame
export const UNLOAD_MARGIN = 3;             // chunks of hysteresis before unloading
