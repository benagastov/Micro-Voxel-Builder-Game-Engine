import { CFG, CHUNK_U32, VOX_U32, MAT_U32, MOUNTAINS } from "./config.js";
import { SH_SHARED, SH_DRAW, SH_LIGHT, SH_BUILD, SH_ARGS, SH_BLIT } from "./shaders.js";

import { BUILD, BUILD_MATS, circleCursorXZ, circleParams, genCircleTower, squareCursorXZ, squareParams, genSquareRoom, genGhosts } from "./building.js";
"use strict";

/* ============================================================================
   DoonEngine — WebGPU port
   Original (C / OpenGL compute) by frozein:  https://github.com/frozein/DoonEngine
   This file re-implements the core rendering technique in WGSL compute shaders.

   Pipeline per frame:
     1. clear request counter
     2. DRAW   compute  — two-level DDA ray cast → storage texture, flags visible chunks
     3. BUILD  compute  — scan map, append (chunk,voxelGroup) lighting requests for
                          visible+loaded chunks in this frame's split; clears visible bit
     4. ARGS   compute  — write indirect dispatch args = clamp(requestCount, cap)
     5. LIGHT  compute  — indirect; per-voxel diffuse(path)+shadow+specular, accumulate
     6. BLIT   render   — full-screen sample of the storage texture to the canvas
   ============================================================================ */

/* =========================  W O R L D   G E N  ===========================  */
// Builds the sparse chunk representation exactly as the engine expects:
// per non-empty map cell → 16×u32 bitmask + 3 prefix counts, voxels stored in
// ascending local index, interior/occluded faces culled, albedo linearized.

async function buildWorld(progress, seed) {
  const [MX, MY, MZ] = CFG.MAP;
  const VX = MX*8, VY = MY*8, VZ = MZ*8;
  const cells = MX*MY*MZ;

  // ---- seeded fractal terrain (deterministic per seed, Minecraft-style) ----
  const SEED = (seed>>>0) || 1;
  const hf = new Float32Array(VX*VZ);
  // seeded 32-bit integer hash → [0,1)
  const ih=(x,z)=>{ let h=Math.imul(x|0,374761393)^Math.imul(z|0,668265263)^Math.imul(SEED,2246822519);
    h=Math.imul(h^(h>>>15),2246822519); h^=h>>>13; h=Math.imul(h,3266489917); h^=h>>>16; return (h>>>0)/4294967296; };
  const hash=(x,z)=>ih(x,z);
  // seeded per-voxel grain (used for texturing)
  const hash3=(x,y,z)=>{ let h=Math.imul(x|0,374761393)^Math.imul(y|0,668265263)^Math.imul(z|0,1442695041)^Math.imul(SEED,2654435761);
    h=Math.imul(h^(h>>>13),1274126177); return ((h^(h>>>16))>>>0)/4294967295; };
  const vn3=(x,y,z)=>{
    const xi=Math.floor(x),yi=Math.floor(y),zi=Math.floor(z);
    const xf=x-xi,yf=y-yi,zf=z-zi;
    const u=xf*xf*(3-2*xf),v=yf*yf*(3-2*yf),w=zf*zf*(3-2*zf);
    const L=(a,b,t)=>a+(b-a)*t, c=(X,Y,Z)=>hash3(X,Y,Z);
    return L(L(L(c(xi,yi,zi),c(xi+1,yi,zi),u),L(c(xi,yi+1,zi),c(xi+1,yi+1,zi),u),v),
             L(L(c(xi,yi,zi+1),c(xi+1,yi,zi+1),u),L(c(xi,yi+1,zi+1),c(xi+1,yi+1,zi+1),u),v),w);
  };
  const vnoise=(x,z)=>{ const xi=Math.floor(x),zi=Math.floor(z),xf=x-xi,zf=z-zi;
    const u=xf*xf*(3-2*xf), v=zf*zf*(3-2*zf);
    const a=ih(xi,zi),b=ih(xi+1,zi),c=ih(xi,zi+1),d=ih(xi+1,zi+1);
    return a*(1-u)*(1-v)+b*u*(1-v)+c*(1-u)*v+d*u*v; };
  const fbm=(x,z,oct)=>{ let O=oct||5,s=0,amp=1,f=1,norm=0; for(let o=0;o<O;o++){s+=amp*vnoise(x*f,z*f);norm+=amp;amp*=0.5;f*=2;} return s/norm; };
  // ridged multifractal → sharp cliff/ridge profiles
  const ridged=(x,z,oct)=>{ let O=oct||5,s=0,amp=1,f=1,norm=0; for(let o=0;o<O;o++){ let n=vnoise(x*f,z*f); n=1-Math.abs(2*n-1); n*=n; s+=amp*n; norm+=amp; amp*=0.5; f*=2; } return s/norm; };

  const tcx = Math.floor(VX/2), tcz = Math.floor(VZ/2);

  /* Terrain: TOTALLY FLAT plain at one build height; the only elevation comes
     from the MOUNTAINS entity list above.  Each entity raises a radial massif
     whose silhouette is carved by seeded ridged noise (cliffs, ridges, ledges).
     Everywhere outside every entity's radius the ground is dead flat.          */
  const plainsBase = Math.round(VY*0.31);     // the single, perfectly-flat build height (~40)
  const clearR = 120;                          // keep the keep's clearing flat even if a massif overlaps
  // -- pass 1: raise terraced / cliffy massifs from the flat plain --
  for (let z=0; z<VZ; z++) for (let x=0; x<VX; x++){
    let h = plainsBase;                         // dead-flat baseline
    const dKeep = Math.hypot(x-tcx, z-tcz);
    for (let mi=0; mi<MOUNTAINS.length; mi++){
      const m = MOUNTAINS[mi];
      const r = m.r || Math.max(24,(m.y-plainsBase)*1.5);
      const d = Math.hypot(x-m.x, z-m.z);
      if (d >= r) continue;
      let t = 1 - d/r; t = t*t*(3-2*t);                    // smooth radial falloff (foothills → summit)
      const rn = ridged(x*0.013 + m.x*0.7, z*0.013 + m.z*0.7, 5);   // seeded ridge detail
      let s = Math.min(1, Math.max(0, Math.pow(t,1.25)*(0.55+0.80*rn)));
      let mh = plainsBase + (m.y-plainsBase)*s;
      if (dKeep < clearR)                                  // fade the massif out near the keep clearing
        mh = plainsBase + (mh-plainsBase)*Math.min(1,Math.max(0,(dKeep-clearR*0.6)/(clearR*0.4)));
      // broken (non-concentric) terraced cliff faces, scaled by per-entity cliffiness
      const cliffiness=(m.cliff!==undefined)?m.cliff:0.45;
      const region=vnoise(x*0.026+m.x*0.5+9.1, z*0.026+m.z*0.5+4.2);
      const cliffHere=Math.min(1,Math.max(0, cliffiness*1.3*(0.30+region)));
      const step=6;
      const phase=(vnoise(x*0.05+m.x*1.3, z*0.05+m.z*1.3)-0.5)*step*0.9; // jitter breaks contour repetition
      const fl=Math.floor((mh+phase)/step)*step - phase;
      const terraced=fl + (mh-fl)*0.18;
      mh = mh*(1-cliffHere) + terraced*cliffHere;
      h = Math.max(h, mh);
    }
    hf[x+z*VX] = Math.max(3, Math.min(VY-4, h));
  }

  // -- pass 2: carve a GUARANTEED walkable spiral ramp into every mountain --
  // (a clean linear-slope channel, slope ≤ ~0.8/voxel, overwriting terraces so
  //  it can't be blocked by ridge noise or by overlapping massifs)
  for (let mi=0; mi<MOUNTAINS.length; mi++){
    const m = MOUNTAINS[mi];
    if (m.cliff===0) continue;                             // pure-ramp hills don't need carving
    const r = m.r || Math.max(24,(m.y-plainsBase)*1.5);
    const windings = (m.windings!==undefined) ? m.windings : 1.25;
    let rampBase;
    if (m.ramp!==undefined) rampBase = m.ramp*Math.PI/180;
    else { const hs=Math.sin(m.x*12.9898+m.z*78.233)*43758.5453; rampBase=(hs-Math.floor(hs))*6.2831853; }
    const x0=Math.max(0,(m.x-r)|0), x1=Math.min(VX-1,(m.x+r)|0);
    const z0=Math.max(0,(m.z-r)|0), z1=Math.min(VZ-1,(m.z+r)|0);
    for (let z=z0; z<=z1; z++) for (let x=x0; x<=x1; x++){
      const d = Math.hypot(x-m.x, z-m.z); if (d>=r) continue;
      const level = Math.min(1, Math.max(0, 1 - d/r));     // 0 base → 1 summit
      const ang   = Math.atan2(z-m.z, x-m.x);
      const spiral= rampBase + level*windings*6.2831853;
      let da=(ang-spiral)%6.2831853; if(da<0) da+=6.2831853;
      const adist=Math.min(da, 6.2831853-da);
      if (adist >= (0.55 - 0.22*level)) continue;          // outside the ramp wedge
      let rh = plainsBase + (m.y-plainsBase)*level;          // LINEAR cone → gentle, walkable
      const dKeep=Math.hypot(x-tcx,z-tcz);
      if (dKeep < clearR)
        rh = plainsBase + (rh-plainsBase)*Math.min(1,Math.max(0,(dKeep-clearR*0.6)/(clearR*0.4)));
      hf[x+z*VX] = Math.max(3, Math.min(VY-4, rh));         // overwrite to lay the ramp surface
    }
  }
  const waterLevel = Math.floor(VY*0.22);

  /* ============================  T H E   T O W E R  =======================
     A circular stone keep with staggered masonry, mossy lower courses,
     wooden string-courses, arched windows, crenellated battlements and a
     glowing beacon on top — modelled to evoke the "Lay of the Land" look. */
  const plazaH   = plainsBase;
  const Rout = 11, wallT = 3, Rin = Rout - wallT;
  const baseY    = plazaH - 2;
  const towerTop = baseY + 56;
  const merlonTop= towerTop + 5;
  const lanY0 = merlonTop + 3, lanY1 = merlonTop + 5;
  const plazaR = Rout + 8;

  // flatten a circular cobbled plaza so the keep sits on level ground
  for (let z=0; z<VZ; z++) for (let x=0; x<VX; x++){
    const d = Math.hypot(x-tcx, z-tcz);
    if (d < plazaR){
      const e = Math.min(1, Math.max(0,(plazaR-d)/6));   // soft blended rim
      hf[x+z*VX] = hf[x+z*VX]*(1-e) + plazaH*e;
    }
  }

  const angDiff=(a,b)=>{ let d=a-b; while(d>Math.PI)d-=2*Math.PI; while(d<-Math.PI)d+=2*Math.PI; return d; };

  // geometry classifier → 0 empty · 1 stone · 2 wood · 3 lantern(emissive)
  function towerKind(x,y,z){
    const dx=x-tcx, dz=z-tcz, d2=dx*dx+dz*dz;
    // central beacon: small stone pillar capped by a glowing lantern
    if (d2 <= 2.5){
      if (y>=towerTop && y<=merlonTop+2) return 1;
      if (y>=lanY0 && y<=lanY1)          return 3;
    }
    if (y < baseY || y > merlonTop) return 0;
    const d = Math.sqrt(d2);
    if (d > Rout+0.5) return 0;
    const inWall = d>=Rin-0.5 && d<=Rout+0.5;
    const ang = Math.atan2(dz,dx);

    // battlements
    if (y >= towerTop){
      if (y===towerTop && d<=Rout+0.5) return 1;            // walkable stone cap
      if (inWall){
        const seg = Math.floor((ang+Math.PI)/(Math.PI*2)*16);
        if (seg%2===0) return 1;                            // merlon (8 around)
      }
      return 0;                                             // crenel gap
    }

    if (!inWall) return 0;                                  // hollow interior
    const rel = y - baseY;

    // arched doorway facing the camera (−Z / south)
    if (rel>=1 && rel<9 && Math.abs(angDiff(ang,-Math.PI/2))< (rel>7?0.18:0.30)) return 0;

    // tall arched windows — three tiers, four cardinal directions
    if (rel>=14){
      const wy=(rel-14)%16;
      const slot = Math.abs(angDiff(ang,0))<0.15 || Math.abs(angDiff(ang,Math.PI/2))<0.15 ||
                   Math.abs(angDiff(ang,Math.PI))<0.15 || Math.abs(angDiff(ang,-Math.PI/2))<0.15;
      if (slot){
        if (wy>=1 && wy<=6) return 0;                       // window void
        if (wy===7) return (Math.abs(angDiff(ang,0))<0.08||Math.abs(angDiff(ang,Math.PI/2))<0.08||
                            Math.abs(angDiff(ang,Math.PI))<0.08||Math.abs(angDiff(ang,-Math.PI/2))<0.08)?0:1; // arch crown
        if (wy===8) return 2;                               // wooden lintel
      }
    }

    // wooden string-courses banding the shaft
    if (rel===10||rel===11||rel===38||rel===39) return 2;
    return 1;                                               // stone
  }

  // colour a tower voxel (masonry / timber / beacon)
  function towerColor(kind,x,y,z,o){
    const dx=x-tcx, dz=z-tcz, d=Math.hypot(dx,dz), ang=Math.atan2(dz,dx);
    if (kind===3){ o.mat=3; o.r=255; o.g=198; o.b=120; return; } // emissive lantern
    if (kind===2){                                               // timber band
      o.mat=0;
      const arc=ang*Math.max(Rin,d);
      const plank=Math.floor(arc/3.0);
      let v=94 + (hash3(plank*7+5,0,0)-0.5)*26 + (hash3(x,y,z)-0.5)*10;
      if ((((arc%3)+3)%3) < 0.45) v-=22;                         // plank seam
      if ((y&1)===0) v-=3;                                       // grain
      o.r=Math.max(22,Math.min(172,v*1.48));
      o.g=Math.max(14,Math.min(150,v*0.94));
      o.b=Math.max(8, Math.min(120,v*0.55));
      return;
    }
    // stone masonry — staggered circular brickwork with mortar + moss
    o.mat=0;
    const course=Math.floor((y-baseY)/4);
    const arc=ang*Math.max(Rin,d);
    const off=(course%2)*2.5;
    const vJoint=((((arc+off)%5)+5)%5) < 0.6;
    const hJoint=((y-baseY)%4)===0;
    const brick=Math.floor((arc+off)/5.0);
    let v=158 + (hash3(brick*13+1,course*7+3,0)-0.5)*30 + (hash3(x,y,z)-0.5)*14;
    if (vJoint||hJoint) v-=30;                                   // recessed mortar
    let moss=0;
    if (y < baseY+12) moss += (baseY+12-y)/12*0.45;              // mossy footing
    if (vJoint||hJoint) moss += 0.22;
    moss *= (0.4+0.6*hash3(x*2,y*2,z*2));
    moss = Math.min(0.6,moss);
    let r=v*1.02, g=v*1.0, b=v*0.93;
    r=r*(1-moss)+72*moss; g=g*(1-moss)+106*moss; b=b*(1-moss)+54*moss;
    o.r=Math.max(0,Math.min(255,r)); o.g=Math.max(0,Math.min(255,g)); o.b=Math.max(0,Math.min(255,b));
  }

  // ---- random floating lamps removed; the keep carries a single beacon ----
  const lamps = new Set();

  // sample a voxel at world coords → {solid, mat, r,g,b}
  const out = {solid:false,mat:0,r:0,g:0,b:0};
  function sample(x,y,z){
    if (x<0||y<0||z<0||x>=VX||y>=VY||z>=VZ){ out.solid=false; return out; }
    const tk = towerKind(x,y,z);
    if (tk!==0){ out.solid=true; towerColor(tk,x,y,z,out); return out; }
    const key = x+VX*(y+VY*z);
    if (lamps.has(key)){ out.solid=true; out.mat=3; out.r=255; out.g=200; out.b=120; return out; }
    const h = hf[x+z*VX];
    if (y < h){
      out.solid=true; out.mat=0;
      // slope from heightfield gradient
      const hx = hf[Math.min(VX-1,x+1)+z*VX]-hf[Math.max(0,x-1)+z*VX];
      const hz = hf[x+Math.min(VZ-1,z+1)*VX]-hf[x+Math.max(0,z-1)*VX];
      const slope = Math.sqrt(hx*hx+hz*hz)*0.5;
      const top = y > h-1.5;
      const m1 = vn3(x*0.07, y*0.07, z*0.07);   // broad tonal drift
      const g1 = hash3(x,y,z);                  // fine grain
      const inPlaza = Math.hypot(x-tcx, z-tcz) < plazaR-1;
      if (top && inPlaza){                       // cobbled courtyard
        const cob = hash3(Math.floor(x/2),0,Math.floor(z/2));
        let v = 152 + (cob-0.5)*40 + (g1-0.5)*14;
        if (((Math.floor(x/2)+Math.floor(z/2))&1)===0 && (((x+z)%4)===0)) v -= 26; // joints
        out.r=v*1.0; out.g=v*0.99; out.b=v*0.93;
      } else if (y <= waterLevel+1 && top){      // sand
        const v=198+(g1-0.5)*22; out.r=v; out.g=v*0.92; out.b=v*0.72;
      } else if (y>VY*0.72 && slope<=1.7){       // snow cap on high, gentler peaks
        const v=232+(g1-0.5)*16; out.r=v*0.97; out.g=v*0.99; out.b=Math.min(255,v*1.02);
      } else if (slope>1.3){                      // cliff face — streaked warm rock
        const streak = vn3(x*0.28, y*0.05, z*0.28);
        const v=146+(streak-0.5)*50+(g1-0.5)*14;
        out.r=v*1.04; out.g=v*1.0; out.b=v*0.9;
      } else if (top){                            // soft sage grass (ledges + plains)
        out.r=96 +(m1-0.5)*40+(g1-0.5)*14;
        out.g=142+(m1-0.5)*46+(g1-0.5)*16;
        out.b=66 +(m1-0.5)*26+(g1-0.5)*10;
      } else if (y > h-4){                        // dirt under the turf
        out.r=122+(g1-0.5)*20; out.g=92+(g1-0.5)*16; out.b=60+(g1-0.5)*12;
      } else {                                    // deep subsoil / bedrock
        const v=118+(m1-0.5)*40+(g1-0.5)*14; out.r=v*0.95; out.g=v*0.93; out.b=v*0.9;
      }
      out.r=Math.max(0,Math.min(255,out.r));
      out.g=Math.max(0,Math.min(255,out.g));
      out.b=Math.max(0,Math.min(255,out.b));
      return out;
    }
    if (y < waterLevel && h < waterLevel){ out.solid=true; out.mat=1; out.r=40; out.g=110; out.b=170; return out; }
    out.solid=false; return out;
  }
  const solidAt=(x,y,z)=>{
    if (x<0||y<0||z<0||x>=VX||y>=VY||z>=VZ) return false;
    if (towerKind(x,y,z)!==0) return true;
    if (lamps.has(x+VX*(y+VY*z))) return true;
    const h = hf[x+z*VX];
    if (y < h) return true;
    if (y < waterLevel && h < waterLevel) return true;
    return false;
  };

  // ---- per-column max height (chunk-cull hint) ----
  const colMax = new Int32Array(MX*MZ).fill(0);
  for (let cz=0; cz<MZ; cz++) for (let cx=0; cx<MX; cx++){
    let mx=0; for (let z=0;z<8;z++) for (let x=0;x<8;x++){ const h=hf[(cx*8+x)+(cz*8+z)*VX]|0; if(h>mx)mx=h; }
    colMax[cx+cz*MX]=Math.max(mx, waterLevel);
  }
  // the keep rises far above the terrain — keep its chunk columns alive
  {
    const cx0=Math.max(0,((tcx-Rout-1)/8)|0), cx1=Math.min(MX-1,((tcx+Rout+1)/8)|0);
    const cz0=Math.max(0,((tcz-Rout-1)/8)|0), cz1=Math.min(MZ-1,((tcz+Rout+1)/8)|0);
    for (let cz=cz0; cz<=cz1; cz++) for (let cx=cx0; cx<=cx1; cx++)
      colMax[cx+cz*MX]=Math.max(colMax[cx+cz*MX], merlonTop+6);
  }

  const packed = await packWorld({MX,MY,MZ,VX,VY,VZ,cells,solidAt,sample,colMax,progress});
  return { MX,MY,MZ,VX,VY,VZ, cells, ...packed,
    gen:{ solidAt, sample, colMax, hf, plainsBase, tcx, tcz, VX, VY, VZ, MX, MY, MZ } };
}

// Pack a world (terrain + any edit overlay) into the engine's sparse chunk format.
// Reused for the initial build AND for every live edit commit.
async function packWorld(P){
  const {MX,MY,MZ,VX,VY,VZ,cells,solidAt,sample,colMax,progress}=P;
  const mapFlags    = new Uint32Array(cells);
  const mapVoxIdx   = new Uint32Array(cells);
  const chunkData   = new Uint32Array(cells*CHUNK_U32);
  let voxCap = 1<<20;
  let voxData = new Uint32Array(voxCap*VOX_U32);
  let voxCursor = 0, storedChunks = 0;
  const occ = new Uint8Array(512);

  for (let cz=0; cz<MZ; cz++){
    if (progress) await progress(cz/MZ);
    for (let cy=0; cy<MY; cy++) for (let cx=0; cx<MX; cx++){
      // chunk-column cull: skip if no column-max data is provided.
      const cMaxCol = (typeof colMax==='function') ? colMax(cx,cz)
                    : colMax ? colMax[cx + cz*MX] : Infinity;
      if (cy*8 > cMaxCol+1) continue;
      const cell = cx + MX*(cy + MY*cz);
      const ox=cx*8, oy=cy*8, oz=cz*8;
      let any=false;
      for (let z=0;z<8;z++) for (let y=0;y<8;y++) for (let x=0;x<8;x++){
        const s = solidAt(ox+x, oy+y, oz+z) ? 1 : 0;
        occ[x + 8*(y + 8*z)] = s; if (s) any=true;
      }
      if (!any) continue;
      const cbase = cell*CHUNK_U32;
      chunkData[cbase+0] = (cx)|0; chunkData[cbase+1]=(cy)|0; chunkData[cbase+2]=(cz)|0;
      for (let i=0;i<16;i++) chunkData[cbase+7+i]=0;
      if ((voxCursor+512)*VOX_U32 > voxData.length){
        voxCap = Math.max(voxCap*2, voxCursor+512);
        const nv = new Uint32Array(voxCap*VOX_U32); nv.set(voxData); voxData=nv;
      }
      mapVoxIdx[cell] = voxCursor;
      let n=0;
      for (let z=0;z<8;z++) for (let y=0;y<8;y++) for (let x=0;x<8;x++){
        const index = x + 8*(y + 8*z);
        if ((index&31)===0 && index!==0 && ((index>>5)&3)===0)
          chunkData[cbase+4+((index>>7)-1)] = n;
        if (!occ[index]) continue;
        const wx=ox+x, wy=oy+y, wz=oz+z;
        const visible =
          !solidAt(wx+1,wy,wz)||!solidAt(wx-1,wy,wz)||
          !solidAt(wx,wy+1,wz)||!solidAt(wx,wy-1,wz)||
          !solidAt(wx,wy,wz+1)||!solidAt(wx,wy,wz-1);
        if (!visible) continue;
        const s = sample(wx,wy,wz);
        const smat=s.mat, sr=s.r, sg=s.g, sb=s.b;
        let nx=0,ny=0,nz=0;
        if(!solidAt(wx+1,wy,wz))nx+=1; if(!solidAt(wx-1,wy,wz))nx-=1;
        if(!solidAt(wx,wy+1,wz))ny+=1; if(!solidAt(wx,wy-1,wz))ny-=1;
        if(!solidAt(wx,wy,wz+1))nz+=1; if(!solidAt(wx,wy,wz-1))nz-=1;
        let nl=Math.hypot(nx,ny,nz); if(nl<1e-5){nx=0;ny=1;nz=0;nl=1;}
        nx/=nl;ny/=nl;nz/=nl;
        const bnx=Math.max(0,Math.min(255,(((nx*255)|0)+255)>>1));
        const bny=Math.max(0,Math.min(255,(((ny*255)|0)+255)>>1));
        const bnz=Math.max(0,Math.min(255,(((nz*255)|0)+255)>>1));
        chunkData[cbase+7+(index>>5)] |= (1 << (index & 31)) >>> 0;
        const lr = Math.pow(sr/255, CFG_GAMMA)*255;
        const lg = Math.pow(sg/255, CFG_GAMMA)*255;
        const lb = Math.pow(sb/255, CFG_GAMMA)*255;
        const vb = (voxCursor + n)*VOX_U32;
        voxData[vb+0] = (((smat&255)<<24)|((bnx&255)<<16)|((bny&255)<<8)|(bnz&255))>>>0;
        voxData[vb+1] = ((((lr|0)&255)<<24)|(((lg|0)&255)<<16)|(((lb|0)&255)<<8))>>>0;
        voxData[vb+2] = 0; voxData[vb+3] = 0;
        n++;
      }
      chunkData[cbase+3] = n;
      mapFlags[cell] = 2;
      voxCursor += n;
      storedChunks++;
    }
  }
  voxData = voxData.slice(0, Math.max(1, voxCursor)*VOX_U32);
  return { mapFlags, mapVoxIdx, chunkData, voxData, storedVox:voxCursor, storedChunks };
}
const CFG_GAMMA = 2.2;

/* =============================  M A T E R I A L S  =======================  */
function buildMaterials(){
  const m = new Uint32Array(256*MAT_U32);
  const f = (v)=>{ const b=new Float32Array(1); b[0]=v; return new Uint32Array(b.buffer)[0]; };
  function set(i,{emissive=0,opacity=1,refract=1,specular=0,reflectType=0,shininess=1}){
    const o=i*MAT_U32;
    m[o+0]=emissive?1:0; m[o+1]=f(opacity); m[o+2]=f(refract);
    m[o+3]=f(specular); m[o+4]=reflectType; m[o+5]=shininess;
  }
  set(0,{specular:0});                                   // diffuse terrain
  set(1,{specular:0.85,reflectType:1,shininess:6});      // water / metal (reflective)
  set(2,{opacity:0.5,refract:1.52});                     // glass (semi-transparent)
  set(3,{emissive:1});                                   // lamp
  return m;
}

/* =============================  E N G I N E  =============================  */
const $ = (id)=>document.getElementById(id);
let device, ctx, fmt, world, dims;
let buf = {};
let pipe = {}, bind = {};
let outTex, sampler, ghostDepth;
let ghostCubeBuf=null, ghostUni=null, GHOST_MAX=40000;
let uni; // Float32/Uint32 staging
const uniBytes = 13*16; // 13 vec4

const cam = { pos:[0,0,0], yaw:0, pitch:-0.25, fov:70, speed:0.06 };
let viewMode = 0, sunSpin=false, sunAngle=0.7, frameNum=0;
const keys = {};
const MODE_NAMES=["FULL LIGHTING","ALBEDO","DIFFUSE LIGHT","SPECULAR LIGHT","VOXEL NORMAL","FACE NORMAL"];
let spawnPoint = null;

let worldSeed = 1, busy = false;
const makeBuf=(data,usage)=>{ const b=device.createBuffer({size:Math.max(16,data.byteLength),usage}); device.queue.writeBuffer(b,0,data); return b; };
const progressCb = async p=>{ $("lbar").style.width=(p*100|0)+"%"; await new Promise(r=>requestAnimationFrame(r)); };

// "any text" → 32-bit seed; blank = random (Minecraft-style)
function parseSeed(str){
  str=(str||"").trim();
  if (str==="") return (Math.random()*4294967296)>>>0;
  if (/^-?\d+$/.test(str)) return (parseInt(str,10)>>>0)||1;
  let h=2166136261; for(let i=0;i<str.length;i++){ h^=str.charCodeAt(i); h=Math.imul(h,16777619); } return (h>>>0)||1;
}

// drop the camera into the flat clearing, looking up at the keep (3/4, sun side)
function placeCamera(){
  const cx=world.VX/2/8, cz=world.VZ/2/8;
  const plazaH=Math.round(world.VY*0.31);
  cam.pos=[cx-5, plazaH/8+2.6, cz-6];
  const tgt=[cx, (plazaH+30)/8, cz];
  const dx=tgt[0]-cam.pos[0], dy=tgt[1]-cam.pos[1], dz=tgt[2]-cam.pos[2];
  cam.yaw=Math.atan2(dx,dz); cam.pitch=Math.atan2(dy, Math.hypot(dx,dz));
  setSpawnPoint(false);
}

function absoluteCamPos(){
  const p=[cam.pos[0], cam.pos[1], cam.pos[2]];
  if (world?.streamBase){
    p[0] += streamState.originSector.x * world.streamBase.SMX;
    p[2] += streamState.originSector.z * world.streamBase.SMZ;
  }
  return p;
}

function setSpawnPoint(showMessage=true){
  if (!world) return;
  spawnPoint = absoluteCamPos();
  updateRadar();
  if (showMessage) flashBanner("SPAWN POINT SET · radar updated");
}

function shiftSpawnPoint(dx, dz){
  if (!spawnPoint) return;
  spawnPoint[0] += dx; spawnPoint[2] += dz;
  updateRadar();
}

function radarSaveOffset(){
  if (!world?.streamBase) return [0,0];
  const R=CFG.STREAM_RADIUS;
  return [world.streamBase.SMX*R, world.streamBase.SMZ*R];
}

function livePosToSavePos(pos){
  const [ox,oz]=radarSaveOffset();
  return [+(pos[0]-ox).toFixed(4), +pos[1].toFixed(4), +(pos[2]-oz).toFixed(4)];
}

function savePosToAbsolutePos(pos){
  if (!Array.isArray(pos) || pos.length < 3 || !pos.every(Number.isFinite)) return null;
  const [ox,oz]=radarSaveOffset();
  return [pos[0]+ox, pos[1], pos[2]+oz];
}

function savePosToLocalCamPos(pos){
  const p=savePosToAbsolutePos(pos); if (!p) return null;
  if (world?.streamBase){
    p[0] -= streamState.originSector.x * world.streamBase.SMX;
    p[2] -= streamState.originSector.z * world.streamBase.SMZ;
  }
  return p;
}

function setStreamOriginForSavedPosition(savePos){
  if (!world?.streamBase || !Array.isArray(savePos) || savePos.length < 3) return;
  const {SMX,SMZ}=world.streamBase;
  const R=CFG.STREAM_RADIUS;
  const localX=savePos[0]+SMX*R, localZ=savePos[2]+SMZ*R;
  streamState.originSector.x = Math.floor(localX/SMX) - R;
  streamState.originSector.z = Math.floor(localZ/SMZ) - R;
}

function updateRadar(){
  const dot=$("radar-spawn"), needle=$("radar-needle"), distEl=$("radar-dist");
  if (!dot || !needle || !distEl || !world || !spawnPoint) return;
  const p=absoluteCamPos();
  const dx=spawnPoint[0]-p[0], dz=spawnPoint[2]-p[2];
  const distChunks=Math.hypot(dx,dz), distVox=distChunks*8;
  distEl.textContent = distVox < 1000 ? `${Math.round(distVox)}m` : `${(distVox/1000).toFixed(1)}km`;

  const range=Math.max(12, world.streamBase?.SMX || world.MX || 64);
  const clamped=Math.min(42, (distChunks/range)*42);

  // Radar screen convention: +X on the SVG is east/right, +Y is south/down.
  // Game convention here: when facing +Z (north), camRight points to -X,
  // so world east is -X and world west is +X. Mirror X for compass/radar use.
  const east  = -dx;
  const north =  dz;
  const sx = 50 + (distChunks > 1e-6 ? (east  / distChunks) * clamped : 0);
  const sy = 50 - (distChunks > 1e-6 ? (north / distChunks) * clamped : 0);

  dot.setAttribute("cx", sx.toFixed(1));
  dot.setAttribute("cy", sy.toFixed(1));
  dot.setAttribute("r", distChunks < 0.75 ? "5.5" : "4");

  // Same mirror for the facing needle: yaw +90 means west, not east.
  needle.setAttribute("transform", `rotate(${(-cam.yaw*180/Math.PI).toFixed(1)} 50 50)`);
}
function updateStats(){
  $("dim").textContent = `${world.VX}×${world.VY}×${world.VZ}`;
  $("vox").textContent = fmtNum(world.storedVox);
  $("chk").textContent = fmtNum(world.storedChunks);
  const vram = (world.voxData.byteLength+world.chunkData.byteLength+world.mapFlags.byteLength*2+CFG.REQ_CAP*4);
  $("vram").textContent = (vram/1048576).toFixed(0)+" MB";
  const ss=$("streamstat"); if (ss) ss.textContent = streamState.text;
  const sv=$("seedval"); if (sv) sv.textContent = worldSeed;
}

const streamState = {
  text:"idle", workers:[], unloadWorkers:[], active:false,
  done:0, total:0, jobs:[], next:0, nextJobId:1, targetKey:"0,0", lastTargetKey:"0,0",
  loadedJobs:new Set(), visibleSectors:new Set(["0,0"]), derendering:new Set(),
  originSector:{x:0, z:0},  // cumulative sector shift from world origin
  workerCenter:null, recenterScratch:null, scratchFlip:0, zeroU32:null
};
// monotonic across the whole session so a job id is never reused in a later
// window frame — survivor re-arming relies on loadedJobs ids staying unique.
function allocJobId(){ return streamState.nextJobId++; }

// ---- fast streaming knobs ---------------------------------------------------
// Minecraft-ish behaviour: keep a hot worker pool, prioritize nearby sectors,
// and stream chunk regions in larger batches so the browser is not crushed by
// postMessage/GPU-write overhead.
function streamWorkerBudget(){
  const hc = (navigator.hardwareConcurrency || 8);
  return Math.max(4, Math.min(12, hc - 1));
}
function streamJobStep(center){
  // 16×16 chunk-column jobs are much faster than 8×8 here: 4× fewer worker
  // messages and merge passes, while still appearing progressively.
  return Math.max(8, Math.min(16, center.MX || 16, center.MZ || 16));
}
function zeroU32(n){
  if (!streamState.zeroU32 || streamState.zeroU32.length < n) streamState.zeroU32 = new Uint32Array(n);
  return streamState.zeroU32.subarray(0,n);
}
function getRecenterScratch(){
  const needCells=world.cells, needChunks=world.cells*CHUNK_U32;
  if (!streamState.recenterScratch || streamState.recenterScratch[0].mapFlags.length!==needCells){
    streamState.recenterScratch=[0,1].map(()=>({
      mapFlags:new Uint32Array(needCells),
      mapVoxIdx:new Uint32Array(needCells),
      chunkData:new Uint32Array(needChunks)
    }));
    streamState.scratchFlip=0;
  }
  streamState.scratchFlip ^= 1;
  const s=streamState.recenterScratch[streamState.scratchFlip];
  s.mapFlags.fill(0); s.mapVoxIdx.fill(0); s.chunkData.fill(0);
  return s;
}
function uploadStreamingLayoutNoRebind(){
  // Same GPU buffers, same dimensions: just rewrite layout data. No resize(), no
  // bind-group churn, and no voxel-buffer upload.
  device.queue.writeBuffer(buf.mapFlags,  0, world.mapFlags);
  device.queue.writeBuffer(buf.mapVoxIdx, 0, world.mapVoxIdx);
  device.queue.writeBuffer(buf.chunks,    0, world.chunkData);
  if (buf.sampleCnt) device.queue.writeBuffer(buf.sampleCnt, 0, zeroU32(world.cells));
}

function stopStreamWorkers(){
  for (const w of streamState.workers) w.terminate?.();
  for (const w of streamState.unloadWorkers) w.terminate?.();
  streamState.workers.length = 0;
  streamState.unloadWorkers.length = 0;
  streamState.jobs.length = 0; streamState.loadedJobs.clear(); streamState.visibleSectors = new Set(["0,0"]); streamState.derendering.clear();
  streamState.active = false; streamState.done = 0; streamState.total = 0; streamState.text = "idle";
  streamState.nextJobId = 1;
  streamState.originSector.x = 0; streamState.originSector.z = 0;
  streamState.workerCenter = null;
  for (const s of streamState.recenterScratch || []){ s.mapFlags.fill(0); s.mapVoxIdx.fill(0); s.chunkData.fill(0); }
  const ss=$("streamstat"); if (ss) ss.textContent = streamState.text;
}

function createSectorWorkerURL(){
  const src = `
"use strict";
const CHUNK_U32=${CHUNK_U32}, VOX_U32=${VOX_U32}, CFG_GAMMA=${CFG_GAMMA};
const MOUNTAINS=${JSON.stringify(MOUNTAINS)};
function packVoxel(mat,r,g,b,nx,ny,nz){
  const bnx=Math.max(0,Math.min(255,(((nx*255)|0)+255)>>1));
  const bny=Math.max(0,Math.min(255,(((ny*255)|0)+255)>>1));
  const bnz=Math.max(0,Math.min(255,(((nz*255)|0)+255)>>1));
  const lr=Math.pow(r/255, CFG_GAMMA)*255, lg=Math.pow(g/255, CFG_GAMMA)*255, lb=Math.pow(b/255, CFG_GAMMA)*255;
  return [((mat&255)<<24)|(bnx<<16)|(bny<<8)|bnz, ((lr&255)<<24)|((lg&255)<<16)|((lb&255)<<8)|255, 0, 0];
}

function buildSectorJob(msg){
  const {id, seed, sectorX, sectorZ, originX, originZ, absSX, absSZ, baseCX, baseCZ, MX, MY, MZ, VY, x0, x1, z0, z1, needsClear=false}=msg;
  let voxCap=1<<16, voxData=new Uint32Array(voxCap*VOX_U32), voxCursor=0, storedChunks=0;
  let chunkCap=256, cells=new Uint32Array(chunkCap), starts=new Uint32Array(chunkCap), chunkData=new Uint32Array(chunkCap*CHUNK_U32);
  const SEED=(seed>>>0)||1, VX=MX*8, VZ=MZ*8, plainsBase=Math.round(VY*0.31), waterLevel=Math.floor(VY*0.22), occ=new Uint8Array(512);
  const tcx=Math.floor(VX/2), tcz=Math.floor(VZ/2), clearR=120;
  const plazaH=plainsBase, Rout=11, wallT=3, Rin=Rout-wallT, baseY=plazaH-2, towerTop=baseY+56, merlonTop=towerTop+5, lanY0=merlonTop+3, lanY1=merlonTop+5, plazaR=Rout+8;
  const ih=(x,z)=>{ let h=Math.imul(x|0,374761393)^Math.imul(z|0,668265263)^Math.imul(SEED,2246822519); h=Math.imul(h^(h>>>15),2246822519); h^=h>>>13; h=Math.imul(h,3266489917); h^=h>>>16; return (h>>>0)/4294967296; };
  const hash3=(x,y,z)=>{ let h=Math.imul(x|0,374761393)^Math.imul(y|0,668265263)^Math.imul(z|0,1442695041)^Math.imul(SEED,2654435761); h=Math.imul(h^(h>>>13),1274126177); return ((h^(h>>>16))>>>0)/4294967295; };
  const vnoise=(x,z)=>{ const xi=Math.floor(x),zi=Math.floor(z),xf=x-xi,zf=z-zi; const u=xf*xf*(3-2*xf),v=zf*zf*(3-2*zf); const a=ih(xi,zi),b=ih(xi+1,zi),c=ih(xi,zi+1),d=ih(xi+1,zi+1); return a*(1-u)*(1-v)+b*u*(1-v)+c*(1-u)*v+d*u*v; };
  const ridged=(x,z,oct)=>{ let O=oct||5,s=0,amp=1,f=1,norm=0; for(let o=0;o<O;o++){ let n=vnoise(x*f,z*f); n=1-Math.abs(2*n-1); n*=n; s+=amp*n; norm+=amp; amp*=0.5; f*=2; } return s/norm; };
  const vn3=(x,y,z)=>{
    const xi=Math.floor(x),yi=Math.floor(y),zi=Math.floor(z);
    const xf=x-xi,yf=y-yi,zf=z-zi;
    const u=xf*xf*(3-2*xf),v=yf*yf*(3-2*yf),w=zf*zf*(3-2*zf);
    const L=(a,b,t)=>a+(b-a)*t, c=(X,Y,Z)=>hash3(X,Y,Z);
    return L(L(L(c(xi,yi,zi),c(xi+1,yi,zi),u),L(c(xi,yi+1,zi),c(xi+1,yi+1,zi),u),v),
             L(L(c(xi,yi,zi+1),c(xi+1,yi,zi+1),u),L(c(xi,yi+1,zi+1),c(xi+1,yi+1,zi+1),u),v),w);
  };
  const angDiff=(a,b)=>{ let d=a-b; while(d>Math.PI)d-=2*Math.PI; while(d<-Math.PI)d+=2*Math.PI; return d; };
  const height=(wx,wz)=>{
    let h=plainsBase;
    const dKeep=Math.hypot(wx-tcx,wz-tcz);
    for(let mi=0;mi<MOUNTAINS.length;mi++){
      const m=MOUNTAINS[mi], r=m.r||Math.max(24,(m.y-plainsBase)*1.5), d=Math.hypot(wx-m.x,wz-m.z);
      if(d>=r) continue;
      let t=1-d/r; t=t*t*(3-2*t);
      const rn=ridged(wx*0.013+m.x*0.7,wz*0.013+m.z*0.7,5);
      let s=Math.min(1,Math.max(0,Math.pow(t,1.25)*(0.55+0.80*rn)));
      let mh=plainsBase+(m.y-plainsBase)*s;
      if(dKeep<clearR) mh=plainsBase+(mh-plainsBase)*Math.min(1,Math.max(0,(dKeep-clearR*0.6)/(clearR*0.4)));
      const cliffiness=(m.cliff!==undefined)?m.cliff:0.45;
      const region=vnoise(wx*0.026+m.x*0.5+9.1,wz*0.026+m.z*0.5+4.2);
      const cliffHere=Math.min(1,Math.max(0,cliffiness*1.3*(0.30+region)));
      const step=6, phase=(vnoise(wx*0.05+m.x*1.3,wz*0.05+m.z*1.3)-0.5)*step*0.9;
      const fl=Math.floor((mh+phase)/step)*step-phase, terraced=fl+(mh-fl)*0.18;
      mh=mh*(1-cliffHere)+terraced*cliffHere;
      h=Math.max(h,mh);
    }
    for(let mi=0;mi<MOUNTAINS.length;mi++){
      const m=MOUNTAINS[mi]; if(m.cliff===0) continue;
      const r=m.r||Math.max(24,(m.y-plainsBase)*1.5), d=Math.hypot(wx-m.x,wz-m.z); if(d>=r) continue;
      const level=Math.min(1,Math.max(0,1-d/r));
      const ang=Math.atan2(wz-m.z,wx-m.x);
      let rampBase;
      if(m.ramp!==undefined) rampBase=m.ramp*Math.PI/180; else { const hs=Math.sin(m.x*12.9898+m.z*78.233)*43758.5453; rampBase=(hs-Math.floor(hs))*6.2831853; }
      const spiral=rampBase+level*((m.windings!==undefined)?m.windings:1.25)*6.2831853;
      let da=(ang-spiral)%6.2831853; if(da<0) da+=6.2831853;
      if(Math.min(da,6.2831853-da) >= (0.55-0.22*level)) continue;
      let rh=plainsBase+(m.y-plainsBase)*level;
      if(dKeep<clearR) rh=plainsBase+(rh-plainsBase)*Math.min(1,Math.max(0,(dKeep-clearR*0.6)/(clearR*0.4)));
      h=Math.max(3,Math.min(VY-4,rh));
    }
    const dp=Math.hypot(wx-tcx,wz-tcz);
    if(dp<plazaR){ const e=Math.min(1,Math.max(0,(plazaR-dp)/6)); h=h*(1-e)+plazaH*e; }
    return Math.max(3,Math.min(VY-4,h));
  };
  const hCache=new Map();
  const cachedHeight=(wx,wz)=>{
    const k=wx+','+wz;
    let h=hCache.get(k);
    if(h===undefined){ h=height(wx,wz); hCache.set(k,h); }
    return h;
  };
  const towerKind=(x,y,z)=>{
    const dx=x-tcx,dz=z-tcz,d2=dx*dx+dz*dz;
    if(d2<=2.5){ if(y>=towerTop&&y<=merlonTop+2)return 1; if(y>=lanY0&&y<=lanY1)return 3; }
    if(y<baseY||y>merlonTop)return 0;
    const d=Math.sqrt(d2); if(d>Rout+0.5)return 0;
    const inWall=d>=Rin-0.5&&d<=Rout+0.5, ang=Math.atan2(dz,dx);
    if(y>=towerTop){ if(y===towerTop&&d<=Rout+0.5)return 1; if(inWall){ const seg=Math.floor((ang+Math.PI)/(Math.PI*2)*16); if(seg%2===0)return 1; } return 0; }
    if(!inWall)return 0;
    const rel=y-baseY;
    if(rel>=1&&rel<9&&Math.abs(angDiff(ang,-Math.PI/2))<(rel>7?0.18:0.30))return 0;
    if(rel>=14){ const wy=(rel-14)%16; const slot=Math.abs(angDiff(ang,0))<0.15||Math.abs(angDiff(ang,Math.PI/2))<0.15||Math.abs(angDiff(ang,Math.PI))<0.15||Math.abs(angDiff(ang,-Math.PI/2))<0.15; if(slot){ if(wy>=1&&wy<=6)return 0; if(wy===7)return (Math.abs(angDiff(ang,0))<0.08||Math.abs(angDiff(ang,Math.PI/2))<0.08||Math.abs(angDiff(ang,Math.PI))<0.08||Math.abs(angDiff(ang,-Math.PI/2))<0.08)?0:1; if(wy===8)return 2; } }
    if(rel===10||rel===11||rel===38||rel===39)return 2;
    return 1;
  };
  const towerColor=(kind,x,y,z)=>{
    const dx=x-tcx,dz=z-tcz,d=Math.hypot(dx,dz),ang=Math.atan2(dz,dx);
    if(kind===3)return {mat:3,r:255,g:198,b:120};
    if(kind===2){ const arc=ang*Math.max(Rin,d), plank=Math.floor(arc/3.0); let v=94+(hash3(plank*7+5,0,0)-0.5)*26+(hash3(x,y,z)-0.5)*10; if((((arc%3)+3)%3)<0.45)v-=22; if((y&1)===0)v-=3; return {mat:0,r:Math.max(22,Math.min(172,v*1.48)),g:Math.max(14,Math.min(150,v*0.94)),b:Math.max(8,Math.min(120,v*0.55))}; }
    const course=Math.floor((y-baseY)/4), arc=ang*Math.max(Rin,d), off=(course%2)*2.5, vJoint=((((arc+off)%5)+5)%5)<0.6, hJoint=((y-baseY)%4)===0, brick=Math.floor((arc+off)/5.0);
    let v=158+(hash3(brick*13+1,course*7+3,0)-0.5)*30+(hash3(x,y,z)-0.5)*14; if(vJoint||hJoint)v-=30;
    let moss=0; if(y<baseY+12)moss+=(baseY+12-y)/12*0.45; if(vJoint||hJoint)moss+=0.22; moss*=0.4+0.6*hash3(x*2,y*2,z*2); moss=Math.min(0.6,moss);
    let r=v*1.02,g=v*1.0,b=v*0.93; r=r*(1-moss)+72*moss; g=g*(1-moss)+106*moss; b=b*(1-moss)+54*moss;
    return {mat:0,r:Math.max(0,Math.min(255,r)),g:Math.max(0,Math.min(255,g)),b:Math.max(0,Math.min(255,b))};
  };
  const solidAt=(wx,y,wz)=>{ if(y<0||y>=VY) return false; if(towerKind(wx,y,wz)!==0)return true; const h=cachedHeight(wx,wz); return y<h || (y<waterLevel && h<waterLevel); };
  const sample=(wx,y,wz)=>{
    const tk=towerKind(wx,y,wz); if(tk!==0)return towerColor(tk,wx,y,wz);
    const h=cachedHeight(wx,wz), g=hash3(wx,y,wz); let r,gc,b;
    if (y < h){
      const hx=cachedHeight(wx+1,wz)-cachedHeight(wx-1,wz), hz=cachedHeight(wx,wz+1)-cachedHeight(wx,wz-1), slope=Math.sqrt(hx*hx+hz*hz)*0.5;
      const m1=vn3(wx*0.07, y*0.07, wz*0.07);
      const top=y>h-1.5, inPlaza=Math.hypot(wx-tcx,wz-tcz)<plazaR-1;
      if(top&&inPlaza){ const cob=hash3(Math.floor(wx/2),0,Math.floor(wz/2)); let v=152+(cob-0.5)*40+(g-0.5)*14; if(((Math.floor(wx/2)+Math.floor(wz/2))&1)===0&&(((wx+wz)%4)===0))v-=26; r=v; gc=v*0.99; b=v*0.93; }
      else if(y<=waterLevel+1&&top){ const v=198+(g-0.5)*22; r=v; gc=v*0.92; b=v*0.72; }
      else if(y>VY*0.72&&slope<=1.7){ const v=232+(g-0.5)*16; r=v*0.97; gc=v*0.99; b=Math.min(255,v*1.02); }
      else if(slope>1.3){ const streak=vn3(wx*0.28,y*0.05,wz*0.28); const v=146+(streak-0.5)*50+(g-0.5)*14; r=v*1.04; gc=v; b=v*0.9; }
      else if(top){ r=96+(m1-0.5)*40+(g-0.5)*14; gc=142+(m1-0.5)*46+(g-0.5)*16; b=66+(m1-0.5)*26+(g-0.5)*10; }
      else if(y>h-4){ r=120+(g-0.5)*18; gc=88+(g-0.5)*14; b=58+(g-0.5)*10; }
      else { const v=118+(m1-0.5)*40+(g-0.5)*14; r=v*0.95; gc=v*0.93; b=v*0.9; }
      return {mat:0,r:Math.max(0,Math.min(255,r)),g:Math.max(0,Math.min(255,gc)),b:Math.max(0,Math.min(255,b))};
    }
    return {mat:1,r:40,g:110,b:170};
  };
  const growChunks=()=>{ const nc=new Uint32Array(chunkCap*2), ns=new Uint32Array(chunkCap*2), nd=new Uint32Array(chunkCap*2*CHUNK_U32); nc.set(cells); ns.set(starts); nd.set(chunkData); chunkCap*=2; cells=nc; starts=ns; chunkData=nd; };
  for(let cz=z0;cz<z1;cz++) for(let cy=0;cy<MY;cy++) for(let cx=x0;cx<x1;cx++){
    const cell=cx+MX*(cy+MY*cz), ox=(baseCX+cx)*8, oy=cy*8, oz=(baseCZ+cz)*8; let any=false;
    for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){ const s=solidAt(ox+x,oy+y,oz+z)?1:0; occ[x+8*(y+8*z)]=s; if(s)any=true; }
    if(!any) continue;
    if(storedChunks>=chunkCap) growChunks();
    const cbase=storedChunks*CHUNK_U32; chunkData[cbase]=cx; chunkData[cbase+1]=cy; chunkData[cbase+2]=cz;
    if((voxCursor+512)*VOX_U32>voxData.length){ const nv=new Uint32Array(voxData.length*2); nv.set(voxData); voxData=nv; }
    const vStart=voxCursor; let n=0;
    for(let z=0;z<8;z++)for(let y=0;y<8;y++)for(let x=0;x<8;x++){
      const index=x+8*(y+8*z); if((index&31)===0 && index!==0 && ((index>>5)&3)===0) chunkData[cbase+4+((index>>7)-1)] = n;
      if(!occ[index]) continue; const wx=ox+x, wy=oy+y, wz=oz+z;
      const vis=!solidAt(wx+1,wy,wz)||!solidAt(wx-1,wy,wz)||!solidAt(wx,wy+1,wz)||!solidAt(wx,wy-1,wz)||!solidAt(wx,wy,wz+1)||!solidAt(wx,wy,wz-1);
      if(!vis) continue;
      let nx=0,ny=0,nz=0; if(!solidAt(wx+1,wy,wz))nx++; if(!solidAt(wx-1,wy,wz))nx--; if(!solidAt(wx,wy+1,wz))ny++; if(!solidAt(wx,wy-1,wz))ny--; if(!solidAt(wx,wy,wz+1))nz++; if(!solidAt(wx,wy,wz-1))nz--;
      let nl=Math.hypot(nx,ny,nz)||1; nx/=nl; ny/=nl; nz/=nl; chunkData[cbase+7+(index>>5)] |= (1 << (index & 31)) >>> 0;
      const s=sample(wx,wy,wz), pv=packVoxel(s.mat,s.r,s.g,s.b,nx,ny,nz), off=(voxCursor+n)*VOX_U32; voxData[off]=pv[0]; voxData[off+1]=pv[1]; voxData[off+2]=pv[2]; voxData[off+3]=pv[3]; n++;
    }
    if(n>0){ chunkData[cbase+3]=n; cells[storedChunks]=cell; starts[storedChunks]=vStart; voxCursor+=n; storedChunks++; }
  }
  voxData=voxData.slice(0,Math.max(1,voxCursor)*VOX_U32);
  cells=cells.slice(0,storedChunks); starts=starts.slice(0,storedChunks); chunkData=chunkData.slice(0,storedChunks*CHUNK_U32);
  return {id, sectorX, sectorZ, originX, originZ, absSX, absSZ, x0, x1, z0, z1, MX, MY, MZ, needsClear, cells, starts, chunkData, voxData, storedVox:voxCursor, storedChunks};
}
onmessage=e=>{ const r=buildSectorJob(e.data); postMessage(r,[r.cells.buffer,r.starts.buffer,r.chunkData.buffer,r.voxData.buffer]); };
`;
  return URL.createObjectURL(new Blob([src], {type:"text/javascript"}));
}

function createUnloadWorkerURL(){
  const src = `"use strict"; onmessage=e=>{
    const {id,sx,sz,SMX,SMY,SMZ,MX,MY,R}=e.data;
    const baseX=(sx+R)*SMX, baseZ=(sz+R)*SMZ;
    const cells=new Uint32Array(SMX*SMY*SMZ); let n=0;
    for(let cz=0;cz<SMZ;cz++)for(let cy=0;cy<SMY;cy++)for(let cx=0;cx<SMX;cx++)
      cells[n++]=(baseX+cx)+MX*(cy+MY*(baseZ+cz));
    postMessage({id,sx,sz,cells},[cells.buffer]);
  };`;
  return URL.createObjectURL(new Blob([src], {type:"text/javascript"}));
}

function makeExpandedWorldFromCenter(center){
  const R=CFG.STREAM_RADIUS;
  const SMX=center.MX, SMY=center.MY, SMZ=center.MZ, MX=SMX*(2*R+1), MY=SMY, MZ=SMZ*(2*R+1);
  const cells=MX*MY*MZ, mapFlags=new Uint32Array(cells), mapVoxIdx=new Uint32Array(cells), chunkData=new Uint32Array(cells*CHUNK_U32);
  let voxData=new Uint32Array(Math.max(center.voxData.length, center.storedVox*VOX_U32));
  voxData.set(center.voxData.subarray(0, center.storedVox*VOX_U32));
  for(let cz=0;cz<SMZ;cz++)for(let cy=0;cy<SMY;cy++)for(let cx=0;cx<SMX;cx++){
    const oldCell=cx+SMX*(cy+SMY*cz), ncx=cx+SMX*R, ncz=cz+SMZ*R, newCell=ncx+MX*(cy+MY*ncz);
    mapFlags[newCell]=center.mapFlags[oldCell]; mapVoxIdx[newCell]=center.mapVoxIdx[oldCell];
    const oldBase=oldCell*CHUNK_U32, newBase=newCell*CHUNK_U32;
    chunkData.set(center.chunkData.subarray(oldBase, oldBase+CHUNK_U32), newBase);
    chunkData[newBase]=ncx; chunkData[newBase+2]=ncz;
  }
  return {...center, MX,MY,MZ,VX:MX*8,VY:MY*8,VZ:MZ*8,cells,mapFlags,mapVoxIdx,chunkData,voxData,streamBase:{SMX,SMY,SMZ},streamLoaded:new Set(["0,0"])};
}

function remapCenterEditsToExpanded(center, expanded){
  if (ABS_EDITS.size){ syncLiveEditsFromPersistent(); return; }
  if (!EDITS.size || !expanded?.streamBase) return;
  const R=CFG.STREAM_RADIUS;
  const oldVX=center.VX, oldVY=center.VY;
  const ox=expanded.streamBase.SMX*8*R, oz=expanded.streamBase.SMZ*8*R;
  const remapped=new Map();
  for (const [k,e] of EDITS){
    const x =  k % oldVX;
    const y = ((k / oldVX) | 0) % oldVY;
    const z =  (k / (oldVX * oldVY)) | 0;
    const nk = (x+ox) + expanded.VX*(y + expanded.VY*(z+oz));
    remapped.set(nk, e);
  }
  EDITS.clear();
  for (const [k,e] of remapped) EDITS.set(k,e);
}

function ensureStreamingVoxelCapacity(needU32){
  if (world.voxData.length >= needU32) return;
  const nv=new Uint32Array(Math.max(needU32, world.voxData.length*2)); nv.set(world.voxData); world.voxData=nv;
  buf.voxels?.destroy?.();
  buf.voxels=device.createBuffer({size:world.voxData.byteLength, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(buf.voxels,0,world.voxData,0,world.storedVox*VOX_U32);
  buf._voxCap=world.voxData.length; rebindWorldBuffers();
}

function mergeStreamSector(sec){
  if (!world?.streamBase) return;
  if (sec.originX !== streamState.originSector.x || sec.originZ !== streamState.originSector.z) return;
  streamState.loadedJobs.add(sec.id);
  const R=CFG.STREAM_RADIUS;
  const {SMX,SMY,SMZ}=world.streamBase, dstBaseCX=(sec.sectorX+R)*SMX, dstBaseCZ=(sec.sectorZ+R)*SMZ;
  clearStreamJobRegion(sec, dstBaseCX, dstBaseCZ);
  const startVox=world.storedVox, needU32=(startVox+sec.storedVox)*VOX_U32;
  ensureStreamingVoxelCapacity(needU32);
  world.voxData.set(sec.voxData.subarray(0,sec.storedVox*VOX_U32), startVox*VOX_U32);
  device.queue.writeBuffer(buf.voxels,startVox*VOX_U32*4,world.voxData.subarray(startVox*VOX_U32,needU32));
  for(let i=0;i<sec.storedChunks;i++){
    const sCell=sec.cells[i], cx=sCell%SMX, cy=((sCell/SMX)|0)%SMY, cz=(sCell/(SMX*SMY))|0;
    const dcx=dstBaseCX+cx, dcz=dstBaseCZ+cz, dCell=dcx+world.MX*(cy+world.MY*dcz);
    const sBase=i*CHUNK_U32, dBase=dCell*CHUNK_U32;
    world.chunkData.set(sec.chunkData.subarray(sBase,sBase+CHUNK_U32),dBase);
    world.chunkData[dBase]=dcx; world.chunkData[dBase+2]=dcz;
    world.mapFlags[dCell]=2; world.mapVoxIdx[dCell]=startVox+sec.starts[i];
    device.queue.writeBuffer(buf.chunks,dBase*4,world.chunkData.subarray(dBase,dBase+CHUNK_U32));
    writeU1(buf.mapVoxIdx,dCell*4,world.mapVoxIdx[dCell]); writeU1(buf.mapFlags,dCell*4,2); writeU1(buf.sampleCnt,dCell*4,0);
  }
  // Terrain streaming writes raw generated chunks.  If the save/build overlay has
  // edits inside this sector, immediately re-pack those chunks so streamed-in
  // neighbours do not erase player changes.
  const edited = editedCellsForSector(sec.sectorX, sec.sectorZ);
  world.storedVox += sec.storedVox; world.storedChunks += sec.storedChunks;
  if (edited.size) rebuildChunks(edited);
  streamState.visibleSectors.add(`${sec.sectorX},${sec.sectorZ}`);
  streamState.done++; streamState.text=`load ${streamState.done}/${streamState.total} · target ${streamState.targetKey}`;
  updateStats();
}

function clearStreamJobRegion(sec, dstBaseCX, dstBaseCZ){
  // Normal streaming writes into zeroed/newly-slid regions, so clearing every
  // 8³ chunk before every merge is pure overhead. Keep this path only for
  // forced refresh jobs that explicitly request it.
  if (!sec.needsClear) return;
  const zero = new Uint32Array(1);
  const blankChunk = new Uint32Array(CHUNK_U32);
  for(let cz=sec.z0; cz<sec.z1; cz++) for(let cy=0; cy<world.MY; cy++) for(let cx=sec.x0; cx<sec.x1; cx++){
    const dcx=dstBaseCX+cx, dcz=dstBaseCZ+cz, cell=dcx+world.MX*(cy+world.MY*dcz);
    if(cell<0 || cell>=world.cells) continue;
    world.mapFlags[cell]=0; world.mapVoxIdx[cell]=0;
    const base=cell*CHUNK_U32;
    world.chunkData.set(blankChunk, base);
    world.chunkData[base]=dcx; world.chunkData[base+1]=cy; world.chunkData[base+2]=dcz;
    device.queue.writeBuffer(buf.chunks, base*4, world.chunkData.subarray(base, base+CHUNK_U32));
    writeU1(buf.mapFlags, cell*4, 0); writeU1(buf.mapVoxIdx, cell*4, 0); writeU1(buf.sampleCnt, cell*4, 0);
  }
}

function editedCellsForSector(sx, sz){
  const out=new Set();
  if (!world?.streamBase || !ABS_EDITS.size) return out;
  const R=CFG.STREAM_RADIUS;
  const {SMX,SMY,SMZ}=world.streamBase;
  const ax0=(streamState.originSector.x+sx)*SMX*8, ax1=ax0+SMX*8;
  const az0=(streamState.originSector.z+sz)*SMZ*8, az1=az0+SMZ*8;
  const MX=world.MX, MY=world.MY;
  const add=(x,y,z)=>{ if(x<0||y<0||z<0||x>=world.VX||y>=world.VY||z>=world.VZ) return;
    out.add((x>>3)+MX*((y>>3)+MY*(z>>3))); };
  for (const [key] of ABS_EDITS){
    const [ax,y,az]=key.split(',').map(Number);
    if (ax<ax0 || ax>=ax1 || az<az0 || az>=az1) continue;
    const [x,ly,z]=authoringVoxelToLive(ax,y,az);
    add(x,ly,z);
    add(x+1,ly,z); add(x-1,ly,z);
    add(x,ly+1,z); add(x,ly-1,z);
    add(x,ly,z+1); add(x,ly,z-1);
  }
  return out;
}

function streamJobPriority(j){
  const [tx,tz]=streamState.targetKey.split(',').map(Number);
  const targetPenalty = (j.sx===tx && j.sz===tz) ? -10000 : Math.abs(j.sx-tx)*600 + Math.abs(j.sz-tz)*600;
  return targetPenalty + j.edgeDist;
}

function pickNextStreamJob(){
  let best=-1, bestP=Infinity;
  for(let i=0;i<streamState.jobs.length;i++){
    const j=streamState.jobs[i]; if(j.sent) continue;
    const p=streamJobPriority(j); if(p<bestP){ bestP=p; best=i; }
  }
  if(best<0) return null;
  streamState.jobs[best].sent=true;
  return streamState.jobs[best];
}

function launchNextStreamJob(w, center){
  const j=pickNextStreamJob(); if(!j) return false;
  // absSX/absSZ: world-absolute sector coords used for deterministic terrain generation.
  // sectorX/sectorZ: GPU-grid coords (−R..R) used by mergeStreamSector for placement.
  const o=streamState.originSector;
  const absSX=(o.x+j.sx), absSZ=(o.z+j.sz);
  w._busy = true;
  w._center = center;
  w.postMessage({id:j.id, seed:worldSeed, sectorX:j.sx, sectorZ:j.sz,
    originX:o.x, originZ:o.z, absSX, absSZ,
    baseCX:absSX*center.MX, baseCZ:absSZ*center.MZ,
    MX:center.MX, MY:center.MY, MZ:center.MZ, VY:center.VY,
    x0:j.x0, x1:j.x1, z0:j.z0, z1:j.z1, needsClear:!!j.needsClear});
  return true;
}

function streamHasPendingJobs(){
  return streamState.jobs.some(j=>!j.sent);
}
function finishStreamingIfIdle(){
  if (streamHasPendingJobs()) return;
  if (streamState.workers.some(w=>w._busy)) return;
  streamState.active=false;
  streamState.text="dynamic stream ready";
  updateStats();
}
function ensureStreamWorkers(center){
  if(!streamState.workerURL) return;
  streamState.workerCenter=center;
  const want=Math.min(streamWorkerBudget(), Math.max(1, streamState.jobs.length || 1));
  while(streamState.workers.length < want){
    const w=new Worker(streamState.workerURL);
    w._busy=false; w._center=center;
    w.onmessage=e=>{
      w._busy=false;
      mergeStreamSector(e.data);
      scheduleDerenderUnused();
      pumpStreamWorkers(w._center || streamState.workerCenter || center);
    };
    w.onerror=err=>{
      console.error("stream worker",err);
      w._busy=false; streamState.text="worker error"; updateStats();
      pumpStreamWorkers(w._center || streamState.workerCenter || center);
    };
    streamState.workers.push(w);
  }
}
function pumpStreamWorkers(center){
  if(!center || !streamState.workerURL) return;
  ensureStreamWorkers(center);
  for(const w of streamState.workers){
    w._center=center;
    if(!w._busy) launchNextStreamJob(w, center);
  }
  finishStreamingIfIdle();
}

function sectorKey(sx,sz){ return `${sx},${sz}`; }

function wantedSectorSet(){
  const [tx,tz]=streamState.targetKey.split(',').map(Number);
  const R=CFG.STREAM_RADIUS, want=new Set(["0,0"]);
  for(let dz=-1;dz<=1;dz++) for(let dx=-1;dx<=1;dx++){
    const nx=tx+dx, nz=tz+dz;
    if(Math.abs(nx)<=R && Math.abs(nz)<=R) want.add(sectorKey(nx,nz));
  }
  return want;
}

function updateStreamTarget(){
  if(!world?.streamBase) return;
  const {SMX,SMZ}=world.streamBase;
  const R=CFG.STREAM_RADIUS;
  const localX=cam.pos[0]-R*SMX, localZ=cam.pos[2]-R*SMZ;
  const sx=Math.max(-R, Math.min(R, Math.floor(localX/SMX)));
  const sz=Math.max(-R, Math.min(R, Math.floor(localZ/SMZ)));
  // Player crossed into a neighbouring sector — slide the window so they're at (0,0) again
  if(sx!==0 || sz!==0){ recenterWorld(sx, sz); return; }
  rerenderWantedSectors();
  scheduleDerenderUnused();
}

function applyDerenderCells(cells, sx, sz){
  if (!world) return;
  const zero = new Uint32Array(1);
  for (let i=0;i<cells.length;i++){
    const cell=cells[i];
    if (world.mapFlags[cell]!==0){
      world.mapFlags[cell]=0;
      device.queue.writeBuffer(buf.mapFlags, cell*4, zero);
      device.queue.writeBuffer(buf.sampleCnt, cell*4, zero);
    }
  }
  streamState.visibleSectors.delete(sectorKey(sx,sz));
  streamState.derendering.delete(sectorKey(sx,sz));
  updateStats();
}

function rerenderSector(sx, sz){
  if(!world?.streamBase) return;
  const key=sectorKey(sx,sz); if(streamState.visibleSectors.has(key)) return;
  const R=CFG.STREAM_RADIUS;
  const {SMX,SMY,SMZ}=world.streamBase, baseX=(sx+R)*SMX, baseZ=(sz+R)*SMZ;
  const two=new Uint32Array([2]), zero=new Uint32Array([0]);
  let any=false;
  for(let cz=0;cz<SMZ;cz++)for(let cy=0;cy<SMY;cy++)for(let cx=0;cx<SMX;cx++){
    const cell=(baseX+cx)+world.MX*(cy+world.MY*(baseZ+cz));
    if(world.chunkData[cell*CHUNK_U32+3]>0){
      world.mapFlags[cell]=2; device.queue.writeBuffer(buf.mapFlags, cell*4, two); device.queue.writeBuffer(buf.sampleCnt, cell*4, zero); any=true;
    }
  }
  if(any) streamState.visibleSectors.add(key);
}

function rerenderWantedSectors(){
  for(const key of wantedSectorSet()){
    const [sx,sz]=key.split(',').map(Number);
    rerenderSector(sx,sz);
  }
}

function derenderSectorInline(sx,sz){
  const R=CFG.STREAM_RADIUS;
  const {SMX,SMY,SMZ}=world.streamBase; const MX=world.MX, MY=world.MY;
  const baseX=(sx+R)*SMX, baseZ=(sz+R)*SMZ;
  const cells=new Uint32Array(SMX*SMY*SMZ); let n=0;
  for(let cz=0;cz<SMZ;cz++)for(let cy=0;cy<SMY;cy++)for(let cx=0;cx<SMX;cx++)
    cells[n++]=(baseX+cx)+MX*(cy+MY*(baseZ+cz));
  applyDerenderCells(cells,sx,sz);
}

// Slide the 3×3 GPU window so the player is always at sector (0,0).
// shiftX/shiftZ: the sector the player just entered (e.g. 1,0 = moved east).
function recenterWorld(shiftX, shiftZ){
  if(!world?.streamBase) return;
  const R=CFG.STREAM_RADIUS;
  const {SMX,SMY,SMZ}=world.streamBase;
  const MX=world.MX, MY=world.MY;

  // Remap CPU arrays row-by-row into alternating scratch buffers. This avoids
  // allocating/GCing ~60 MB and avoids the old per-cell copy loop.
  const oldMapFlags=world.mapFlags, oldMapVoxIdx=world.mapVoxIdx, oldChunkData=world.chunkData;
  const scratch=getRecenterScratch();
  const newMapFlags=scratch.mapFlags, newMapVoxIdx=scratch.mapVoxIdx, newChunkData=scratch.chunkData;
  for(let gz=-R;gz<=R;gz++) for(let gx=-R;gx<=R;gx++){
    const old_gx=gx+shiftX, old_gz=gz+shiftZ;
    if(Math.abs(old_gx)>R || Math.abs(old_gz)>R) continue; // sector leaving window
    const new_bx=(gx+R)*SMX, new_bz=(gz+R)*SMZ;
    const old_bx=(old_gx+R)*SMX, old_bz=(old_gz+R)*SMZ;
    for(let cz=0;cz<SMZ;cz++) for(let cy=0;cy<SMY;cy++){
      const oldRow=old_bx + MX*(cy + MY*(old_bz+cz));
      const newRow=new_bx + MX*(cy + MY*(new_bz+cz));
      newMapFlags.set(oldMapFlags.subarray(oldRow, oldRow+SMX), newRow);
      newMapVoxIdx.set(oldMapVoxIdx.subarray(oldRow, oldRow+SMX), newRow);
      newChunkData.set(oldChunkData.subarray(oldRow*CHUNK_U32, (oldRow+SMX)*CHUNK_U32), newRow*CHUNK_U32);
      // Stored chunk coordinates must match the new GPU-grid slot.
      for(let cx=0;cx<SMX;cx++){
        const nb=(newRow+cx)*CHUNK_U32;
        newChunkData[nb]=new_bx+cx;
        newChunkData[nb+1]=cy;
        newChunkData[nb+2]=new_bz+cz;
      }
    }
  }
  world.mapFlags=newMapFlags; world.mapVoxIdx=newMapVoxIdx; world.chunkData=newChunkData;

  // Shift camera so player stays at the same physical voxel position
  cam.pos[0]-=shiftX*SMX; cam.pos[2]-=shiftZ*SMZ;

  // Remap visible sectors set
  const newVisible=new Set();
  for(const key of streamState.visibleSectors){
    const [kx,kz]=key.split(',').map(Number);
    const nx=kx-shiftX, nz=kz-shiftZ;
    if(Math.abs(nx)<=R && Math.abs(nz)<=R) newVisible.add(sectorKey(nx,nz));
  }
  streamState.visibleSectors=newVisible;

  // Accumulate absolute origin for deterministic terrain generation
  streamState.originSector.x+=shiftX; streamState.originSector.z+=shiftZ;
  streamState.targetKey="0,0"; streamState.derendering.clear();
  syncLiveEditsFromPersistent();

  uploadStreamingLayoutNoRebind();
  rerenderWantedSectors();
  queueNewSectors(shiftX, shiftZ);
}

// Re-establish full streaming coverage after the window slides by (shiftX,shiftZ).
//
// REGRESSION THIS FIXES: the old code filtered out every already-dispatched job
// here, while mergeStreamSector silently drops any worker result whose origin no
// longer matches the current one.  So a sector job still in flight when the window
// recentred was abandoned: its region had been cleared but was never refilled,
// leaving permanent empty/black chunks.  Returning to spawn then showed the tower
// and mountains riddled with holes — "they don't come back".
//
// Now: every UNMERGED old job is remapped into the new window frame and re-armed,
// and fresh full-sector jobs are added only for sectors the slide newly exposed.
// Result: coverage self-heals on every recenter, no matter how fast you travel.
function queueNewSectors(shiftX, shiftZ){
  if(!streamState.workerURL || !world?.streamBase) return;
  const R=CFG.STREAM_RADIUS;
  const center={MX:world.streamBase.SMX,MY:world.streamBase.SMY,MZ:world.streamBase.SMZ,VY:world.VY};
  const STEP=streamJobStep(center);

  // 1) carry over unfinished jobs, shifted into the new frame.
  // If the player steps into a neighbour before that neighbour has finished
  // streaming, its remaining jobs remap to (0,0).  Keep those jobs too: the
  // centre sector is "carried" only for chunks that had already arrived.
  const jobs=[];
  const covered=new Set();
  for(const j of streamState.jobs){
    if(streamState.loadedJobs.has(j.id)) continue;      // already merged → present as geometry
    const nsx=j.sx-shiftX, nsz=j.sz-shiftZ;             // old GPU coords → new frame
    if(Math.abs(nsx)>R || Math.abs(nsz)>R) continue;    // scrolled out of the window
    jobs.push({id:j.id, sx:nsx, sz:nsz, x0:j.x0, x1:j.x1, z0:j.z0, z1:j.z1, edgeDist:j.edgeDist, sent:false});
    covered.add(nsx+','+nsz);
  }

  // 2) fresh jobs for sectors that just entered the window for the first time
  for(let sz=-R;sz<=R;sz++) for(let sx=-R;sx<=R;sx++){
    if(sx===0 && sz===0) continue;                      // centre existed in the old window; unfinished centre jobs were carried above
    const old_sx=sx+shiftX, old_sz=sz+shiftZ;
    if(Math.abs(old_sx)<=R && Math.abs(old_sz)<=R) continue;  // carried sector (covered by survivors if incomplete)
    if(covered.has(sx+','+sz)) continue;
    for(let z0=0;z0<center.MZ;z0+=STEP) for(let x0=0;x0<center.MX;x0+=STEP){
      const edgeDist = Math.min(
        sx<0 ? (center.MX-1-x0) : sx>0 ? x0 : Math.min(x0, center.MX-1-x0),
        sz<0 ? (center.MZ-1-z0) : sz>0 ? z0 : Math.min(z0, center.MZ-1-z0)
      );
      jobs.push({id:allocJobId(),sx,sz,x0,x1:Math.min(center.MX,x0+STEP),z0,z1:Math.min(center.MZ,z0+STEP),edgeDist,sent:false});
    }
  }

  streamState.jobs = jobs;
  if(!jobs.length) return;
  streamState.total = streamState.done + jobs.length;
  streamState.active=true;
  streamState.text=`stream ${streamState.done}/${streamState.total} · recenter`; updateStats();

  // Keep the worker pool hot instead of destroying/recreating workers per slide.
  pumpStreamWorkers(center);
}
function scheduleDerenderUnused(){
  if(!world?.streamBase || !streamState.unloadURL) return;
  const want=wantedSectorSet();
  for(const key of Array.from(streamState.visibleSectors)){
    if(want.has(key) || key==="0,0" || streamState.derendering.has(key)) continue;
    const [sx,sz]=key.split(',').map(Number);
    streamState.derendering.add(key);
    let w; try{ w=new Worker(streamState.unloadURL); } catch(err){
      derenderSectorInline(sx,sz); // blob: workers blocked on null origin (file://)
      if(streamState.unloadWorkers.length>=CFG.STREAM_UNLOAD_WORKERS) break; continue;
    }
    streamState.unloadWorkers.push(w);
    w.onmessage=e=>{ applyDerenderCells(e.data.cells, e.data.sx, e.data.sz); w.terminate(); streamState.unloadWorkers=streamState.unloadWorkers.filter(x=>x!==w); };
    w.onerror=e=>{ console.error("unload worker", e); streamState.derendering.delete(key); w.terminate(); };
    const {SMX,SMY,SMZ}=world.streamBase;
    w.postMessage({id:key,sx,sz,SMX,SMY,SMZ,MX:world.MX,MY:world.MY,R:CFG.STREAM_RADIUS});
    if(streamState.unloadWorkers.length>=CFG.STREAM_UNLOAD_WORKERS) break;
  }
}

function startNeighborStreaming(){
  if (!CFG.STREAM_TEST || !world || streamState.active || streamState.jobs.length) return;
  streamState.active=true; streamState.done=0; streamState.total=0; streamState.text="expanding"; updateStats();
  let center;
  if (world.streamBase){
    const {SMX,SMY,SMZ}=world.streamBase;
    center={MX:SMX,MY:SMY,MZ:SMZ,VY:world.VY};
  } else {
    center=world; world=makeExpandedWorldFromCenter(center); dims=world;
    remapCenterEditsToExpanded(center, world);
    uploadWorld(); resetLighting();
    const R0=CFG.STREAM_RADIUS;
    cam.pos[0]+=center.MX*R0; cam.pos[2]+=center.MZ*R0;
    shiftSpawnPoint(center.MX*R0, center.MZ*R0);
  }
  const R=CFG.STREAM_RADIUS;
  streamState.text=`alloc ${(2*R+1)**2} sectors`; updateStats();
  flashBanner("STREAM ALLOCATED · loading border chunks");
  const url=createSectorWorkerURL(); const jobs=[];
  streamState.workerURL=url;
  streamState.unloadURL=createUnloadWorkerURL();
  const STEP=streamJobStep(center);
  for(let sz=-R;sz<=R;sz++)for(let sx=-R;sx<=R;sx++) if(sx||sz){
    for(let z0=0;z0<center.MZ;z0+=STEP) for(let x0=0;x0<center.MX;x0+=STEP){
      const edgeDist = Math.min(
        sx<0 ? (center.MX-1-x0) : sx>0 ? x0 : Math.min(x0, center.MX-1-x0),
        sz<0 ? (center.MZ-1-z0) : sz>0 ? z0 : Math.min(z0, center.MZ-1-z0)
      );
      jobs.push({id:allocJobId(),sx,sz,x0,x1:Math.min(center.MX,x0+STEP),z0,z1:Math.min(center.MZ,z0+STEP),edgeDist,sent:false});
    }
  }
  streamState.jobs=jobs;
  streamState.total=jobs.length; streamState.text=`load 0/${streamState.total} · target ${streamState.targetKey}`; updateStats();
  pumpStreamWorkers(center);
}

// rebuild the world live from a new seed (no page reload)
async function regenerate(seed){
  if (busy || !device) return;
  stopStreamWorkers();
  EDITS.clear(); colMaxEdit=null;
  busy = true; worldSeed = seed>>>0;
  $("ltext").textContent = "Generating world · seed "+worldSeed+" …";
  $("lbar").style.width="0%"; $("loading").style.display="flex";
  await new Promise(r=>setTimeout(r,30));
  await device.queue.onSubmittedWorkDone?.();
  const rw=dims.rw, rh=dims.rh;
  world = await buildWorld(progressCb, worldSeed);
  dims = world; dims.rw=rw; dims.rh=rh;
  uploadWorld();
  resetLighting();
  placeCamera();
  updateStats();
  $("loading").style.display="none";
  busy = false;
  setTimeout(startNeighborStreaming, 80);
}

// (re)upload the current `world` arrays to the GPU and rebind. Grows the voxel
// buffer only when needed; fixed-size buffers are just rewritten.
function uploadWorld(){
  const SU = GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;
  const needMapBytes = Math.max(16, world.mapFlags.byteLength);
  const needIdxBytes = Math.max(16, world.mapVoxIdx.byteLength);
  const needChunkBytes = Math.max(16, world.chunkData.byteLength);
  const needSampleBytes = Math.max(16, world.cells*4);
  if (!buf.mapFlags || (buf._mapBytes||0) < needMapBytes){
    buf.mapFlags?.destroy?.(); buf.mapFlags = device.createBuffer({size:needMapBytes, usage:SU}); buf._mapBytes=needMapBytes;
  }
  if (!buf.mapVoxIdx || (buf._idxBytes||0) < needIdxBytes){
    buf.mapVoxIdx?.destroy?.(); buf.mapVoxIdx = device.createBuffer({size:needIdxBytes, usage:SU}); buf._idxBytes=needIdxBytes;
  }
  if (!buf.chunks || (buf._chunkBytes||0) < needChunkBytes){
    buf.chunks?.destroy?.(); buf.chunks = device.createBuffer({size:needChunkBytes, usage:SU}); buf._chunkBytes=needChunkBytes;
  }
  if (!buf.sampleCnt || (buf._sampleBytes||0) < needSampleBytes){
    buf.sampleCnt?.destroy?.(); buf.sampleCnt = device.createBuffer({size:needSampleBytes, usage:SU}); buf._sampleBytes=needSampleBytes;
  }
  if (!buf.voxels || (buf._voxCap||0) < world.voxData.length){
    buf.voxels?.destroy?.();
    buf.voxels = makeBuf(world.voxData, SU);
    buf._voxCap = world.voxData.length;
  } else {
    device.queue.writeBuffer(buf.voxels, 0, world.voxData);
  }
  // keep the CPU voxel mirror as large as the GPU buffer so post-regen edits
  // append in place (instant) instead of forcing a grow on the first click.
  if (world.voxData.length < buf._voxCap){
    const nv=new Uint32Array(buf._voxCap); nv.set(world.voxData); world.voxData=nv;
  }
  device.queue.writeBuffer(buf.mapFlags, 0, world.mapFlags);
  device.queue.writeBuffer(buf.mapVoxIdx,0, world.mapVoxIdx);
  device.queue.writeBuffer(buf.chunks,   0, world.chunkData);
  device.queue.writeBuffer(buf.sampleCnt,0, new Uint32Array(world.cells));
  if (pipe.build){
    bind.build = device.createBindGroup({layout:pipe.build.getBindGroupLayout(0),entries:[
      {binding:0,resource:{buffer:buf.mapFlags}},{binding:1,resource:{buffer:buf.chunks}},
      {binding:2,resource:{buffer:buf.requests}},{binding:3,resource:{buffer:buf.counter}},
      {binding:4,resource:{buffer:buf.uni2}}]});
  }
  resize();          // rebinds draw/light/blit against current buffers + outTex
}

/* ===========================================================================
   B U I L D   M O D E  — Minecraft + Sims wall-mode + SketchUp line tool.
   Edits live in an overlay map; committing re-packs the sparse world (terrain
   + edits) and re-uploads.  A ghost-block preview shows where blocks will land.
   =========================================================================== */
const EDITS = new Map();                 // vkey → {s:1,mat,r,g,b}  (s:0 = removed)
const ABS_EDITS = new Map();             // "authoringX,y,authoringZ" → edit, survives streaming/recenter
let colMaxEdit = null;                    // per-column max edit height (chunk-cull)
const vkey=(x,y,z)=> x + world.VX*(y + world.VY*z);

function editClone(e){ return e.s===1 ? {s:1,mat:e.mat,r:e.r,g:e.g,b:e.b} : {s:0}; }
function absEditKey(x,y,z){ return `${x},${y},${z}`; }
function liveVoxelToAuthoring(x,y,z){
  if (!world?.streamBase) return [x,y,z];
  const R=CFG.STREAM_RADIUS, {SMX,SMZ}=world.streamBase;
  return [x + (streamState.originSector.x - R)*SMX*8, y, z + (streamState.originSector.z - R)*SMZ*8];
}
function authoringVoxelToLive(x,y,z){
  if (!world?.streamBase) return [x,y,z];
  const R=CFG.STREAM_RADIUS, {SMX,SMZ}=world.streamBase;
  return [x - (streamState.originSector.x - R)*SMX*8, y, z - (streamState.originSector.z - R)*SMZ*8];
}
function syncLiveEditsFromPersistent(){
  if (!world) return;
  EDITS.clear(); colMaxEdit=null;
  const VX=world.VX, VY=world.VY, VZ=world.VZ;
  for (const [key,e] of ABS_EDITS){
    const [ax,y,az]=key.split(',').map(Number);
    const [x,ly,z]=authoringVoxelToLive(ax,y,az);
    if (x<0||ly<0||z<0||x>=VX||ly>=VY||z>=VZ) continue;
    EDITS.set(vkey(x,ly,z), editClone(e));
  }
}

const EOUT={solid:true,mat:0,r:0,g:0,b:0};
const BASEOUT={solid:false,mat:0,r:0,g:0,b:0};
function baseHash3(x,y,z){
  let h=Math.imul(x|0,374761393)^Math.imul(y|0,668265263)^Math.imul(z|0,1442695041)^Math.imul(worldSeed>>>0,2654435761);
  h=Math.imul(h^(h>>>13),1274126177); return ((h^(h>>>16))>>>0)/4294967295;
}
function baseVNoise3(x,y,z){
  const xi=Math.floor(x),yi=Math.floor(y),zi=Math.floor(z);
  const xf=x-xi,yf=y-yi,zf=z-zi;
  const u=xf*xf*(3-2*xf),v=yf*yf*(3-2*yf),w=zf*zf*(3-2*zf);
  const L=(a,b,t)=>a+(b-a)*t, c=(X,Y,Z)=>baseHash3(X,Y,Z);
  return L(L(L(c(xi,yi,zi),c(xi+1,yi,zi),u),L(c(xi,yi+1,zi),c(xi+1,yi+1,zi),u),v),
           L(L(c(xi,yi,zi+1),c(xi+1,yi,zi+1),u),L(c(xi,yi+1,zi+1),c(xi+1,yi+1,zi+1),u),v),w);
}
function centerLocalVoxel(x,z){
  if (!world.streamBase || !world.gen) return [x,z];
  // Map the live (window-local) voxel to ABSOLUTE base-world coords.  The base
  // world (tower + mountains) lives at absolute [0,gen.VX/VZ); everything else
  // is flat plains.  This MUST follow the current stream origin — the worker
  // generates every sector at absolute coords, so the base region drifts through
  // the 3x3 window as the player travels.  Using only the static window-center
  // offset (the old behaviour) reconstructs the base region in the wrong cell,
  // so a rebuild overwrites streamed-in mountain/tower terrain with flat plains
  // and the now-empty neighbour chunks get CULLED on return to spawn.
  const R=CFG.STREAM_RADIUS, {SMX,SMZ}=world.streamBase;
  const ax = x + (streamState.originSector.x - R)*SMX*8;
  const az = z + (streamState.originSector.z - R)*SMZ*8;
  if (ax>=0 && az>=0 && ax<world.gen.VX && az<world.gen.VZ) return [ax,az];
  return null;
}
function baseSolidAt(x,y,z){
  if (x<0||y<0||z<0||x>=world.VX||y>=world.VY||z>=world.VZ) return false;
  const local=centerLocalVoxel(x,z);
  if (local) return world.gen.solidAt(local[0], y, local[1]);
  const plainsBase=Math.round(world.VY*0.31), waterLevel=Math.floor(world.VY*0.22);
  return y<plainsBase || (y<waterLevel && plainsBase<waterLevel);
}
function baseSample(x,y,z){
  const local=centerLocalVoxel(x,z);
  if (local) return world.gen.sample(local[0], y, local[1]);
  const plainsBase=Math.round(world.VY*0.31), waterLevel=Math.floor(world.VY*0.22);
  if (y<plainsBase){
    const g=baseHash3(x,y,z), m1=baseVNoise3(x*0.07,y*0.07,z*0.07);
    let r,gc,b;
    if (y>plainsBase-1.5){ r=96+(m1-0.5)*40+(g-0.5)*14; gc=142+(m1-0.5)*46+(g-0.5)*16; b=66+(m1-0.5)*26+(g-0.5)*10; }
    else if (y>plainsBase-4){ r=120+(g-0.5)*18; gc=88+(g-0.5)*14; b=58+(g-0.5)*10; }
    else { const v=118+(m1-0.5)*40+(g-0.5)*14; r=v*0.95; gc=v*0.93; b=v*0.9; }
    BASEOUT.solid=true; BASEOUT.mat=0;
    BASEOUT.r=Math.max(0,Math.min(255,r)); BASEOUT.g=Math.max(0,Math.min(255,gc)); BASEOUT.b=Math.max(0,Math.min(255,b));
    return BASEOUT;
  }
  if (y<waterLevel && plainsBase<waterLevel){ BASEOUT.solid=true; BASEOUT.mat=1; BASEOUT.r=40; BASEOUT.g=110; BASEOUT.b=170; return BASEOUT; }
  BASEOUT.solid=false; BASEOUT.mat=0; BASEOUT.r=0; BASEOUT.g=0; BASEOUT.b=0; return BASEOUT;
}
function solidAtEdited(x,y,z){
  if (x<0||y<0||z<0||x>=world.VX||y>=world.VY||z>=world.VZ) return false;
  const e=EDITS.get(x + world.VX*(y + world.VY*z));
  if (e!==undefined) return e.s===1;
  return baseSolidAt(x,y,z);
}
function sampleEdited(x,y,z){
  const e=EDITS.get(x + world.VX*(y + world.VY*z));
  if (e!==undefined){
    if (e.s!==1){ EOUT.solid=false; return EOUT; }
    // a little per-voxel grain so placed blocks aren't dead flat
    let g=Math.sin(x*12.9898+y*78.233+z*37.719)*43758.5453; g=g-Math.floor(g);
    const j=(g-0.5)*16;
    EOUT.solid=true; EOUT.mat=e.mat;
    EOUT.r=Math.max(0,Math.min(255,e.r+j));
    EOUT.g=Math.max(0,Math.min(255,e.g+j));
    EOUT.b=Math.max(0,Math.min(255,e.b+j));
    return EOUT;
  }
  return baseSample(x,y,z);
}

// ---- cursor/crosshair ray (matches the path tracer's ray-gen) ----
function aimRay(){
  const {fwd,r,up}=camBasis();
  const aspect=dims.rw/dims.rh, th=Math.tan(cam.fov*Math.PI/360);
  const nx=BUILD.on?BUILD.aim.x:0, ny=BUILD.on?BUILD.aim.y:0;
  let dx=fwd[0]+nx*aspect*th*r[0]+ny*th*up[0];
  let dy=fwd[1]+nx*aspect*th*r[1]+ny*th*up[1];
  let dz=fwd[2]+nx*aspect*th*r[2]+ny*th*up[2];
  const L=Math.hypot(dx,dy,dz)||1; return {ro:[cam.pos[0]*8,cam.pos[1]*8,cam.pos[2]*8], rd:[dx/L,dy/L,dz/L]};
}
// DDA voxel raycast → first solid hit and the empty cell just before it
function raycastVoxel(){
  const {ro,rd}=aimRay();
  let px=ro[0],py=ro[1],pz=ro[2];
  let ix=Math.floor(px),iy=Math.floor(py),iz=Math.floor(pz);
  const sx=rd[0]>=0?1:-1, sy=rd[1]>=0?1:-1, sz=rd[2]>=0?1:-1;
  const tdx=Math.abs(1/(rd[0]||1e-9)), tdy=Math.abs(1/(rd[1]||1e-9)), tdz=Math.abs(1/(rd[2]||1e-9));
  let txm=((sx>0?ix+1-px:px-ix))*tdx, tym=((sy>0?iy+1-py:py-iy))*tdy, tzm=((sz>0?iz+1-pz:pz-iz))*tdz;
  let nfx=0,nfy=0,nfz=0;
  let tCur=0;
  for (let i=0;i<420;i++){
    if (iy>=0 && iy<world.VY && solidAtEdited(ix,iy,iz)){
      return { hit:[ix,iy,iz], place:[ix-nfx,iy-nfy,iz-nfz], face:[nfx,nfy,nfz], pos:[px+rd[0]*tCur, py+rd[1]*tCur, pz+rd[2]*tCur] };
    }
    if (txm<tym && txm<tzm){ tCur=txm; ix+=sx; txm+=tdx; nfx=sx; nfy=0; nfz=0; }
    else if (tym<tzm){ tCur=tym; iy+=sy; tym+=tdy; nfx=0; nfy=sy; nfz=0; }
    else { tCur=tzm; iz+=sz; tzm+=tdz; nfx=0; nfy=0; nfz=sz; }
  }
  return null;
}

function refreshGhosts(){
  if (!BUILD.on){ BUILD.ghosts=[]; uploadGhosts(); return; }
  const rc=raycastVoxel();
  BUILD._lastRC = rc;
  BUILD._lastHit = rc? rc.hit : null;
  BUILD.hover = rc? rc.place : null;
  BUILD.ghosts = genGhosts();
  uploadGhosts();
  const ht=$("bd-hover"); if(ht) ht.textContent = BUILD.hover? `${BUILD.hover[0]}, ${BUILD.hover[1]}, ${BUILD.hover[2]}` : '—';
}

/* ===========================================================================
   I N C R E M E N T A L   C O M M I T   ("instant build")

   Instead of re-packing the whole 64×16×64-chunk world and re-uploading every
   GPU buffer (the old path, which needed a "Building…" overlay), an edit now
   re-packs ONLY the handful of 8³ chunks it actually touches and streams just
   those tiny regions to the GPU.  This is the same strategy the reference
   sparse-microvoxel engine uses, so a placed wall/disc/arc appears the instant
   the click lands — no stall, no loading bar, no global re-light flash.

   Key pieces:
     • applyVoxels() writes the edit overlay AND returns the exact set of
       affected chunks (the touched chunk plus any face-neighbour chunk, since
       adding/removing a voxel can expose or occlude a neighbour's face).
     • rebuildChunks() re-packs each affected chunk in place using a single
       512-voxel scratch pass, reusing the chunk's existing voxel slot when the
       new voxel count fits, else appending at world.storedVox.  Per chunk it
       uploads only: 24×u32 chunk record, the voxel sub-range, mapVoxIdx,
       mapFlags and a zeroed sampleCnt (re-lights just that chunk).
     • ensureVoxCapacity() grows the voxel buffer only if the append cursor ever
       runs past the (generous, pre-allocated) headroom — normally never fires.
   =========================================================================== */

// reused scratch so we never allocate inside the hot path
const _occ      = new Uint8Array(512);
const _scratch  = new Uint32Array(512*VOX_U32);   // packed voxels for one chunk
const _u1       = new Uint32Array(1);
const writeU1 = (b, byteOff, v)=>{ _u1[0]=v>>>0; device.queue.writeBuffer(b, byteOff, _u1); };

// ---- apply edits to the overlay, returning the chunks that must be rebuilt ----
function applyVoxels(list, remove){
  const m = BUILD_MATS[BUILD.matIndex];
  const VX=world.VX, VY=world.VY, VZ=world.VZ, MX=world.MX, MY=world.MY;
  const aff = new Set();
  const add = (x,y,z)=>{ if (x<0||y<0||z<0||x>=VX||y>=VY||z>=VZ) return;
    aff.add((x>>3) + MX*((y>>3) + MY*(z>>3))); };
  for (const [x,y,z] of list){
    if (x<0||y<0||z<0||x>=VX||y>=VY||z>=VZ) continue;
    const k=vkey(x,y,z);
    const e = remove ? {s:0} : {s:1,mat:m.mat,r:m.r,g:m.g,b:m.b};
    EDITS.set(k,e);
    const [ax,ay,az]=liveVoxelToAuthoring(x,y,z);
    ABS_EDITS.set(absEditKey(ax,ay,az), editClone(e));
    // the voxel's own chunk + its 6 face-neighbours (their culling may change)
    add(x,y,z);
    add(x+1,y,z); add(x-1,y,z);
    add(x,y+1,z); add(x,y-1,z);
    add(x,y,z+1); add(x,y,z-1);
  }
  return aff;
}

// grow the voxel storage (CPU array + GPU buffer) only when the append cursor
// would overflow.  Pre-allocated headroom (see init) keeps this from ever
// firing during normal building.
function ensureVoxCapacity(needVox){
  const needU32 = needVox*VOX_U32;
  if (needU32 <= world.voxData.length) return true;
  let cap = world.voxData.length || (VOX_U32);
  while (cap < needU32) cap *= 2;
  const maxU32 = Math.floor((device.limits.maxStorageBufferBindingSize||134217728)/4);
  if (cap > maxU32) cap = maxU32;
  if (cap < needU32){ console.warn("voxel buffer full — edit skipped"); return false; }
  const nv = new Uint32Array(cap); nv.set(world.voxData); world.voxData = nv;
  buf.voxels.destroy?.();
  buf.voxels = device.createBuffer({size:world.voxData.byteLength,
    usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(buf.voxels, 0, world.voxData, 0, world.storedVox*VOX_U32);
  buf._voxCap = world.voxData.length;
  rebindWorldBuffers();          // draw/light reference buf.voxels
  return true;
}

// rebuild ONLY the given chunks, in place, and stream the touched regions.
function rebuildChunks(cells){
  if (!cells || !cells.size) return;
  const MX=world.MX, MY=world.MY, MZ=world.MZ;
  const sAt = solidAtEdited, smp = sampleEdited;
  const chunkData = world.chunkData;
  const occ = _occ, scr = _scratch;

  for (const cell of cells){
    if (cell<0 || cell>=world.cells) continue;
    const cx = cell % MX, cy = ((cell/MX)|0)%MY, cz = (cell/(MX*MY))|0;
    const ox=cx*8, oy=cy*8, oz=cz*8;
    const cbase = cell*CHUNK_U32;
    const oldCnt = chunkData[cbase+3];          // visible voxels currently stored here
    const oldIdx = world.mapVoxIdx[cell];       // its slot in the voxel buffer
    const wasNonEmpty = oldCnt > 0;

    // --- occupancy ---
    let any=false;
    for (let z=0;z<8;z++) for (let y=0;y<8;y++) for (let x=0;x<8;x++){
      const s = sAt(ox+x, oy+y, oz+z) ? 1 : 0;
      occ[x + 8*(y + 8*z)] = s; if (s) any=true;
    }

    chunkData[cbase+0]=cx; chunkData[cbase+1]=cy; chunkData[cbase+2]=cz;

    if (!any){
      // chunk emptied — clear its record and free its GPU presence
      for (let i=3;i<CHUNK_U32;i++) chunkData[cbase+i]=0;
      world.mapFlags[cell]=0; world.mapVoxIdx[cell]=0;
      device.queue.writeBuffer(buf.chunks,   cbase*4, chunkData.subarray(cbase, cbase+CHUNK_U32));
      writeU1(buf.mapFlags,  cell*4, 0);
      writeU1(buf.mapVoxIdx, cell*4, 0);
      writeU1(buf.sampleCnt, cell*4, 0);
      if (wasNonEmpty) world.storedChunks--;
      continue;
    }

    // --- single pass: build bitmask + prefix counts + packed voxels (scratch) ---
    for (let i=4;i<CHUNK_U32;i++) chunkData[cbase+i]=0;
    let m=0;
    for (let z=0;z<8;z++) for (let y=0;y<8;y++) for (let x=0;x<8;x++){
      const index = x + 8*(y + 8*z);
      if ((index&31)===0 && index!==0 && ((index>>5)&3)===0)
        chunkData[cbase+4+((index>>7)-1)] = m;     // running prefix count @128-bit groups
      if (!occ[index]) continue;
      const wx=ox+x, wy=oy+y, wz=oz+z;
      const visible =
        !sAt(wx+1,wy,wz)||!sAt(wx-1,wy,wz)||
        !sAt(wx,wy+1,wz)||!sAt(wx,wy-1,wz)||
        !sAt(wx,wy,wz+1)||!sAt(wx,wy,wz-1);
      if (!visible) continue;
      const s = smp(wx,wy,wz);
      let nx=0,ny=0,nz=0;
      if(!sAt(wx+1,wy,wz))nx+=1; if(!sAt(wx-1,wy,wz))nx-=1;
      if(!sAt(wx,wy+1,wz))ny+=1; if(!sAt(wx,wy-1,wz))ny-=1;
      if(!sAt(wx,wy,wz+1))nz+=1; if(!sAt(wx,wy,wz-1))nz-=1;
      let nl=Math.hypot(nx,ny,nz); if(nl<1e-5){nx=0;ny=1;nz=0;nl=1;}
      nx/=nl;ny/=nl;nz/=nl;
      const bnx=Math.max(0,Math.min(255,(((nx*255)|0)+255)>>1));
      const bny=Math.max(0,Math.min(255,(((ny*255)|0)+255)>>1));
      const bnz=Math.max(0,Math.min(255,(((nz*255)|0)+255)>>1));
      chunkData[cbase+7+(index>>5)] |= (1 << (index & 31)) >>> 0;
      const lr = Math.pow(s.r/255, CFG_GAMMA)*255;
      const lg = Math.pow(s.g/255, CFG_GAMMA)*255;
      const lb = Math.pow(s.b/255, CFG_GAMMA)*255;
      const vb = m*VOX_U32;
      scr[vb+0] = (((s.mat&255)<<24)|((bnx&255)<<16)|((bny&255)<<8)|(bnz&255))>>>0;
      scr[vb+1] = ((((lr|0)&255)<<24)|(((lg|0)&255)<<16)|(((lb|0)&255)<<8))>>>0;
      scr[vb+2] = 0; scr[vb+3] = 0;
      m++;
    }
    chunkData[cbase+3] = m;

    // --- choose voxel slot: reuse in place if it still fits, else append ---
    let vCursor;
    if (oldIdx > 0 && oldCnt >= m){
      vCursor = oldIdx;                       // reuse — new voxels fit the old slot
    } else {
      if (!ensureVoxCapacity(world.storedVox + m)) continue;
      vCursor = world.storedVox;              // append at the end of the buffer
      world.storedVox += m;
      world.mapVoxIdx[cell] = vCursor;
    }

    // copy scratch → world.voxData (fresh ref: ensureVoxCapacity may have grown it)
    world.voxData.set(scr.subarray(0, m*VOX_U32), vCursor*VOX_U32);
    world.mapFlags[cell] = 2;
    if (!wasNonEmpty) world.storedChunks++;

    // --- stream just this chunk's regions ---
    device.queue.writeBuffer(buf.chunks,    cbase*4,          chunkData.subarray(cbase, cbase+CHUNK_U32));
    device.queue.writeBuffer(buf.voxels,    vCursor*VOX_U32*4, world.voxData.subarray(vCursor*VOX_U32, (vCursor+m)*VOX_U32));
    writeU1(buf.mapVoxIdx, cell*4, vCursor);
    writeU1(buf.mapFlags,  cell*4, 2);
    writeU1(buf.sampleCnt, cell*4, 0);   // re-light this chunk only
  }
}

// instant commit — synchronous, only touches the affected chunks
function commitEdits(affected){
  if (!affected || !affected.size){ return; }
  rebuildChunks(affected);
  updateStats();
}

/* ===========================================================================
   S A V E   /   L O A D   —   tiny JSON world files
   The whole world is reproducible from two things: the SEED (terrain is pure
   deterministic math) and the SPARSE EDIT LIST (only the voxels you changed).
   So a save file is just { version, seed, edits:[...] } — a sprawling custom
   landscape is a few KB of text, never the millions of generated voxels.
   =========================================================================== */
const SAVE_VERSION = 1;

// pack the live EDITS map → compact JSON-friendly object
function serializeWorld(){
  const edits = [];
  for (const [key, e] of ABS_EDITS){
    const [x,y,z] = key.split(',').map(Number);
    if (e.s === 1) edits.push([x, y, z, e.mat, e.r, e.g, e.b]);  // placed
    else           edits.push([x, y, z]);                        // removed (len 3)
  }
  return { version: SAVE_VERSION, seed: worldSeed >>> 0,
           dims: [world.gen?.VX || world.VX, world.VY, world.gen?.VZ || world.VZ],
           player: {
             pos: livePosToSavePos(absoluteCamPos()),
             yaw: +cam.yaw.toFixed(6),
             pitch: +cam.pitch.toFixed(6)
           },
           spawn: spawnPoint ? { pos: livePosToSavePos(spawnPoint) } : null,
           edits };
}

function saveHasNeighborEdits(save){
  if (!save?.edits?.length) return false;
  const d=save.dims || [CFG.MAP[0]*8, CFG.MAP[1]*8, CFG.MAP[2]*8];
  const vx=d[0], vz=d[2];
  // Legacy saves made while streaming was active stored expanded dimensions.
  if (vx > CFG.MAP[0]*8 || vz > CFG.MAP[2]*8) return true;
  for (const row of save.edits){
    const x=row[0], z=row[2];
    if (x<0 || z<0 || x>=vx || z>=vz) return true;
  }
  return false;
}

function expandWorldForLoadedNeighborEdits(){
  if (!CFG.STREAM_TEST || !world || world.streamBase) return;
  stopStreamWorkers();
  const center=world;
  world=makeExpandedWorldFromCenter(center); dims=world;
  // No current EDITS are expected right after regenerate(), but this keeps the
  // function safe if it is reused elsewhere.
  remapCenterEditsToExpanded(center, world);
  uploadWorld(); resetLighting();
  const R=CFG.STREAM_RADIUS;
  cam.pos[0]+=center.MX*R; cam.pos[2]+=center.MZ*R;
  shiftSpawnPoint(center.MX*R, center.MZ*R);
  setTimeout(startNeighborStreaming, 80);
}

// download the current world as a .json file
function exportWorld(){
  if (!world){ return; }
  const data = serializeWorld();
  const text = JSON.stringify(data);
  const blob = new Blob([text], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `doon-world-${data.seed}-${data.edits.length}edits.json`;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  flashBanner(`SAVED · seed ${data.seed} · ${data.edits.length} edits`);
}

// rebuild EDITS from a parsed save object, then re-pack every affected chunk
function applyLoadedEdits(save){
  EDITS.clear(); colMaxEdit = null;
  ABS_EDITS.clear();
  const MX = world.MX, MY = world.MY;
  const legacyExpanded = save.dims && (save.dims[0] > (world.gen?.VX || CFG.MAP[0]*8) || save.dims[2] > (world.gen?.VZ || CFG.MAP[2]*8));
  for (const row of (save.edits || [])){
    let [x, y, z] = row;
    // Current saves store authoring coords directly. Older expanded saves stored
    // live expanded-window coords; fold those back to authoring coords once.
    if (legacyExpanded){
      x -= Math.floor((save.dims[0]-(world.gen?.VX || CFG.MAP[0]*8))*0.5);
      z -= Math.floor((save.dims[2]-(world.gen?.VZ || CFG.MAP[2]*8))*0.5);
    }
    if (!Number.isFinite(x)||!Number.isFinite(y)||!Number.isFinite(z)) continue;
    if (row.length >= 7) ABS_EDITS.set(absEditKey(x,y,z), { s:1, mat:row[3], r:row[4], g:row[5], b:row[6] });
    else                 ABS_EDITS.set(absEditKey(x,y,z), { s:0 });
  }
  syncLiveEditsFromPersistent();
  const aff = new Set();
  const touch = (x,y,z)=>{ if(x<0||y<0||z<0||x>=world.VX||y>=world.VY||z>=world.VZ) return;
    aff.add((x>>3)+MX*((y>>3)+MY*(z>>3))); };
  for (const [k] of EDITS){
    const x = k % world.VX;
    const y = ((k / world.VX) | 0) % world.VY;
    const z = (k / (world.VX * world.VY)) | 0;
    touch(x,y,z);
    touch(x+1,y,z); touch(x-1,y,z);
    touch(x,y+1,z); touch(x,y-1,z);
    touch(x,y,z+1); touch(x,y,z-1);
  }
  if (aff.size) rebuildChunks(aff);
  updateStats();
}

function restoreSavedLocation(save){
  const playerPos = savePosToLocalCamPos(save?.player?.pos);
  if (playerPos){
    cam.pos[0]=Math.max(0.2,Math.min(world.MX-0.2,playerPos[0]));
    cam.pos[1]=Math.max(0.2,Math.min(world.MY-0.2,playerPos[1]));
    cam.pos[2]=Math.max(0.2,Math.min(world.MZ-0.2,playerPos[2]));
    if (Number.isFinite(save.player.yaw)) cam.yaw = save.player.yaw;
    if (Number.isFinite(save.player.pitch)) cam.pitch = Math.max(-1.5, Math.min(1.5, save.player.pitch));
  }
  const spawnPos = savePosToAbsolutePos(save?.spawn?.pos);
  spawnPoint = spawnPos || absoluteCamPos();
  updateStreamTarget();
  updateRadar();
}

// load from a parsed JSON object: set seed, regenerate terrain, replay edits
async function loadWorld(save){
  if (!save || typeof save.seed !== "number"){ flashBanner("⚠ BAD SAVE FILE", true); return; }
  if (save.version !== SAVE_VERSION)
    console.warn("save version mismatch:", save.version, "expected", SAVE_VERSION);
  document.exitPointerLock?.();
  await regenerate(save.seed >>> 0);          // deterministic terrain from the seed
  if (saveHasNeighborEdits(save)) expandWorldForLoadedNeighborEdits();
  setStreamOriginForSavedPosition(save?.player?.pos);
  applyLoadedEdits(save);                      // overlay the sparse edits
  restoreSavedLocation(save);
  $("seed") && ($("seed").value = worldSeed);
  flashBanner(`LOADED · seed ${worldSeed} · location restored`);
}

// open a file picker and load the chosen .json
function importWorldFile(){
  const inp = document.createElement("input");
  inp.type = "file"; inp.accept = "application/json,.json";
  inp.addEventListener("change", () => {
    const f = inp.files && inp.files[0]; if (!f) return;
    const rd = new FileReader();
    rd.onload = () => {
      try { loadWorld(JSON.parse(rd.result)); }
      catch (err){ console.error(err); flashBanner("⚠ COULD NOT PARSE FILE", true); }
    };
    rd.readAsText(f);
  });
  inp.click();
}

// small transient HUD message reusing the focus-banner styling
let _bannerT = 0;
function flashBanner(msg, isErr){
  const b = $("focus-banner"); if (!b) return;
  clearTimeout(_bannerT);
  b.classList.toggle("err", !!isErr);
  b.textContent = msg;
  b.style.display = "flex";
  _bannerT = setTimeout(() => {
    b.classList.remove("err");
    b.textContent = "▶  CLICK TO LOOK  ·  ESC TO RELEASE";
    // hide again only if we're not waiting for a pointer-lock click
    if (document.pointerLockElement === $("gpu") || BUILD.on) b.style.display = "none";
  }, 2400);
}

// ---- click handling per tool (SketchUp-style multi-click) ----
function buildClick(){
  const rc=raycastVoxel(); if(!rc) return;
  BUILD._lastRC = rc;
  const T=BUILD.tool;
  if (T==='block'){ commitEdits(applyVoxels([rc.place],false)); return; }
  if (T==='remove'){ commitEdits(applyVoxels([rc.hit],true)); return; }
  if (T==='circle'){
    if (!BUILD.circle){
      BUILD.circle={step:1,cx:rc.place[0]+0.5,cz:rc.place[2]+0.5,baseY:rc.place[1],radius:null,thick:null,height:null};
      refreshGhosts(); return;
    }
    const C=BUILD.circle, cur=circleCursorXZ(rc.place);
    if (C.step===1){ C.radius=Math.max(1,Math.hypot(cur[0]-C.cx,cur[1]-C.cz)); C.step=2; refreshGhosts(); return; }
    if (C.step===2){ const d=Math.hypot(cur[0]-C.cx,cur[1]-C.cz); C.thick=Math.max(1,Math.min(C.radius,C.radius-d)); C.step=3; refreshGhosts(); return; }
    const h=Math.max(1,Math.round(Math.hypot(cur[0]-C.cx,cur[1]-C.cz))); C.height=h;
    const a=applyVoxels(genCircleTower(circleParams(rc.place)),false); BUILD.circle=null; commitEdits(a); return;
  }
  if (T==='square'){
    if (!BUILD.square){
      BUILD.square={step:1,x0:rc.place[0],z0:rc.place[2],x1:null,z1:null,baseY:rc.place[1],thick:null,height:null};
      refreshGhosts(); return;
    }
    const S=BUILD.square, cur=squareCursorXZ(rc.place);
    const ox=Math.round(cur[0]-0.5), oz=Math.round(cur[1]-0.5);
    if (S.step===1){ S.x1=ox; S.z1=oz; S.step=2; refreshGhosts(); return; }
    if (S.step===2){
      const p=squareParams(rc.place);
      S.thick=p?p.thick:1; S.step=3; refreshGhosts(); return;
    }
    const p=squareParams(rc.place);
    if (p){ S.height=p.height; const a=applyVoxels(genSquareRoom(squareParams(rc.place)),false); BUILD.square=null; commitEdits(a); }
    return;
  }
  if (T==='arc'){
    if (!BUILD.anchor){ BUILD.anchor=rc.place; }
    else if (!BUILD.mid){ BUILD.mid=rc.place; }
    else { const a=applyVoxels(genGhosts(),false); BUILD.anchor=null; BUILD.mid=null; commitEdits(a); }
    return;
  }
  // two-click tools: wall / rect / disc
  if (!BUILD.anchor){ BUILD.anchor=rc.place; }
  else { const a=applyVoxels(genGhosts(),false); BUILD.anchor=null; commitEdits(a); }
}
function buildCancel(){ BUILD.anchor=null; BUILD.mid=null; BUILD.circle=null; BUILD.square=null; refreshGhosts(); }

function setBuildMode(on){
  BUILD.on=on; BUILD.anchor=null; BUILD.mid=null; BUILD.circle=null; BUILD.square=null;
  document.exitPointerLock?.();
  $("buildbar").style.display = on?'block':'none';
  $("bd-badge").style.display = on?'block':'none';
  $("focus-banner").style.display = on?'none':'';
  const c=$("gpu"); c.classList.toggle('build', on);
  $("panel").style.display = on?'none':'';
  refreshGhosts();
}
function selectTool(t){ BUILD.tool=t; BUILD.anchor=null; BUILD.mid=null; BUILD.circle=null; BUILD.square=null;
  document.querySelectorAll('#buildbar .tool').forEach(el=>el.classList.toggle('active', el.dataset.tool===t));
  $("row-h").style.display=(t!=='block'&&t!=='remove')?'':'none';
  $("row-t").style.display=(t==='wall'||t==='arc'||t==='circle'||t==='square')?'':'none';
  $("row-r").style.display=(t==='disc'||t==='circle')?'':'none';
  $("bd-tool").textContent=t.toUpperCase();
  refreshGhosts();
}
function selectMat(i){ BUILD.matIndex=i;
  document.querySelectorAll('#buildbar .swatch').forEach((el,idx)=>el.classList.toggle('active', idx===i));
  $("bd-mat").textContent=BUILD_MATS[i].name;
}

/* ---- ghost preview rendering (translucent instanced cubes over the scene) ---- */
function mat4Perspective(fovy,aspect,near,far){ const f=1/Math.tan(fovy/2), nf=1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,far*nf,-1, 0,0,near*far*nf,0]); }
function mat4LookAt(e,c,u){
  let zx=e[0]-c[0],zy=e[1]-c[1],zz=e[2]-c[2]; let zl=Math.hypot(zx,zy,zz)||1; zx/=zl;zy/=zl;zz/=zl;
  let xx=u[1]*zz-u[2]*zy, xy=u[2]*zx-u[0]*zz, xz=u[0]*zy-u[1]*zx; let xl=Math.hypot(xx,xy,xz)||1; xx/=xl;xy/=xl;xz/=xl;
  let yx=zy*xz-zz*xy, yy=zz*xx-zx*xz, yz=zx*xy-zy*xx;
  return new Float32Array([xx,yx,zx,0, xy,yy,zy,0, xz,yz,zz,0,
    -(xx*e[0]+xy*e[1]+xz*e[2]), -(yx*e[0]+yy*e[1]+yz*e[2]), -(zx*e[0]+zy*e[1]+zz*e[2]), 1]); }
function mat4Mul(a,b){ const o=new Float32Array(16);
  for(let c=0;c<4;c++)for(let r=0;r<4;r++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+r]*b[c*4+k];o[c*4+r]=s;} return o; }

const SH_GHOST = /* wgsl */`
struct GU { vp:mat4x4<f32>, tint:vec4<f32> };
@group(0) @binding(0) var<uniform> G:GU;
struct VO { @builtin(position) pos:vec4<f32>, @location(0) nrm:vec3<f32> };
@vertex fn vs(@location(0) corner:vec3<f32>, @location(1) nor:vec3<f32>, @location(2) inst:vec4<f32>) -> VO {
  var o:VO; let wp=(inst.xyz+corner)*0.125; o.pos = G.vp*vec4<f32>(wp,1.0); o.nrm=nor; return o;
}
@fragment fn fs(i:VO) -> @location(0) vec4<f32> {
  let l = clamp(dot(normalize(i.nrm), normalize(vec3<f32>(0.4,0.95,0.32)))*0.5+0.5, 0.0, 1.0);
  return vec4<f32>(G.tint.rgb*(0.55+0.5*l), G.tint.a);
}`;

function setupGhost(){
  // unit cube: 36 verts (corner xyz + face normal)
  const F=[
    [[0,0,0],[1,0,0],[1,1,0],[0,0,0],[1,1,0],[0,1,0],[0,0,-1]], // -z
    [[0,0,1],[1,1,1],[1,0,1],[0,0,1],[0,1,1],[1,1,1],[0,0,1]],  // +z
    [[0,0,0],[0,1,0],[0,1,1],[0,0,0],[0,1,1],[0,0,1],[-1,0,0]], // -x
    [[1,0,0],[1,1,1],[1,1,0],[1,0,0],[1,0,1],[1,1,1],[1,0,0]],  // +x
    [[0,0,0],[1,0,1],[1,0,0],[0,0,0],[0,0,1],[1,0,1],[0,-1,0]], // -y
    [[0,1,0],[1,1,0],[1,1,1],[0,1,0],[1,1,1],[0,1,1],[0,1,0]],  // +y
  ];
  const data=[]; for(const f of F){ const n=f[6]; for(let i=0;i<6;i++){ const v=f[i]; data.push(v[0],v[1],v[2], n[0],n[1],n[2]); } }
  const arr=new Float32Array(data);
  ghostCubeBuf=device.createBuffer({size:arr.byteLength, usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
  device.queue.writeBuffer(ghostCubeBuf,0,arr);
  buf.ghostInst=device.createBuffer({size:GHOST_MAX*16, usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});
  ghostUni=device.createBuffer({size:80, usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST}); // mat4(64)+vec4(16)
  const m=device.createShaderModule({code:SH_GHOST});
  pipe.ghost=device.createRenderPipeline({layout:"auto",
    vertex:{module:m,entryPoint:"vs",buffers:[
      {arrayStride:24,stepMode:"vertex",attributes:[{shaderLocation:0,offset:0,format:"float32x3"},{shaderLocation:1,offset:12,format:"float32x3"}]},
      {arrayStride:16,stepMode:"instance",attributes:[{shaderLocation:2,offset:0,format:"float32x4"}]} ]},
    fragment:{module:m,entryPoint:"fs",targets:[{format:fmt, blend:{
      color:{srcFactor:"src-alpha",dstFactor:"one-minus-src-alpha",operation:"add"},
      alpha:{srcFactor:"one",dstFactor:"one-minus-src-alpha",operation:"add"}}}]},
    primitive:{topology:"triangle-list",cullMode:"none"},
    depthStencil:{format:"depth24plus",depthWriteEnabled:true,depthCompare:"less"}});
  bind.ghost=device.createBindGroup({layout:pipe.ghost.getBindGroupLayout(0),entries:[{binding:0,resource:{buffer:ghostUni}}]});
}
let ghostCount=0;
function uploadGhosts(){
  const g=BUILD.ghosts; const n=Math.min(g.length, GHOST_MAX);
  const a=new Float32Array(n*4);
  for (let i=0;i<n;i++){ a[i*4]=g[i][0]; a[i*4+1]=g[i][1]; a[i*4+2]=g[i][2]; a[i*4+3]=0; }
  if (buf.ghostInst && n>0) device.queue.writeBuffer(buf.ghostInst,0,a);
  ghostCount=n;
}

async function init(){
  if (!navigator.gpu) throw new Error("WebGPU not available. Use Chrome/Edge 113+ or Firefox Nightly, over http(s)/localhost.");
  const adapter = await navigator.gpu.requestAdapter({powerPreference:"high-performance"});
  if (!adapter) throw new Error("No GPU adapter. Enable hardware acceleration / WebGPU flags.");
  const maxStorage = adapter.limits.maxStorageBufferBindingSize;
  device = await adapter.requestDevice({
    requiredLimits:{ maxStorageBufferBindingSize: Math.min(maxStorage, 1<<30) }
  });
  device.lost.then(i=>showError("GPU device lost: "+i.message));

  const canvas = $("gpu");
  ctx = canvas.getContext("webgpu");
  fmt = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({device, format:fmt, alphaMode:"opaque"});

  // ---- world ----
  $("ltext").textContent = "Generating world · seed "+worldSeed+" …";
  $("loading").style.display="flex";
  await new Promise(r=>setTimeout(r,30));
  world = await buildWorld(progressCb, worldSeed);
  dims = world;
  const mats = buildMaterials();

  // ---- buffers ----
  const S = GPUBufferUsage.STORAGE, CD = GPUBufferUsage.COPY_DST, CS = GPUBufferUsage.COPY_SRC;
  const mk=(data,usage)=>makeBuf(data,usage);

  buf.mapFlags  = mk(world.mapFlags,  S|CD);
  buf.mapVoxIdx = mk(world.mapVoxIdx, S|CD);
  buf.chunks    = mk(world.chunkData, S|CD);
  buf.voxels    = mk(world.voxData,   S|CD);
  buf._mapBytes = Math.max(16, world.mapFlags.byteLength);
  buf._idxBytes = Math.max(16, world.mapVoxIdx.byteLength);
  buf._chunkBytes = Math.max(16, world.chunkData.byteLength);
  buf._voxCap   = world.voxData.length;

  // Pre-allocate generous voxel headroom up front so that live edits can append
  // new chunks into the buffer without ever recreating it mid-build (which would
  // cost a full re-upload + rebind).  Only the used portion is transferred now.
  {
    const maxU32   = Math.floor((device.limits.maxStorageBufferBindingSize||134217728)/4);
    const targetVox= Math.min(4194304, Math.floor(maxU32/VOX_U32));   // up to ~4M voxels
    const targetU32= targetVox*VOX_U32;
    if (world.voxData.length < targetU32){
      const nv=new Uint32Array(targetU32); nv.set(world.voxData); world.voxData=nv;
      buf.voxels.destroy?.();
      buf.voxels = device.createBuffer({size:world.voxData.byteLength, usage:S|CD});
      device.queue.writeBuffer(buf.voxels, 0, world.voxData, 0, world.storedVox*VOX_U32);
      buf._voxCap = world.voxData.length;
    }
  }
  buf.materials = mk(mats,            S|CD);
  buf.sampleCnt = device.createBuffer({size:world.cells*4, usage:S|CD}); // zeroed
  buf._sampleBytes = Math.max(16, world.cells*4);
  device.queue.writeBuffer(buf.sampleCnt,0,zeroU32(world.cells));
  buf.requests  = device.createBuffer({size:CFG.REQ_CAP*4, usage:S});
  buf.counter   = device.createBuffer({size:4, usage:S|CD|CS});
  buf.args      = device.createBuffer({size:12, usage:S|GPUBufferUsage.INDIRECT});
  buf.uniform   = device.createBuffer({size:uniBytes, usage:GPUBufferUsage.UNIFORM|CD});
  buf.uni2      = device.createBuffer({size:32, usage:GPUBufferUsage.UNIFORM|CD}); // build: mapSize+p0
  buf.capU      = device.createBuffer({size:4, usage:GPUBufferUsage.UNIFORM|CD});
  device.queue.writeBuffer(buf.capU,0,new Uint32Array([CFG.REQ_CAP]));

  uni = new ArrayBuffer(uniBytes);

  // ---- pipelines ----
  const mod=(code)=>device.createShaderModule({code});
  const sharedDraw  = mod(SH_SHARED+SH_DRAW);
  const sharedLight = mod(SH_SHARED+SH_LIGHT);

  pipe.draw  = device.createComputePipeline({layout:"auto",compute:{module:sharedDraw, entryPoint:"main"}});
  pipe.light = device.createComputePipeline({layout:"auto",compute:{module:sharedLight,entryPoint:"main"}});
  pipe.build = device.createComputePipeline({layout:"auto",compute:{module:mod(SH_BUILD),entryPoint:"main"}});
  pipe.args  = device.createComputePipeline({layout:"auto",compute:{module:mod(SH_ARGS), entryPoint:"main"}});
  const blitMod = mod(SH_BLIT);
  pipe.blit  = device.createRenderPipeline({layout:"auto",
    vertex:{module:blitMod,entryPoint:"vs"}, fragment:{module:blitMod,entryPoint:"fs",targets:[{format:fmt}]},
    primitive:{topology:"triangle-list"}});

  sampler = device.createSampler({magFilter:"linear",minFilter:"linear"});

  // ---- ghost-block preview pipeline (translucent instanced cubes) ----
  setupGhost();

  // build/args bind groups (static)
  bind.build = device.createBindGroup({layout:pipe.build.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:buf.mapFlags}},{binding:1,resource:{buffer:buf.chunks}},
    {binding:2,resource:{buffer:buf.requests}},{binding:3,resource:{buffer:buf.counter}},
    {binding:4,resource:{buffer:buf.uni2}}]});
  bind.args = device.createBindGroup({layout:pipe.args.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:buf.counter}},{binding:1,resource:{buffer:buf.args}},
    {binding:2,resource:{buffer:buf.capU}}]});

  // place camera in the flat clearing, looking up at the keep
  placeCamera();

  resize();
  window.addEventListener("resize",resize);
  setupInput(canvas);
  $("loading").style.display="none";

  // stats
  updateStats();

  setTimeout(startNeighborStreaming, 80);

  requestAnimationFrame(loop);
}

function resize(){
  const c=$("gpu"); const dpr=Math.min(devicePixelRatio||1, 2);
  c.width = Math.max(2, Math.floor(innerWidth*dpr*1));   // canvas backing
  c.height= Math.max(2, Math.floor(innerHeight*dpr*1));
  const rw = Math.max(2, Math.floor(c.width*CFG.RES_SCALE));
  const rh = Math.max(2, Math.floor(c.height*CFG.RES_SCALE));
  if (outTex) outTex.destroy?.();
  outTex = device.createTexture({size:[rw,rh],format:"rgba8unorm",
    usage:GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.TEXTURE_BINDING});
  if (ghostDepth) ghostDepth.destroy?.();
  ghostDepth = device.createTexture({size:[c.width,c.height],format:"depth24plus",
    usage:GPUTextureUsage.RENDER_ATTACHMENT});
  dims.rw=rw; dims.rh=rh;

  bind.draw = device.createBindGroup({layout:pipe.draw.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:buf.mapFlags}},{binding:1,resource:{buffer:buf.mapVoxIdx}},
    {binding:2,resource:{buffer:buf.chunks}},{binding:3,resource:{buffer:buf.voxels}},
    {binding:4,resource:{buffer:buf.materials}},{binding:5,resource:{buffer:buf.uniform}},
    {binding:6,resource:outTex.createView()}]});
  bind.light = device.createBindGroup({layout:pipe.light.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:buf.mapFlags}},{binding:1,resource:{buffer:buf.mapVoxIdx}},
    {binding:2,resource:{buffer:buf.chunks}},{binding:3,resource:{buffer:buf.voxels}},
    {binding:4,resource:{buffer:buf.materials}},{binding:5,resource:{buffer:buf.uniform}},
    {binding:6,resource:{buffer:buf.requests}},{binding:7,resource:{buffer:buf.sampleCnt}}]});
  bind.blit = device.createBindGroup({layout:pipe.blit.getBindGroupLayout(0),entries:[
    {binding:0,resource:outTex.createView()},{binding:1,resource:sampler}]});
}

// Re-create the draw/light bind groups against the current world buffers
// (called after the voxel buffer is grown).  Reuses the current outTex so it's
// far cheaper than a full resize().
function rebindWorldBuffers(){
  if (!pipe.draw || !outTex) return;
  bind.draw = device.createBindGroup({layout:pipe.draw.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:buf.mapFlags}},{binding:1,resource:{buffer:buf.mapVoxIdx}},
    {binding:2,resource:{buffer:buf.chunks}},{binding:3,resource:{buffer:buf.voxels}},
    {binding:4,resource:{buffer:buf.materials}},{binding:5,resource:{buffer:buf.uniform}},
    {binding:6,resource:outTex.createView()}]});
  bind.light = device.createBindGroup({layout:pipe.light.getBindGroupLayout(0),entries:[
    {binding:0,resource:{buffer:buf.mapFlags}},{binding:1,resource:{buffer:buf.mapVoxIdx}},
    {binding:2,resource:{buffer:buf.chunks}},{binding:3,resource:{buffer:buf.voxels}},
    {binding:4,resource:{buffer:buf.materials}},{binding:5,resource:{buffer:buf.uniform}},
    {binding:6,resource:{buffer:buf.requests}},{binding:7,resource:{buffer:buf.sampleCnt}}]});
}

function camBasis(){
  const cp=Math.cos(cam.pitch),sp=Math.sin(cam.pitch),cy=Math.cos(cam.yaw),sy=Math.sin(cam.yaw);
  const fwd=[cp*sy, sp, cp*cy];
  const wup=[0,1,0];
  let r=[fwd[1]*wup[2]-fwd[2]*wup[1], fwd[2]*wup[0]-fwd[0]*wup[2], fwd[0]*wup[1]-fwd[1]*wup[0]];
  let rl=Math.hypot(...r); r=r.map(v=>v/rl);
  let up=[r[1]*fwd[2]-r[2]*fwd[1], r[2]*fwd[0]-r[0]*fwd[2], r[0]*fwd[1]-r[1]*fwd[0]];
  return {fwd,r,up};
}

function writeUniforms(time){
  const f=new Float32Array(uni), u=new Uint32Array(uni);
  const {fwd,r,up}=camBasis();
  const aspect=dims.rw/dims.rh;
  const tanHalf=Math.tan(cam.fov*Math.PI/360);
  // camRight/up scaled so ray = fwd + ndc.x*right + ndc.y*up
  const Rx=aspect*tanHalf, Uy=tanHalf;
  let o=0;
  const v3=(a,b,c,d=0)=>{f[o]=a;f[o+1]=b;f[o+2]=c;f[o+3]=d;o+=4;};
  v3(cam.pos[0],cam.pos[1],cam.pos[2]);
  v3(fwd[0],fwd[1],fwd[2]);
  v3(r[0]*Rx,r[1]*Rx,r[2]*Rx);
  v3(up[0]*Uy,up[1]*Uy,up[2]*Uy);
  // sun
  const sd=[Math.cos(sunAngle)*0.5, 0.85, Math.sin(sunAngle)*0.5];
  const sl=Math.hypot(...sd); const sdn=sd.map(v=>v/sl);
  v3(sdn[0],sdn[1],sdn[2]);
  v3(1.55,1.4,1.15);               // sunStrength
  v3(0.06,0.07,0.10);              // ambient
  v3(0.55,0.62,0.78);              // skyBot
  v3(0.16,0.26,0.52);              // skyTop
  // mapSize (u32)
  u[o]=world.MX;u[o+1]=world.MY;u[o+2]=world.MZ;u[o+3]=0;o+=4;
  // p0: viewMode, frameNum, split, reqCap
  u[o]=viewMode;u[o+1]=frameNum;u[o+2]=CFG.LIGHT_SPLIT;u[o+3]=CFG.REQ_CAP;o+=4;
  // p1: numDiffuse, maxDiffuse, diffuseBounce, specBounce
  u[o]=CFG.DIFFUSE_SAMPLES;u[o+1]=CFG.MAX_DIFFUSE_SAMPLES;u[o+2]=CFG.DIFFUSE_BOUNCES;u[o+3]=CFG.SPEC_BOUNCES;o+=4;
  // p2: time, shadowSoft, resX, resY
  f[o]=time;f[o+1]=CFG.SHADOW_SOFT;f[o+2]=dims.rw;f[o+3]=dims.rh;o+=4;
  device.queue.writeBuffer(buf.uniform,0,uni);

  // build-pass uniform (mapSize + p0)
  const b2=new Uint32Array(8);
  b2[0]=world.MX;b2[1]=world.MY;b2[2]=world.MZ;b2[3]=0;
  b2[4]=viewMode;b2[5]=frameNum;b2[6]=CFG.LIGHT_SPLIT;b2[7]=CFG.REQ_CAP;
  device.queue.writeBuffer(buf.uni2,0,b2);
}

let lastT=performance.now(), accT=0, frames=0, fpsV=0, msV=0;
function loop(now){
  const dt=(now-lastT)/1000; lastT=now;
  if (!busy){ update(dt, now/1000); render(); }
  // fps
  frames++; accT+=dt;
  if (accT>=0.5){ fpsV=frames/accT; msV=accT/frames*1000; frames=0; accT=0;
    $("fps").textContent=fpsV.toFixed(0); $("ms").textContent=msV.toFixed(1); }
  requestAnimationFrame(loop);
}

function update(dt, time){
  const {fwd,r}=camBasis();
  const sp=cam.speed*(keys["shift"]?3.2:1)*(dt*60);
  const fGround=[fwd[0],0,fwd[2]]; const fl=Math.hypot(...fGround)||1;
  const move=(v,s)=>{cam.pos[0]+=v[0]*s;cam.pos[1]+=v[1]*s;cam.pos[2]+=v[2]*s;};
  if(keys["w"])move([fGround[0]/fl,0,fGround[2]/fl],sp);
  if(keys["s"])move([fGround[0]/fl,0,fGround[2]/fl],-sp);
  if(keys["d"])move(r,sp);
  if(keys["a"])move(r,-sp);
  if(keys[" "])cam.pos[1]+=sp;
  if(keys["control"])cam.pos[1]-=sp;
  cam.pos[0]=Math.max(0.2,Math.min(world.MX-0.2,cam.pos[0]));
  cam.pos[1]=Math.max(0.2,Math.min(world.MY-0.2,cam.pos[1]));
  cam.pos[2]=Math.max(0.2,Math.min(world.MZ-0.2,cam.pos[2]));
  updateStreamTarget();
  updateRadar();
  if(sunSpin) sunAngle+=dt*0.25;
  frameNum=(frameNum+1)%CFG.LIGHT_SPLIT;
  writeUniforms(time);
}

function render(){
  // clear request counter
  device.queue.writeBuffer(buf.counter,0,new Uint32Array([0]));
  const enc=device.createCommandEncoder();

  // 1. draw
  {const p=enc.beginComputePass(); p.setPipeline(pipe.draw); p.setBindGroup(0,bind.draw);
   p.dispatchWorkgroups(Math.ceil(dims.rw/8),Math.ceil(dims.rh/8),1); p.end();}
  // 2. build requests
  {const p=enc.beginComputePass(); p.setPipeline(pipe.build); p.setBindGroup(0,bind.build);
   p.dispatchWorkgroups(Math.ceil(world.cells/64),1,1); p.end();}
  // 3. indirect args
  {const p=enc.beginComputePass(); p.setPipeline(pipe.args); p.setBindGroup(0,bind.args);
   p.dispatchWorkgroups(1,1,1); p.end();}
  // 4. lighting (indirect)
  {const p=enc.beginComputePass(); p.setPipeline(pipe.light); p.setBindGroup(0,bind.light);
   p.dispatchWorkgroupsIndirect(buf.args,0); p.end();}
  // 5. blit
  {const view=ctx.getCurrentTexture().createView();
   const p=enc.beginRenderPass({colorAttachments:[{view,loadOp:"clear",storeOp:"store",clearValue:{r:0,g:0,b:0,a:1}}]});
   p.setPipeline(pipe.blit); p.setBindGroup(0,bind.blit); p.draw(3); p.end();}

  // 6. ghost-block preview (translucent cubes over the scene)
  if (BUILD.on && ghostCount>0){
    const {fwd}=camBasis();
    const aspect=$("gpu").width/$("gpu").height;
    const view=mat4LookAt(cam.pos,[cam.pos[0]+fwd[0],cam.pos[1]+fwd[1],cam.pos[2]+fwd[2]],[0,1,0]);
    const proj=mat4Perspective(cam.fov*Math.PI/180, aspect, 0.02, world.MX*2.2);
    const vp=mat4Mul(proj,view);
    const tint = (BUILD.tool==='remove') ? [1.0,0.42,0.42,0.5] : [0.92,0.97,1.0,0.42];
    const ub=new Float32Array(20); ub.set(vp,0); ub.set(tint,16);
    device.queue.writeBuffer(ghostUni,0,ub);
    const cview=ctx.getCurrentTexture().createView();
    const p=enc.beginRenderPass({
      colorAttachments:[{view:cview,loadOp:"load",storeOp:"store"}],
      depthStencilAttachment:{view:ghostDepth.createView(),depthLoadOp:"clear",depthStoreOp:"store",depthClearValue:1.0}});
    p.setPipeline(pipe.ghost); p.setBindGroup(0,bind.ghost);
    p.setVertexBuffer(0,ghostCubeBuf); p.setVertexBuffer(1,buf.ghostInst);
    p.draw(36, ghostCount); p.end();
  }

  device.queue.submit([enc.finish()]);
}

function resetLighting(){
  const need = Math.max(16, world.cells*4);
  if (!buf.sampleCnt || (buf._sampleBytes||0) < need){
    buf.sampleCnt?.destroy?.();
    buf.sampleCnt = device.createBuffer({size:need, usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});
    buf._sampleBytes = need;
    rebindWorldBuffers();
  }
  device.queue.writeBuffer(buf.sampleCnt,0,zeroU32(world.cells));
}

/* ----------------------------  I N P U T  --------------------------------- */
function setupInput(canvas){
  const banner  = $('focus-banner');
  const xhair   = $('xhair');
  const escHint = $('esc-hint');

  /* ---- pointer-lock state ---- */
  function applyLockState(locked){
    canvas.classList.toggle('locked',   locked);
    canvas.classList.toggle('unlocked', !locked);
    banner .style.display = locked ? 'none'  : 'flex';
    xhair  .style.display = locked ? 'block' : 'none';
    escHint.style.display = locked ? 'block' : 'none';
  }

  document.addEventListener('pointerlockchange', () => {
    applyLockState(document.pointerLockElement === canvas);
  });

  document.addEventListener('pointerlockerror', () => {
    banner.classList.add('err');
    banner.style.display = 'flex';
    banner.textContent = '⚠  POINTER LOCK BLOCKED — click again';
    setTimeout(() => {
      banner.classList.remove('err');
      banner.textContent = '▶  CLICK TO LOOK  ·  ESC TO RELEASE';
    }, 2200);
  });

  /* ---- click canvas → lock; click again while locked is ignored ---- */
  canvas.addEventListener('click', () => {
    if (BUILD.on) return;                       // build mode uses explicit mousedown
    if (document.pointerLockElement !== canvas) {
      const p = canvas.requestPointerLock();
      if (p) p.catch(() => {});   // silence promise-rejection in modern browsers
    }
  });

  /* ---- build-mode pointer: left = place/anchor, right-drag = look ---- */
  canvas.addEventListener('contextmenu', e=>{ if(BUILD.on) e.preventDefault(); });
  canvas.addEventListener('mousedown', e=>{
    if (!BUILD.on) return;
    e.preventDefault();
    if (e.button===0) buildClick();
    else if (e.button===2) BUILD.dragLook=true;
  });
  addEventListener('mouseup', e=>{ if(e.button===2) BUILD.dragLook=false; });

  /* ---- keyboard ---- */
  addEventListener('keydown', e => {
    const k = e.key.toLowerCase();
    if (k === 'escape') { keys[k] = false; if(BUILD.on) buildCancel(); return; }
    if (e.ctrlKey && k === 'u') { e.preventDefault(); keys[k] = false; setSpawnPoint(true); return; }
    keys[k] = true;
    if (k === 'b'){ setBuildMode(!BUILD.on); return; }
    if (k >= '1' && k <= '6'){ viewMode = +k-1; $('modename').textContent = MODE_NAMES[viewMode]; }
    if (k === 'l'){ sunSpin = !sunSpin; $('sunstate').textContent = sunSpin ? 'ANIMATED' : 'FIXED'; if(sunSpin) resetLighting(); }
    if (k === 'r'){ resetLighting(); }
    if (k === 'g'){ document.exitPointerLock?.(); EDITS.clear(); ABS_EDITS.clear(); colMaxEdit=null; regenerate((Math.random()*4294967296)>>>0); }
    if (k === 'o'){ document.exitPointerLock?.(); exportWorld(); }      // O = save world to JSON
    if (k === 'i'){ document.exitPointerLock?.(); importWorldFile(); }  // I = load world from JSON
    if (k === ' ' || k === 'control' || k === 'arrowup' || k === 'arrowdown') e.preventDefault();
  });
  addEventListener('keyup', e => { keys[e.key.toLowerCase()] = false; });

  /* ---- mouse look (locked FPS) + build-mode aim/drag-look ---- */
  addEventListener('mousemove', e => {
    if (BUILD.on){
      if (BUILD.dragLook){
        cam.yaw   -= e.movementX * 0.0026;
        cam.pitch -= e.movementY * 0.0026;
        cam.pitch  = Math.max(-1.5, Math.min(1.5, cam.pitch));
      } else {
        const r=canvas.getBoundingClientRect();
        BUILD.aim.x = ((e.clientX-r.left)/r.width)*2-1;
        BUILD.aim.y = 1-((e.clientY-r.top)/r.height)*2;
      }
      refreshGhosts();
      return;
    }
    if (document.pointerLockElement !== canvas) return;
    cam.yaw   -= e.movementX * 0.0022;
    cam.pitch -= e.movementY * 0.0022;
    cam.pitch  = Math.max(-1.5, Math.min(1.5, cam.pitch));
  });

  addEventListener('wheel', e => {
    if (BUILD.on){
      const dir=-Math.sign(e.deltaY);
      if (BUILD.tool==='disc'){ BUILD.radius=Math.max(1,Math.min(64,BUILD.radius+dir)); const s=$("bd-r"); if(s){s.value=BUILD.radius; $("bd-rv").textContent=BUILD.radius;} }
      else { BUILD.height=Math.max(1,Math.min(48,BUILD.height+dir)); const s=$("bd-hh"); if(s){s.value=BUILD.height; $("bd-hv").textContent=BUILD.height;} }
      refreshGhosts(); return;
    }
    cam.fov = Math.max(40, Math.min(95, cam.fov + Math.sign(e.deltaY) * 2));
  }, { passive: true });

  // start unlocked (game is running, waiting for first click)
  applyLockState(false);
}

/* --------------------------  U T I L / B O O T  --------------------------- */
function fmtNum(n){ return n>=1e6?(n/1e6).toFixed(2)+"M":n>=1e3?(n/1e3).toFixed(1)+"k":""+n; }
function showError(m){ $("err").textContent=m; $("overlay").style.display="flex"; $("loading").style.display="none"; }

$("reroll").addEventListener("click",()=>{ $("seed").value = ((Math.random()*4294967296)>>>0); });

$("start").addEventListener("click",()=>{
  worldSeed = parseSeed($("seed").value);
  $("seed").value = worldSeed;            // echo the resolved numeric seed
  $("overlay").style.display="none";
  init().catch(e=>{ console.error(e); showError((e&&e.message)||String(e)); });
});

// prefill a random seed so first launch differs each time
$("seed").value = ((Math.random()*4294967296)>>>0);

// ---- BUILD toolbar wiring ----
(function wireBuild(){
  // material swatches
  const sw=$("swatches");
  BUILD_MATS.forEach((m,i)=>{
    const d=document.createElement('div'); d.className='swatch';
    d.style.background=`rgb(${m.r},${m.g},${m.b})`; d.title=m.name;
    d.addEventListener('click',()=>selectMat(i)); sw.appendChild(d);
  });
  // tool buttons
  document.querySelectorAll('#buildbar .tool').forEach(el=>
    el.addEventListener('click',()=>selectTool(el.dataset.tool)));
  // sliders
  const bind=(id,vid,prop,max)=>{ const s=$(id); s.addEventListener('input',()=>{
    BUILD[prop]=+s.value; $(vid).textContent=s.value; refreshGhosts(); }); };
  bind('bd-hh','bd-hv','height'); bind('bd-tt','bd-tv','thick'); bind('bd-r','bd-rv','radius');
  // defaults
  selectTool('wall'); selectMat(0);
})();