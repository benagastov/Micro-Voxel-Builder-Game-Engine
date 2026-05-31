"use strict";

export const BUILD = {
  on:false, tool:'wall', matIndex:0,
  height:4, thick:1, radius:8,
  circle:null,
  square:null,
  anchor:null, mid:null,                  // click points (voxel [x,y,z])
  aim:{x:0,y:0}, hover:null,              // cursor ndc + current target placeVoxel
  ghosts:[],                              // [x,y,z] list for preview
  dragLook:false,
};
export const BUILD_MATS = [
  {name:'Stone',       mat:0, r:150,g:150,b:150},
  {name:'Mossy Stone', mat:0, r:120,g:135,b:96 },
  {name:'Dark Stone',  mat:0, r:74, g:78, b:82 },
  {name:'White Stone', mat:0, r:226,g:224,b:214},
  {name:'Sandstone',   mat:0, r:206,g:190,b:150},
  {name:'Wood',        mat:0, r:120,g:80, b:46 },
  {name:'Plank',       mat:0, r:170,g:130,b:80 },
  {name:'Brick',       mat:0, r:150,g:74, b:58 },
  {name:'Grass',       mat:0, r:96, g:142,b:66 },
  {name:'Glass',       mat:2, r:170,g:205,b:225},
  {name:'Lamp',        mat:3, r:255,g:200,b:120},
];


// ---- tool geometry → list of voxel coords ----
export function lineXZ(x0,z0,x1,z1){            // dense march so circles/lines have no gaps
  const pts=[]; const dx=x1-x0, dz=z1-z0; const n=Math.max(1,Math.ceil(Math.hypot(dx,dz)));
  for (let i=0;i<=n;i++){ pts.push([Math.round(x0+dx*i/n), Math.round(z0+dz*i/n)]); }
  return pts;
}
export function thicken(cols,thick){
  if (thick<=1) return cols;
  const set=new Set(), out=[]; const rad=(thick-1);
  for (const [x,z] of cols) for (let dz=-rad;dz<=rad;dz++) for (let dx=-rad;dx<=rad;dx++){
    if (dx*dx+dz*dz> rad*rad+0.5) continue; const k=(x+dx)+'_'+(z+dz);
    if (!set.has(k)){ set.add(k); out.push([x+dx,z+dz]); }
  }
  return out;
}
export function raiseColumns(cols, baseY, height){
  const out=[]; const CAP=180000;
  for (const [x,z] of cols){ for (let h=0;h<height;h++){ out.push([x,baseY+h,z]); if(out.length>=CAP) return out; } }
  return out;
}
export function circleCursorXZ(P){
  const rc=BUILD._lastRC;
  return (rc&&rc.pos) ? [rc.pos[0], rc.pos[2]] : [P[0]+0.5, P[2]+0.5];
}
export function circleParams(P){
  const C=BUILD.circle; if(!C) return null;
  const cur=P?circleCursorXZ(P):[C.cx,C.cz];
  const radius=C.radius ?? Math.max(1,Math.hypot(cur[0]-C.cx, cur[1]-C.cz));
  const thick=C.thick ?? Math.max(1, radius - Math.hypot(cur[0]-C.cx, cur[1]-C.cz));
  const height=C.height ?? (C.step===3 ? Math.max(1, Math.round(Math.hypot(cur[0]-C.cx, cur[1]-C.cz))) : 1);
  return {cx:C.cx, cz:C.cz, baseY:C.baseY, radius:Math.max(1,radius), thick:Math.max(1,Math.min(radius,thick)), height:Math.max(1,Math.min(128,height))};
}
export function genCircleTower(params){
  const out=[]; if(!params) return out;
  const {cx,cz,baseY,radius,thick,height}=params;
  const inner=Math.max(0, radius-thick);
  const x0=Math.floor(cx-radius-1), x1=Math.ceil(cx+radius+1), z0=Math.floor(cz-radius-1), z1=Math.ceil(cz+radius+1);
  const CAP=180000;
  for(let z=z0;z<=z1;z++)for(let x=x0;x<=x1;x++){
    const d=Math.hypot((x+0.5)-cx,(z+0.5)-cz);
    if(d<=radius+0.35 && d>=inner-0.35){
      for(let y=0;y<height;y++){ out.push([x,baseY+y,z]); if(out.length>=CAP) return out; }
    }
  }
  return out;
}
export function squareCursorXZ(P){
  const rc=BUILD._lastRC;
  return (rc&&rc.pos) ? [rc.pos[0], rc.pos[2]] : [P[0]+0.5, P[2]+0.5];
}
export function squareParams(P){
  const S=BUILD.square; if(!S) return null;
  const cur=P?squareCursorXZ(P):[S.x1+0.5,S.z1+0.5];
  const ox=Math.round(cur[0]-0.5), oz=Math.round(cur[1]-0.5);
  const x1=S.x1 ?? ox, z1=S.z1 ?? oz;
  const x0=Math.min(S.x0,x1), x2=Math.max(S.x0,x1), z0=Math.min(S.z0,z1), z2=Math.max(S.z0,z1);
  const maxIn=Math.max(1, Math.floor((Math.min(x2-x0+1,z2-z0+1)+1)/2));
  const edgeDist=Math.min(Math.abs(ox-x0),Math.abs(ox-x2),Math.abs(oz-z0),Math.abs(oz-z2));
  const thick=S.thick ?? Math.max(1, Math.min(maxIn, edgeDist+1));
  const cx=(x0+x2+1)*0.5, cz=(z0+z2+1)*0.5;
  const height=S.height ?? (S.step===3 ? Math.max(1, Math.round(Math.hypot(cur[0]-cx, cur[1]-cz))) : 1);
  return {x0,x1:x2,z0,z1:z2,baseY:S.baseY,thick:Math.max(1,Math.min(maxIn,thick)),height:Math.max(1,Math.min(128,height))};
}
export function genSquareRoom(params){
  const out=[]; if(!params) return out;
  const {x0,x1,z0,z1,baseY,thick,height}=params;
  const CAP=180000;
  for(let z=z0;z<=z1;z++)for(let x=x0;x<=x1;x++){
    const d=Math.min(x-x0,x1-x,z-z0,z1-z);
    if(d>=0 && d<thick){
      for(let y=0;y<height;y++){ out.push([x,baseY+y,z]); if(out.length>=CAP) return out; }
    }
  }
  return out;
}
export function genGhosts(){
  const T=BUILD.tool, hgt=BUILD.height, th=BUILD.thick, A=BUILD.anchor, M=BUILD.mid, P=BUILD.hover;
  if (!P && !A) return [];
  const base = (A?A[1]:P[1]);
  let cols=[];
  if (T==='block'){ const t=P||A; return t?[[t[0],t[1],t[2]]]:[]; }
  if (T==='remove'){ const t=P||A; if(!t) return []; // remove targets the solid hit, not the air
      const hit=BUILD._lastHit; return hit?[[hit[0],hit[1],hit[2]]]:[]; }
  if (T==='wall'){ const a=A||P, b=P||A; cols=thicken(lineXZ(a[0],a[2],b[0],b[2]),th); return raiseColumns(cols,base,hgt); }
  if (T==='rect'){ const a=A||P, b=P||A; const x0=Math.min(a[0],b[0]),x1=Math.max(a[0],b[0]),z0=Math.min(a[2],b[2]),z1=Math.max(a[2],b[2]);
      for(let z=z0;z<=z1;z++)for(let x=x0;x<=x1;x++)cols.push([x,z]); return raiseColumns(cols,base,Math.max(1,hgt)); }
  if (T==='disc'){ const c=A||P, e=P||A; const R=A?Math.round(Math.hypot(e[0]-c[0],e[2]-c[2])):BUILD.radius;
      for(let z=-R;z<=R;z++)for(let x=-R;x<=R;x++){ const d=Math.hypot(x,z);
        if(d<=R+0.5) cols.push([c[0]+x,c[2]+z]); }
      return raiseColumns(cols, base, Math.max(1,hgt)); }
  if (T==='circle'){
    if(!BUILD.circle) return P?[[P[0],P[1],P[2]]]:[];   // pre-click 1: show center marker
    return genCircleTower(circleParams(P)); }             // steps 1-3: live ring preview
  if (T==='square'){
    if(!BUILD.square) return P?[[P[0],P[1],P[2]]]:[];   // pre-click 1: show first corner
    return genSquareRoom(squareParams(P)); }             // steps 1-3: live rectangular-room preview
  if (T==='arc'){ const a=A, m=M;
      if(!a){ if(!P) return []; cols=thicken(lineXZ(P[0],P[2],P[0],P[2]),th); return raiseColumns(cols,base,hgt); }
      if(!m){ // phase 1: chord preview from start to cursor
        const end=P||a; cols=thicken(lineXZ(a[0],a[2],end[0],end[2]),1); return raiseColumns(cols,base,hgt); }
      // phase 2: a=start, m=end, P=through-point; control pt so curve passes through P at t=0.5
      const end=m, thr=P||end;
      const cpx=2*thr[0]-0.5*(a[0]+end[0]), cpz=2*thr[2]-0.5*(a[2]+end[2]);
      const D=Math.hypot(end[0]-a[0],end[2]-a[2]);
      const pts=[]; const N=Math.max(8,Math.ceil(D*1.5)+1);
      for(let i=0;i<=N;i++){ const t=i/N, it=1-t;
        const x=it*it*a[0]+2*it*t*cpx+t*t*end[0];
        const z=it*it*a[2]+2*it*t*cpz+t*t*end[2];
        pts.push([Math.round(x),Math.round(z)]); }
      cols=thicken(pts,th); return raiseColumns(cols,base,hgt); }
  return [];
}