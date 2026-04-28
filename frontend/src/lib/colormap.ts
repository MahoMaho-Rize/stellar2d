import type { ColormapName } from './types';

// Pre-baked colormap LUTs (256 entries, RGB [0,1])
const INFERNO_DATA: [number,number,number][] = [];
const VIRIDIS_DATA: [number,number,number][] = [];

// Generate simplified colormaps via interpolation
function lerp3(a: [number,number,number], b: [number,number,number], t: number): [number,number,number] {
  return [a[0]+(b[0]-a[0])*t, a[1]+(b[1]-a[1])*t, a[2]+(b[2]-a[2])*t];
}

function buildLUT(stops: {t:number,c:[number,number,number]}[]): [number,number,number][] {
  const lut: [number,number,number][] = [];
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let si = 0;
    while (si < stops.length - 2 && stops[si + 1].t < t) si++;
    const s0 = stops[si], s1 = stops[si + 1];
    const lt = (t - s0.t) / (s1.t - s0.t);
    lut.push(lerp3(s0.c, s1.c, lt));
  }
  return lut;
}

const INFERNO = buildLUT([
  {t:0,c:[0,0,.02]},{t:.13,c:[.26,.01,.4]},{t:.25,c:[.53,.06,.42]},
  {t:.38,c:[.76,.17,.27]},{t:.5,c:[.93,.35,.13]},{t:.63,c:[.99,.56,.04]},
  {t:.75,c:[.96,.78,.15]},{t:.88,c:[.95,.94,.42]},{t:1,c:[.99,.99,.75]},
]);
const VIRIDIS = buildLUT([
  {t:0,c:[.27,.004,.33]},{t:.13,c:[.28,.14,.45]},{t:.25,c:[.23,.29,.5]},
  {t:.38,c:[.17,.43,.5]},{t:.5,c:[.13,.57,.45]},{t:.63,c:[.2,.69,.35]},
  {t:.75,c:[.45,.77,.22]},{t:.88,c:[.74,.84,.15]},{t:1,c:[.99,.91,.14]},
]);
const PLASMA = buildLUT([
  {t:0,c:[.05,.03,.53]},{t:.13,c:[.33,.01,.63]},{t:.25,c:[.55,.05,.59]},
  {t:.38,c:[.72,.15,.48]},{t:.5,c:[.86,.27,.34]},{t:.63,c:[.95,.43,.2]},
  {t:.75,c:[.99,.6,.09]},{t:.88,c:[.97,.79,.15]},{t:1,c:[.94,.98,.13]},
]);
const MAGMA = buildLUT([
  {t:0,c:[0,0,.02]},{t:.13,c:[.17,.04,.39]},{t:.25,c:[.41,.05,.51]},
  {t:.38,c:[.6,.12,.5]},{t:.5,c:[.78,.24,.45]},{t:.63,c:[.93,.39,.38]},
  {t:.75,c:[.98,.58,.37]},{t:.88,c:[.99,.79,.56]},{t:1,c:[.99,.99,.75]},
]);
const COOLWARM = buildLUT([
  {t:0,c:[.23,.3,.75]},{t:.25,c:[.55,.69,.96]},{t:.5,c:[.87,.87,.87]},
  {t:.75,c:[.96,.6,.47]},{t:1,c:[.71,.21,.18]},
]);

const LUTS: Record<ColormapName, [number,number,number][]> = {
  inferno: INFERNO, viridis: VIRIDIS, plasma: PLASMA, magma: MAGMA, coolwarm: COOLWARM,
};

/**
 * Map a normalized value [0,1] to RGB [0,255].
 */
export function colormapRGB(t: number, name: ColormapName = 'inferno'): [number,number,number] {
  const lut = LUTS[name];
  const idx = Math.max(0, Math.min(255, Math.round(t * 255)));
  const c = lut[idx];
  return [Math.round(c[0]*255), Math.round(c[1]*255), Math.round(c[2]*255)];
}

/**
 * Build an ImageData-like Uint8Array (RGBA) for a field mapped to [vmin, vmax].
 * Output is nr*nt*4 bytes, row-major (i=radial outer, j=theta inner).
 */
export function fieldToRGBA(
  data: Float64Array, nr: number, nt: number,
  vmin: number, vmax: number, cmap: ColormapName = 'inferno', logScale = false,
): Uint8Array {
  const buf = new Uint8Array(nr * nt * 4);
  const lo = logScale ? Math.log10(Math.max(vmin, 1e-30)) : vmin;
  const hi = logScale ? Math.log10(Math.max(vmax, 1e-30)) : vmax;
  const range = hi - lo || 1;
  for (let idx = 0; idx < nr * nt; idx++) {
    const val = logScale ? Math.log10(Math.max(data[idx], 1e-30)) : data[idx];
    const t = Math.max(0, Math.min(1, (val - lo) / range));
    const [r, g, b] = colormapRGB(t, cmap);
    buf[idx * 4] = r;
    buf[idx * 4 + 1] = g;
    buf[idx * 4 + 2] = b;
    buf[idx * 4 + 3] = 255;
  }
  return buf;
}
