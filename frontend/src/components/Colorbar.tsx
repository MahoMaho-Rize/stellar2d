import { useMemo } from 'react';
import type { FieldName, ColormapName } from '../lib/types';
import { FIELD_LABELS } from '../lib/types';
import { colormapRGB } from '../lib/colormap';

interface Props {
  field: FieldName;
  colormap: ColormapName;
  vmin: number;
  vmax: number;
  logScale: boolean;
  height?: number;
}

export default function Colorbar({ field, colormap, vmin, vmax, logScale, height = 300 }: Props) {
  const gradientUrl = useMemo(() => {
    const w = 1, h = 256;
    const canvas = document.createElement('canvas');
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d')!;
    for (let y = 0; y < h; y++) {
      const t = 1 - y / (h - 1); // top = max, bottom = min
      const [r, g, b] = colormapRGB(t, colormap);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(0, y, w, 1);
    }
    return canvas.toDataURL();
  }, [colormap]);

  const ticks = useMemo(() => {
    const n = 5;
    const lo = logScale ? Math.log10(Math.max(vmin, 1e-30)) : vmin;
    const hi = logScale ? Math.log10(Math.max(vmax, 1e-30)) : vmax;
    return Array.from({ length: n }, (_, i) => {
      const frac = i / (n - 1);
      const val = lo + frac * (hi - lo);
      const display = logScale ? Math.pow(10, val) : val;
      return { frac, label: display.toExponential(2) };
    });
  }, [vmin, vmax, logScale]);

  return (
    <div style={{
      position: 'absolute', right: 16, top: '50%', transform: 'translateY(-50%)',
      display: 'flex', alignItems: 'stretch', gap: 6, pointerEvents: 'none',
    }}>
      {/* Gradient bar */}
      <div style={{
        width: 18, height, borderRadius: 3, overflow: 'hidden',
        border: '1px solid rgba(255,255,255,0.3)',
      }}>
        <img src={gradientUrl} alt="" style={{ width: '100%', height: '100%', display: 'block' }} />
      </div>

      {/* Tick labels */}
      <div style={{
        position: 'relative', height, display: 'flex', flexDirection: 'column',
        justifyContent: 'space-between',
      }}>
        {ticks.slice().reverse().map((tick, i) => (
          <span key={i} style={{
            fontSize: 11, color: '#ccc', lineHeight: 1, whiteSpace: 'nowrap',
            textShadow: '0 0 4px #000, 0 0 2px #000',
          }}>
            {tick.label}
          </span>
        ))}
      </div>

      {/* Field label (rotated) */}
      <div style={{
        writingMode: 'vertical-rl', textOrientation: 'mixed',
        fontSize: 12, color: '#aaa', textAlign: 'center', alignSelf: 'center',
        textShadow: '0 0 4px #000',
      }}>
        {FIELD_LABELS[field]}{logScale ? ' (log)' : ''}
      </div>
    </div>
  );
}
