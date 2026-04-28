import { useState, useCallback, useMemo } from 'react';
import type { VTKSnapshot, FieldName, ColormapName, StepDiagnostics } from './lib/types';
import { FIELD_LABELS, COLORMAPS } from './lib/types';
import { parseVTK, parseDiagnostics } from './lib/vtk-parser';
import Heatmap2D from './components/Heatmap2D';
import RadialProfile from './components/RadialProfile';
import Star3D from './components/Star3D';
import Colorbar from './components/Colorbar';

type ViewMode = 'heatmap' | 'profile' | 'star3d';

export default function App() {
  const [snapshots, setSnapshots] = useState<VTKSnapshot[]>([]);
  const [diagnostics, setDiagnostics] = useState<StepDiagnostics[]>([]);
  const [frameIdx, setFrameIdx] = useState(0);
  const [field, setField] = useState<FieldName>('density');
  const [colormap, setColormap] = useState<ColormapName>('inferno');
  const [logScale, setLogScale] = useState(true);
  const [view, setView] = useState<ViewMode>('heatmap');
  const [profileFields, setProfileFields] = useState<FieldName[]>(['density', 'pressure']);

  const handleFiles = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    const vtkFiles = files.filter(f => f.name.endsWith('.vtk')).sort((a, b) => a.name.localeCompare(b.name));
    const logFiles = files.filter(f => f.name.endsWith('.log') || f.name.endsWith('.txt'));

    const snaps: VTKSnapshot[] = [];
    for (const f of vtkFiles) {
      const text = await f.text();
      snaps.push(parseVTK(text));
    }
    setSnapshots(snaps);
    if (snaps.length > 0) setFrameIdx(0);

    if (logFiles.length > 0) {
      const text = await logFiles[0].text();
      setDiagnostics(parseDiagnostics(text));
    }
  }, []);

  const snap = snapshots[frameIdx];

  const fieldRange: [number, number] = useMemo(() => {
    if (!snap) return [0, 1];
    const data = snap[field];
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < lo) lo = data[i];
      if (data[i] > hi) hi = data[i];
    }
    return [lo, hi];
  }, [snap, field]);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', fontFamily: 'system-ui, sans-serif', background: '#0a0a0a', color: '#e0e0e0' }}>
      {/* Header */}
      <header style={{ padding: '8px 16px', background: '#111', borderBottom: '1px solid #333', display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
        <h1 style={{ margin: 0, fontSize: 18, fontWeight: 600 }}>stellar2d viewer</h1>

        <label style={{ cursor: 'pointer', padding: '4px 12px', background: '#2563eb', borderRadius: 4, fontSize: 13 }}>
          Load VTK files
          <input type="file" multiple accept=".vtk,.log,.txt" onChange={handleFiles} style={{ display: 'none' }} />
        </label>

        {snap && (
          <>
            <span style={{ fontSize: 12, opacity: 0.6 }}>Grid: {snap.nr}x{snap.nt} | {snapshots.length} frames</span>

            {/* View mode */}
            <div style={{ display: 'flex', gap: 4 }}>
              {(['heatmap', 'profile', 'star3d'] as ViewMode[]).map(v => (
                <button key={v} onClick={() => setView(v)}
                  style={{ padding: '3px 10px', fontSize: 12, background: view === v ? '#2563eb' : '#222', border: '1px solid #444', borderRadius: 3, color: '#ddd', cursor: 'pointer' }}>
                  {v === 'heatmap' ? '2D Map' : v === 'profile' ? 'Profile' : '3D Star'}
                </button>
              ))}
            </div>

            {/* Field selector */}
            <select value={field} onChange={e => setField(e.target.value as FieldName)}
              style={{ background: '#222', color: '#ddd', border: '1px solid #444', borderRadius: 3, padding: '3px 8px', fontSize: 12 }}>
              {(Object.keys(FIELD_LABELS) as FieldName[]).map(f => (
                <option key={f} value={f}>{FIELD_LABELS[f]}</option>
              ))}
            </select>

            {/* Colormap */}
            <select value={colormap} onChange={e => setColormap(e.target.value as ColormapName)}
              style={{ background: '#222', color: '#ddd', border: '1px solid #444', borderRadius: 3, padding: '3px 8px', fontSize: 12 }}>
              {COLORMAPS.map(c => <option key={c} value={c}>{c}</option>)}
            </select>

            {/* Log scale */}
            <label style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 4 }}>
              <input type="checkbox" checked={logScale} onChange={e => setLogScale(e.target.checked)} />
              Log
            </label>
          </>
        )}
      </header>

      {/* Main content */}
      <main style={{ flex: 1, overflow: 'hidden', position: 'relative' }}>
        {!snap ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', opacity: 0.4 }}>
            <p>Drop or load VTK output files to begin</p>
          </div>
        ) : view === 'heatmap' ? (
          <>
            <Heatmap2D snapshot={snap} field={field} colormap={colormap} logScale={logScale} />
            <Colorbar field={field} colormap={colormap} vmin={fieldRange[0]} vmax={fieldRange[1]} logScale={logScale} />
          </>
        ) : view === 'profile' ? (
          <div style={{ padding: 16 }}>
            <div style={{ marginBottom: 8, display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {(Object.keys(FIELD_LABELS) as FieldName[]).map(f => (
                <label key={f} style={{ fontSize: 12, display: 'flex', alignItems: 'center', gap: 3 }}>
                  <input type="checkbox" checked={profileFields.includes(f)}
                    onChange={e => {
                      if (e.target.checked) setProfileFields([...profileFields, f]);
                      else setProfileFields(profileFields.filter(x => x !== f));
                    }} />
                  {FIELD_LABELS[f]}
                </label>
              ))}
            </div>
            <RadialProfile snapshot={snap} fields={profileFields} logY={logScale} />
          </div>
        ) : (
          <>
            <Star3D snapshot={snap} field={field} colormap={colormap} logScale={logScale} />
            <Colorbar field={field} colormap={colormap} vmin={fieldRange[0]} vmax={fieldRange[1]} logScale={logScale} />
          </>
        )}
      </main>

      {/* Timeline slider */}
      {snapshots.length > 1 && (
        <footer style={{ padding: '8px 16px', background: '#111', borderTop: '1px solid #333', display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 12, minWidth: 80 }}>Frame {frameIdx + 1}/{snapshots.length}</span>
          <input type="range" min={0} max={snapshots.length - 1} value={frameIdx}
            onChange={e => setFrameIdx(parseInt(e.target.value))}
            style={{ flex: 1 }} />
        </footer>
      )}
    </div>
  );
}
