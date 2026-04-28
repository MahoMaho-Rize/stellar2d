import { useMemo, useRef, useEffect } from 'react';
import * as THREE from 'three';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import type { VTKSnapshot, FieldName, ColormapName } from '../lib/types';
import { colormapRGB } from '../lib/colormap';

interface Props {
  snapshot: VTKSnapshot;
  field: FieldName;
  colormap: ColormapName;
  logScale: boolean;
}

/** Build a BufferGeometry mesh from the structured grid cell faces */
function HeatmapMesh({ snapshot, field, colormap, logScale }: Props) {
  const meshRef = useRef<THREE.Mesh>(null);
  const { nr, nt, rFace, thetaFace } = snapshot;
  const data = snapshot[field];

  const { geometry, colors } = useMemo(() => {
    // Build triangle mesh: each cell → 2 triangles (quad split)
    const nCells = nr * nt;
    const positions = new Float32Array(nCells * 6 * 3); // 6 verts per cell (2 tris)
    const cols = new Float32Array(nCells * 6 * 3);

    // Compute vmin/vmax
    let vmin = Infinity, vmax = -Infinity;
    for (let k = 0; k < nCells; k++) {
      const v = logScale ? Math.log10(Math.max(data[k], 1e-30)) : data[k];
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
    const range = vmax - vmin || 1;

    for (let i = 0; i < nr; i++) {
      for (let j = 0; j < nt; j++) {
        const flat = i * nt + j;
        const val = logScale ? Math.log10(Math.max(data[flat], 1e-30)) : data[flat];
        const t = Math.max(0, Math.min(1, (val - vmin) / range));
        const [cr, cg, cb] = colormapRGB(t, colormap);
        const r0 = rFace[i], r1 = rFace[i + 1];
        const t0 = thetaFace[j], t1 = thetaFace[j + 1];

        // 4 corners in (x, z) plane: x = r*sin(theta), z = r*cos(theta)
        const x00 = r0*Math.sin(t0), z00 = r0*Math.cos(t0);
        const x10 = r1*Math.sin(t0), z10 = r1*Math.cos(t0);
        const x01 = r0*Math.sin(t1), z01 = r0*Math.cos(t1);
        const x11 = r1*Math.sin(t1), z11 = r1*Math.cos(t1);

        const base = flat * 18; // 6 vertices * 3 coords
        // Triangle 1: (00, 10, 11)
        positions[base] = x00; positions[base+1] = 0; positions[base+2] = z00;
        positions[base+3] = x10; positions[base+4] = 0; positions[base+5] = z10;
        positions[base+6] = x11; positions[base+7] = 0; positions[base+8] = z11;
        // Triangle 2: (00, 11, 01)
        positions[base+9] = x00;  positions[base+10] = 0; positions[base+11] = z00;
        positions[base+12] = x11; positions[base+13] = 0; positions[base+14] = z11;
        positions[base+15] = x01; positions[base+16] = 0; positions[base+17] = z01;

        const cBase = flat * 18;
        for (let v = 0; v < 6; v++) {
          cols[cBase + v*3] = cr/255;
          cols[cBase + v*3+1] = cg/255;
          cols[cBase + v*3+2] = cb/255;
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(cols, 3));
    return { geometry: geo, colors: cols };
  }, [snapshot, field, colormap, logScale, data, nr, nt, rFace, thetaFace]);

  return (
    <mesh ref={meshRef} geometry={geometry}>
      <meshBasicMaterial vertexColors side={THREE.DoubleSide} />
    </mesh>
  );
}

export default function Heatmap2D({ snapshot, field, colormap, logScale }: Props) {
  const rMax = snapshot.rFace[snapshot.nr];
  return (
    <div style={{ width: '100%', height: '100%', minHeight: 400 }}>
      <Canvas
        camera={{ position: [0, rMax * 2, 0], up: [0, 0, 1], near: 0.01, far: rMax * 10 }}
        orthographic={false}
      >
        <ambientLight intensity={1} />
        <HeatmapMesh snapshot={snapshot} field={field} colormap={colormap} logScale={logScale} />
        <OrbitControls enableRotate={true} enablePan={true} enableZoom={true} />
        {/* Axis helper */}
        <axesHelper args={[rMax * 0.3]} />
      </Canvas>
    </div>
  );
}
