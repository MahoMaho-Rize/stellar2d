import { useMemo } from 'react';
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
  azimuthalSegments?: number;
}

/**
 * 3D star rendered by revolving the axisymmetric (r,theta) data around the z-axis.
 * Creates a LatheGeometry-like mesh with per-vertex coloring from the selected field.
 */
function StarMesh({ snapshot, field, colormap, logScale, azimuthalSegments = 64 }: Props) {
  const { nr, nt, rFace, thetaFace } = snapshot;
  const data = snapshot[field];
  const nPhi = azimuthalSegments;

  const geometry = useMemo(() => {
    // Compute vmin/vmax
    const nCells = nr * nt;
    let vmin = Infinity, vmax = -Infinity;
    for (let k = 0; k < nCells; k++) {
      const v = logScale ? Math.log10(Math.max(data[k], 1e-30)) : data[k];
      if (v < vmin) vmin = v;
      if (v > vmax) vmax = v;
    }
    const range = vmax - vmin || 1;

    // Build revolution surface: for each cell (i,j), extrude around z-axis
    // We create a surface at the OUTER radial face of each cell for simplicity
    // Vertices: (nr+1) * (nt+1) * nPhi, forming quads revolved around z-axis
    const nR = nr + 1, nT = nt + 1;
    const totalVerts = nR * nT * nPhi;
    const positions = new Float32Array(totalVerts * 3);
    const colors = new Float32Array(totalVerts * 3);

    for (let ip = 0; ip < nPhi; ip++) {
      const phi = (2 * Math.PI * ip) / nPhi;
      const cosPhi = Math.cos(phi), sinPhi = Math.sin(phi);

      for (let i = 0; i < nR; i++) {
        const r = rFace[i];
        for (let jt = 0; jt < nT; jt++) {
          const theta = thetaFace[jt];
          const rSin = r * Math.sin(theta);
          const x = rSin * cosPhi;
          const y = rSin * sinPhi;
          const z = r * Math.cos(theta);

          const vidx = (ip * nR * nT + i * nT + jt) * 3;
          positions[vidx] = x;
          positions[vidx + 1] = y;
          positions[vidx + 2] = z;

          // Color from nearest cell
          const ci = Math.min(i, nr - 1);
          const cj = Math.min(jt, nt - 1);
          const flat = ci * nt + cj;
          const val = logScale ? Math.log10(Math.max(data[flat], 1e-30)) : data[flat];
          const t = Math.max(0, Math.min(1, (val - vmin) / range));
          const [cr, cg, cb] = colormapRGB(t, colormap);
          colors[vidx] = cr / 255;
          colors[vidx + 1] = cg / 255;
          colors[vidx + 2] = cb / 255;
        }
      }
    }

    // Build index buffer (triangulated quads)
    const indices: number[] = [];
    for (let ip = 0; ip < nPhi; ip++) {
      const ipNext = (ip + 1) % nPhi;
      for (let i = 0; i < nr; i++) {
        for (let jt = 0; jt < nt; jt++) {
          const v00 = ip * nR * nT + i * nT + jt;
          const v10 = ip * nR * nT + (i + 1) * nT + jt;
          const v01 = ip * nR * nT + i * nT + (jt + 1);
          const v11 = ip * nR * nT + (i + 1) * nT + (jt + 1);
          const w00 = ipNext * nR * nT + i * nT + jt;
          const w10 = ipNext * nR * nT + (i + 1) * nT + jt;
          const w01 = ipNext * nR * nT + i * nT + (jt + 1);
          const w11 = ipNext * nR * nT + (i + 1) * nT + (jt + 1);

          // Only render the outermost radial shell for the "surface" look
          // For full volume rendering we'd need raymarching — just do outer surface
          if (i === nr - 1) {
            // Outer face quad (v10-v11-w11-w10) for i=nr-1
            indices.push(v10, v11, w11, v10, w11, w10);
          }
          // Optionally render a cross-section slice at phi=0
          if (ip === 0) {
            indices.push(v00, v10, v11, v00, v11, v01);
          }
        }
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geo.setIndex(indices);
    geo.computeVertexNormals();
    return geo;
  }, [snapshot, field, colormap, logScale, nPhi, data, nr, nt, rFace, thetaFace]);

  return (
    <mesh geometry={geometry}>
      <meshBasicMaterial vertexColors side={THREE.DoubleSide} transparent opacity={0.95} />
    </mesh>
  );
}

export default function Star3D({ snapshot, field, colormap, logScale, azimuthalSegments }: Props) {
  const rMax = snapshot.rFace[snapshot.nr];
  return (
    <div style={{ width: '100%', height: '100%', minHeight: 500 }}>
      <Canvas camera={{ position: [rMax * 2, rMax * 1.5, rMax * 2], fov: 45 }}>
        <ambientLight intensity={0.6} />
        <directionalLight position={[5, 5, 5]} intensity={0.8} />
        <StarMesh
          snapshot={snapshot}
          field={field}
          colormap={colormap}
          logScale={logScale}
          azimuthalSegments={azimuthalSegments}
        />
        <OrbitControls enableDamping dampingFactor={0.1} />
        <axesHelper args={[rMax * 0.3]} />
      </Canvas>
    </div>
  );
}
