import type { VTKSnapshot, StepDiagnostics } from './types';

/**
 * Parse a stellar2d VTK Legacy ASCII file.
 *
 * Expected format:
 *   STRUCTURED_GRID, DIMENSIONS (nt+1) (nr+1) 1
 *   CELL_DATA with fields: density, pressure, phi, velocity
 */
export function parseVTK(text: string): VTKSnapshot {
  const lines = text.split('\n');
  let cursor = 0;

  const next = () => lines[cursor++]?.trim() ?? '';

  // Skip header lines (title, description, format)
  while (cursor < lines.length && !lines[cursor].startsWith('DIMENSIONS')) cursor++;
  const dimLine = next();
  const dims = dimLine.split(/\s+/).slice(1).map(Number);
  const ntNodes = dims[0]; // nt+1
  const nrNodes = dims[1]; // nr+1
  const nt = ntNodes - 1;
  const nr = nrNodes - 1;
  const nCells = nr * nt;
  const nPoints = nrNodes * ntNodes;

  // Read POINTS
  while (cursor < lines.length && !lines[cursor].startsWith('POINTS')) cursor++;
  next(); // skip POINTS line
  const points = new Float64Array(nPoints * 3);
  let pi = 0;
  while (pi < nPoints * 3) {
    const tokens = next().split(/\s+/).map(Number);
    for (const v of tokens) {
      if (pi < nPoints * 3) points[pi++] = v;
    }
  }

  // Reconstruct r_face and theta_face from node coordinates
  // Points are in order: outer loop i (radial), inner loop j (theta)
  // x = r*sin(theta), z = r*cos(theta)
  const rFace = new Float64Array(nrNodes);
  const thetaFace = new Float64Array(ntNodes);

  // r_face from i-th radial ring at j=0: r = sqrt(x^2 + z^2)
  for (let i = 0; i < nrNodes; i++) {
    const base = i * ntNodes * 3;
    const x = points[base], z = points[base + 2];
    rFace[i] = Math.sqrt(x * x + z * z);
  }
  // Ensure r_face[0] = 0 (origin)
  if (rFace[0] < 1e-15) rFace[0] = 0;

  // theta_face from j-th theta node at outermost ring (i=nrNodes-1)
  for (let j = 0; j < ntNodes; j++) {
    const base = (nrNodes - 1) * ntNodes * 3 + j * 3;
    const x = points[base], z = points[base + 2];
    thetaFace[j] = Math.atan2(x, z); // atan2(sin*r, cos*r) = theta
  }

  // Cell centers
  const rCenter = new Float64Array(nr);
  for (let i = 0; i < nr; i++) rCenter[i] = 0.5 * (rFace[i] + rFace[i + 1]);
  const thetaCenter = new Float64Array(nt);
  for (let j = 0; j < nt; j++) thetaCenter[j] = 0.5 * (thetaFace[j] + thetaFace[j + 1]);

  // Read CELL_DATA fields
  while (cursor < lines.length && !lines[cursor].startsWith('CELL_DATA')) cursor++;
  next(); // skip CELL_DATA line

  let density = new Float64Array(nCells);
  let pressure = new Float64Array(nCells);
  let phi = new Float64Array(nCells);
  let velocity = new Float64Array(nCells * 3);

  while (cursor < lines.length) {
    const line = lines[cursor]?.trim() ?? '';
    if (line.startsWith('SCALARS') || line.startsWith('VECTORS')) {
      const tokens = line.split(/\s+/);
      const name = tokens[1];
      const isVector = line.startsWith('VECTORS');
      cursor++;
      // Skip LOOKUP_TABLE line for scalars
      if (!isVector && cursor < lines.length && lines[cursor]?.trim().startsWith('LOOKUP_TABLE')) {
        cursor++;
      }
      const count = isVector ? nCells * 3 : nCells;
      const data = new Float64Array(count);
      let di = 0;
      while (di < count && cursor < lines.length) {
        const toks = next().split(/\s+/).map(Number);
        for (const v of toks) {
          if (di < count) data[di++] = v;
        }
      }
      switch (name) {
        case 'density': density = data; break;
        case 'pressure': pressure = data; break;
        case 'phi': phi = data; break;
        case 'velocity': velocity = data; break;
      }
    } else {
      cursor++;
    }
  }

  // Derive fields
  const gamma = 5.0 / 3.0;
  const vr = new Float64Array(nCells);
  const vtheta = new Float64Array(nCells);
  const mach = new Float64Array(nCells);
  const entropy = new Float64Array(nCells);

  for (let i = 0; i < nr; i++) {
    const theta_eq = thetaCenter[Math.floor(nt / 2)]; // equator for reference
    for (let j = 0; j < nt; j++) {
      const flat = i * nt + j;
      const rho = Math.max(density[flat], 1e-30);
      const P = Math.max(pressure[flat], 1e-30);
      const vx = velocity[flat * 3], vz = velocity[flat * 3 + 2];
      const th = thetaCenter[j];
      // Cartesian → spherical
      vr[flat] = vx * Math.sin(th) + vz * Math.cos(th);
      vtheta[flat] = vx * Math.cos(th) - vz * Math.sin(th);
      const speed = Math.sqrt(vr[flat] ** 2 + vtheta[flat] ** 2);
      const cs = Math.sqrt(gamma * P / rho);
      mach[flat] = speed / Math.max(cs, 1e-30);
      entropy[flat] = P / Math.pow(rho, gamma);
    }
  }

  return {
    nr, nt, points, rCenter, thetaCenter, rFace, thetaFace,
    density, pressure, phi, velocity, vr, vtheta, mach, entropy,
  };
}

/**
 * Parse stdout diagnostics lines:
 *   Step  NNNN  t = X.Xe+XX  dt = X.Xe+XX  M = X.Xe+XX  E = X.Xe+XX
 */
export function parseDiagnostics(text: string): StepDiagnostics[] {
  const re = /Step\s+(\d+)\s+t\s*=\s*([^\s]+)\s+dt\s*=\s*([^\s]+)\s+M\s*=\s*([^\s]+)\s+E\s*=\s*([^\s]+)/g;
  const result: StepDiagnostics[] = [];
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    result.push({
      step: parseInt(m[1]),
      t: parseFloat(m[2]),
      dt: parseFloat(m[3]),
      mass: parseFloat(m[4]),
      energy: parseFloat(m[5]),
    });
  }
  return result;
}
