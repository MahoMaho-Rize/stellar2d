/** Parsed VTK simulation snapshot */
export interface VTKSnapshot {
  nr: number;
  nt: number;
  /** Node positions: (nr+1)*(nt+1) points, each [x, y, z] */
  points: Float64Array;
  /** Cell center r coordinates, length nr */
  rCenter: Float64Array;
  /** Cell center theta coordinates, length nt */
  thetaCenter: Float64Array;
  /** r face positions, length nr+1 */
  rFace: Float64Array;
  /** theta face positions, length nt+1 */
  thetaFace: Float64Array;
  /** Cell data fields (flat arrays of nr*nt) */
  density: Float64Array;
  pressure: Float64Array;
  phi: Float64Array;
  /** Velocity per cell: [vx, vy, vz] interleaved (3*nr*nt) */
  velocity: Float64Array;
  /** Derived fields */
  vr: Float64Array;
  vtheta: Float64Array;
  mach: Float64Array;
  entropy: Float64Array;
}

/** Time-series diagnostics from stdout */
export interface StepDiagnostics {
  step: number;
  t: number;
  dt: number;
  mass: number;
  energy: number;
}

export type FieldName = 'density' | 'pressure' | 'phi' | 'mach' | 'entropy' | 'vr' | 'vtheta';

export const FIELD_LABELS: Record<FieldName, string> = {
  density: 'Density \u03c1',
  pressure: 'Pressure P',
  phi: 'Gravity \u03a6',
  mach: 'Mach number',
  entropy: 'Entropy s',
  vr: 'Radial velocity v_r',
  vtheta: '\u03b8-velocity v_\u03b8',
};

export const COLORMAPS = ['inferno', 'viridis', 'plasma', 'magma', 'coolwarm'] as const;
export type ColormapName = typeof COLORMAPS[number];
