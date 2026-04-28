import { useMemo } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import type { VTKSnapshot, FieldName } from '../lib/types';
import { FIELD_LABELS } from '../lib/types';

interface Props {
  snapshot: VTKSnapshot;
  fields: FieldName[];
  thetaIndex?: number; // which theta-line to sample (default: equator)
  logY?: boolean;
}

export default function RadialProfile({ snapshot, fields, thetaIndex, logY = false }: Props) {
  const { nr, nt, rCenter } = snapshot;
  const j = thetaIndex ?? Math.floor(nt / 2);

  const chartData = useMemo(() => {
    return Array.from({ length: nr }, (_, i) => {
      const flat = i * nt + j;
      const entry: Record<string, number> = { r: rCenter[i] };
      for (const f of fields) {
        entry[f] = snapshot[f][flat];
      }
      return entry;
    });
  }, [snapshot, fields, j, nr, nt, rCenter]);

  const colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22'];

  return (
    <ResponsiveContainer width="100%" height={350}>
      <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
        <XAxis dataKey="r" type="number" label={{ value: 'r', position: 'insideBottom', offset: -5 }} />
        <YAxis
          scale={logY ? 'log' : 'auto'}
          domain={logY ? ['auto', 'auto'] : undefined}
          allowDataOverflow={logY}
        />
        <Tooltip />
        <Legend />
        {fields.map((f, idx) => (
          <Line
            key={f}
            type="monotone"
            dataKey={f}
            name={FIELD_LABELS[f]}
            stroke={colors[idx % colors.length]}
            dot={false}
            strokeWidth={2}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}
