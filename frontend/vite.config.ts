import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'vtk-api',
      configureServer(server) {
        // GET /api/list?dir=/path/to/runs/xxx → list VTK files
        server.middlewares.use('/api/list', (req, res) => {
          const url = new URL(req.url!, `http://${req.headers.host}`);
          const dir = url.searchParams.get('dir') || '';
          try {
            const files = fs.readdirSync(dir)
              .filter((f: string) => f.endsWith('.vtk'))
              .sort();
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify(files));
          } catch {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: 'Cannot read directory' }));
          }
        });
        // GET /api/vtk?path=/path/to/file.vtk → serve raw VTK file
        server.middlewares.use('/api/vtk', (req, res) => {
          const url = new URL(req.url!, `http://${req.headers.host}`);
          const filePath = url.searchParams.get('path') || '';
          try {
            const data = fs.readFileSync(filePath, 'utf-8');
            res.setHeader('Content-Type', 'text/plain');
            res.end(data);
          } catch {
            res.statusCode = 404;
            res.end('File not found');
          }
        });
      },
    },
  ],
});
