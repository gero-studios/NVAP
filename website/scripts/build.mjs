import { mkdir, copyFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const root = new URL('../', import.meta.url);
const output = new URL('public/', root);
await mkdir(new URL('assets/', output), { recursive: true });
const files = ['landing.html', 'landing.css', 'landing.js', 'segmentation.js', 'mark.svg', 'robots.txt', 'sitemap.xml',
  'assets/analyzed-cortex.bin', 'assets/analyzed-cortex.json', 'assets/analyzed-cortex.png'];
for (const file of files) await copyFile(new URL(file, root), new URL(file, output));
await copyFile(new URL('landing.html', root), new URL('index.html', output));
console.log(`Built ${files.length + 1} public files in ${fileURLToPath(output)}`);
