import test from 'node:test';
import assert from 'node:assert/strict';
import worker from '../worker.js';

const assets = { ASSETS: { fetch: async () => new Response('asset', { status: 200 }) } };

test('canonical homepage serves assets without a redirect loop', async () => {
  const response = await worker.fetch(new Request('https://neurolytics.net/?utm_source=demo'), assets);
  assert.equal(response.status, 200);
  assert.equal(await response.text(), 'asset');
});

test('www and a legacy entry point consolidate in a single hop with query intact', async () => {
  const response = await worker.fetch(new Request('https://www.neurolytics.net/landing.html?utm_source=demo'), assets);
  assert.equal(response.status, 301);
  assert.equal(response.headers.get('location'), 'https://neurolytics.net/?utm_source=demo');
});

test('www asset paths and HTTP upgrade preserve paths', async () => {
  const response = await worker.fetch(new Request('http://www.neurolytics.net/assets/analyzed-cortex.bin'), assets);
  assert.equal(response.status, 301);
  assert.equal(response.headers.get('location'), 'https://neurolytics.net/assets/analyzed-cortex.bin');
});

test('local preview is not sent to production', async () => {
  const response = await worker.fetch(new Request('http://localhost:8787/landing.html'), assets);
  assert.equal(response.status, 200);
});

test('missing pages retain the asset service 404 instead of a homepage soft 404', async () => {
  const missing = { ASSETS: { fetch: async () => new Response('Not found', { status: 404 }) } };
  const response = await worker.fetch(new Request('https://neurolytics.net/not-a-page'), missing);
  assert.equal(response.status, 404);
});

test('old domain preserves paths and query strings on permanent redirects', async () => {
  const response = await worker.fetch(new Request('https://nueroanalytics.com/assets/analyzed-cortex.bin?v=10'), assets);
  assert.equal(response.status, 301);
  assert.equal(response.headers.get('location'), 'https://neurolytics.net/assets/analyzed-cortex.bin?v=10');
});
