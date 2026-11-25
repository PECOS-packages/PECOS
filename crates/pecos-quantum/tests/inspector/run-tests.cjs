// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

// Fresh exports for every run: never accept a stale checked-in HTML fixture.
const directory = fs.mkdtempSync(path.join(os.tmpdir(), 'pecos-inspector-'));
function run(command, args, options) {
  const result = spawnSync(command, args, { stdio: 'inherit', ...options });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (${result.signal || result.status})`);
}
try {
  run('cargo', ['run', '--locked', '-p', 'pecos-quantum', '--example', 'inspector_fixtures', '--', directory], {
    cwd: path.resolve(__dirname, '../../../..'),
  });
  run(process.execPath, ['--test', 'search.test.cjs', 'navigation.test.cjs'], {
    cwd: __dirname,
    env: { ...process.env, PECOS_INSPECTOR_FIXTURES: directory },
  });
} finally {
  fs.rmSync(directory, { recursive: true, force: true });
}
