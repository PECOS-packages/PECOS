#!/usr/bin/env node
// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import process from "node:process";

function usage(message) {
  if (message) console.error(message);
  console.error(
    "Usage: node scripts/validate_frontier_wasm_predictions.mjs MODULE [--fixture SHOTS.fwr] [--shots N] [--expect-errors N] [--expect-sha256 HEX] [--json PATH]",
  );
  process.exit(2);
}

function parseNonnegative(value, name) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) usage(`${name} must be a nonnegative integer`);
  return parsed;
}

const argv = process.argv.slice(2);
if (argv.length === 0 || argv.includes("--help")) usage();
const modulePath = argv.shift();
const options = {
  fixture: undefined,
  shots: undefined,
  expectErrors: undefined,
  expectSha256: undefined,
  json: undefined,
};
while (argv.length) {
  const name = argv.shift();
  const value = argv.shift();
  if (value === undefined) usage(`missing value for ${name}`);
  if (name === "--fixture") options.fixture = value;
  else if (name === "--shots") options.shots = parseNonnegative(value, name);
  else if (name === "--expect-errors") options.expectErrors = parseNonnegative(value, name);
  else if (name === "--expect-sha256") {
    if (!/^[0-9a-f]{64}$/iu.test(value)) usage("--expect-sha256 must be a 64-digit hexadecimal digest");
    options.expectSha256 = value.toLowerCase();
  } else if (name === "--json") options.json = value;
  else usage(`unknown option: ${name}`);
}

const bytes = await readFile(modulePath);
const { instance } = await WebAssembly.instantiate(bytes, {});
const wasm = instance.exports;
for (const name of ["init", "frontier_status", "frontier_result_0", "frontier_result_1", "frontier_result_2", "frontier_result_3"]) {
  if (typeof wasm[name] !== "function") throw new Error(`missing Wasm export: ${name}`);
}

wasm.init();
if (wasm.frontier_status() !== 0) throw new Error(`decoder initialization failed with status ${wasm.frontier_status()}`);
let fixtureShots;
let replayShot;
let fixturePath = null;
if (options.fixture !== undefined) {
  if (typeof wasm.frontier_decode !== "function") throw new Error("missing Wasm export: frontier_decode");
  fixturePath = options.fixture;
  const fixture = await readFile(fixturePath);
  if (fixture.length < 20 || fixture.toString("ascii", 0, 4) !== "FWR1") throw new Error("invalid FWR1 fixture");
  const version = fixture.readUInt32LE(4);
  fixtureShots = fixture.readUInt32LE(8);
  const detectors = fixture.readUInt32LE(12);
  const observables = fixture.readUInt32LE(16);
  if (version !== 2) throw new Error(`unsupported FWR1 version: ${version}`);
  if (detectors > 128 || observables > 128) throw new Error("fixture exceeds the Wasm 128-bit ABI");
  const detectorWords = Math.ceil(detectors / 32);
  const observableWords = Math.ceil(observables / 32);
  const recordWords = detectorWords + observableWords;
  if (fixture.length !== 20 + fixtureShots * recordWords * 4) throw new Error("invalid FWR1 fixture length");
  replayShot = (index) => {
    const offset = 20 + index * recordWords * 4;
    const words = [0, 0, 0, 0];
    for (let word = 0; word < detectorWords; word += 1) words[word] = fixture.readInt32LE(offset + word * 4);
    wasm.frontier_decode(...words);
    if (wasm.frontier_status() !== 0) return -1;
    let mismatch = false;
    for (let word = 0; word < observableWords; word += 1) {
      const actual = fixture.readUInt32LE(offset + (detectorWords + word) * 4);
      mismatch ||= (wasm[`frontier_result_${word}`]() >>> 0) !== actual;
    }
    return Number(mismatch);
  };
} else {
  for (const name of ["frontier_replay_shot_count", "frontier_replay_shot"]) {
    if (typeof wasm[name] !== "function") throw new Error(`missing Wasm export: ${name}`);
  }
  fixtureShots = wasm.frontier_replay_shot_count();
  replayShot = (index) => wasm.frontier_replay_shot(index);
}
const shots = options.shots ?? fixtureShots;
if (shots > fixtureShots) usage(`requested ${shots} shots, but the fixture contains ${fixtureShots}`);

const hash = createHash("sha256");
let logicalErrors = 0;
const started = performance.now();
for (let index = 0; index < shots; index += 1) {
  const mismatch = replayShot(index);
  if (mismatch < 0) throw new Error(`shot ${index} failed with status ${wasm.frontier_status()}`);
  logicalErrors += mismatch;
  let prediction = 0n;
  for (let word = 0; word < 4; word += 1) {
    prediction |= BigInt(wasm[`frontier_result_${word}`]() >>> 0) << BigInt(32 * word);
  }
  if (index !== 0) hash.update("\n");
  hash.update(prediction.toString());
  if ((index + 1) % 100 === 0 || index + 1 === shots) {
    const elapsedSeconds = (performance.now() - started) / 1000;
    console.log(`Validated ${index + 1}/${shots} shots (${elapsedSeconds.toFixed(1)} s)`);
  }
}
const elapsedSeconds = (performance.now() - started) / 1000;
const predictionSha256 = hash.digest("hex");
const errorsMatch = options.expectErrors === undefined || logicalErrors === options.expectErrors;
const sha256Match = options.expectSha256 === undefined || predictionSha256 === options.expectSha256;
const validated = errorsMatch && sha256Match;
const report = {
  schema_version: 1,
  module: modulePath,
  fixture: fixturePath,
  wasm_bytes: bytes.length,
  runtime: `Node ${process.version}`,
  platform: `${process.platform}/${process.arch}`,
  cpu: os.cpus()[0]?.model ?? "unknown",
  fixture_shots: fixtureShots,
  validated_shots: shots,
  logical_errors: logicalErrors,
  logical_error_rate: shots === 0 ? null : logicalErrors / shots,
  prediction_sha256: predictionSha256,
  expected_logical_errors: options.expectErrors ?? null,
  expected_prediction_sha256: options.expectSha256 ?? null,
  errors_match: errorsMatch,
  prediction_sha256_match: sha256Match,
  validated,
  elapsed_seconds: elapsedSeconds,
  shots_per_second: elapsedSeconds === 0 ? null : shots / elapsedSeconds,
};

console.log(`Logical errors: ${logicalErrors}/${shots}`);
console.log(`Prediction SHA-256: ${predictionSha256}`);
console.log(`Reference parity: ${validated ? "PASS" : "FAIL"}`);
if (options.json) await writeFile(options.json, `${JSON.stringify(report, null, 2)}\n`);
if (!validated) process.exitCode = 1;
