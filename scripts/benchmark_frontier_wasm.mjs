#!/usr/bin/env node
// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

import { readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import process from "node:process";

function usage(message) {
  if (message) console.error(message);
  console.error(
    "Usage: node scripts/benchmark_frontier_wasm.mjs MODULE [--start N] [--shots N] [--stride N] [--warmup N] [--repeat N] [--mode shot|range|stream|correction] [--rounds N] [--json PATH]",
  );
  process.exit(2);
}

function parseNonnegative(value, name) {
  const parsed = Number(value);
  if (!Number.isSafeInteger(parsed) || parsed < 0) usage(`${name} must be a nonnegative integer`);
  return parsed;
}

function percentile(sortedValues, probability) {
  if (sortedValues.length === 0) return null;
  const position = (sortedValues.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sortedValues[lower];
  const fraction = position - lower;
  return sortedValues[lower] * (1 - fraction) + sortedValues[upper] * fraction;
}

const argv = process.argv.slice(2);
if (argv.length === 0 || argv.includes("--help")) usage();
const modulePath = argv.shift();
const options = {
  start: 0,
  shots: undefined,
  stride: 1,
  warmup: 1,
  repeat: 1,
  mode: "shot",
  rounds: 21,
  json: undefined,
};
while (argv.length) {
  const name = argv.shift();
  const value = argv.shift();
  if (value === undefined) usage(`missing value for ${name}`);
  if (name === "--mode") {
    if (!["shot", "range", "stream", "correction"].includes(value)) {
      usage("--mode must be shot, range, stream, or correction");
    }
    options.mode = value;
  } else if (name === "--json") options.json = value;
  else if (name === "--start") options.start = parseNonnegative(value, name);
  else if (name === "--shots") options.shots = parseNonnegative(value, name);
  else if (name === "--stride") options.stride = parseNonnegative(value, name);
  else if (name === "--warmup") options.warmup = parseNonnegative(value, name);
  else if (name === "--repeat") options.repeat = parseNonnegative(value, name);
  else if (name === "--rounds") options.rounds = parseNonnegative(value, name);
  else usage(`unknown option: ${name}`);
}
if (options.repeat === 0) usage("--repeat must be positive");
if (options.rounds === 0) usage("--rounds must be positive");
if (options.stride === 0) usage("--stride must be positive");
if (options.mode === "range" && options.stride !== 1) usage("--stride is not supported in range mode");

const bytes = await readFile(modulePath);
const compileStart = performance.now();
const { instance } = await WebAssembly.instantiate(bytes, {});
const compileMs = performance.now() - compileStart;
const wasm = instance.exports;
for (const name of ["init", "frontier_status", "frontier_replay_shot_count", "frontier_replay_shot", "frontier_replay_range"]) {
  if (typeof wasm[name] !== "function") throw new Error(`missing Wasm export: ${name}`);
}

const initStart = performance.now();
wasm.init();
const initMs = performance.now() - initStart;
if (wasm.frontier_status() !== 0) throw new Error(`decoder initialization failed with status ${wasm.frontier_status()}`);

const fixtureShots = wasm.frontier_replay_shot_count();
const availableShots =
  options.start >= fixtureShots ? 0 : Math.floor((fixtureShots - 1 - options.start) / options.stride) + 1;
const shots = options.shots ?? availableShots;
const finalSelectedIndex = shots === 0 ? options.start : options.start + (shots - 1) * options.stride;
if (options.start > fixtureShots || finalSelectedIndex >= fixtureShots) {
  usage(`selected shot index ${finalSelectedIndex} exceeds the ${fixtureShots}-shot fixture`);
}

function selectedShotIndex(position) {
  return options.start + position * options.stride;
}

function runOnce() {
  let errors = 0;
  if (options.mode === "range") {
    errors = wasm.frontier_replay_range(options.start, shots);
    if (errors < 0) throw new Error(`range replay failed with status ${wasm.frontier_status()}`);
  } else if (options.mode === "stream") {
    if (typeof wasm.frontier_replay_stream_shot !== "function") {
      throw new Error("module does not export frontier_replay_stream_shot");
    }
    for (let position = 0; position < shots; position += 1) {
      const index = selectedShotIndex(position);
      const mismatch = wasm.frontier_replay_stream_shot(index, options.rounds);
      if (mismatch < 0) throw new Error(`streamed shot ${index} failed with status ${wasm.frontier_status()}`);
      errors += mismatch;
    }
  } else {
    for (let position = 0; position < shots; position += 1) {
      const index = selectedShotIndex(position);
      const mismatch = wasm.frontier_replay_shot(index);
      if (mismatch < 0) throw new Error(`shot ${index} failed with status ${wasm.frontier_status()}`);
      errors += mismatch;
    }
  }
  return errors;
}

function runCorrectionOnce() {
  for (const name of ["frontier_replay_stream_prepare", "frontier_replay_stream_finish"]) {
    if (typeof wasm[name] !== "function") throw new Error(`missing Wasm export: ${name}`);
  }
  if (typeof wasm.frontier_reset === "function") wasm.frontier_reset();
  let errors = 0;
  let measuredMs = 0;
  const samplesMs = [];
  const sampleRecords = [];
  for (let position = 0; position < shots; position += 1) {
    const index = selectedShotIndex(position);
    if (wasm.frontier_replay_stream_prepare(index, options.rounds) !== 0) {
      throw new Error(`stream preparation for shot ${index} failed with status ${wasm.frontier_status()}`);
    }
    const start = performance.now();
    const mismatch = wasm.frontier_replay_stream_finish();
    const sampleMs = performance.now() - start;
    measuredMs += sampleMs;
    samplesMs.push(sampleMs);
    sampleRecords.push({ shot_index: index, latency_ms: sampleMs });
    if (mismatch < 0) throw new Error(`correction for shot ${index} failed with status ${wasm.frontier_status()}`);
    errors += mismatch;
  }
  return { errors, measuredMs, samplesMs, sampleRecords };
}

for (let index = 0; index < options.warmup; index += 1) {
  if (options.mode === "correction") runCorrectionOnce();
  else runOnce();
}
const elapsed = [];
const correctionSamplesMs = [];
const correctionSampleRecords = [];
let errors = 0;
for (let index = 0; index < options.repeat; index += 1) {
  if (options.mode === "correction") {
    const result = runCorrectionOnce();
    errors = result.errors;
    elapsed.push(result.measuredMs);
    correctionSamplesMs.push(...result.samplesMs);
    correctionSampleRecords.push(
      ...result.sampleRecords.map((sample) => ({ repeat_index: index, ...sample })),
    );
  } else {
    const start = performance.now();
    errors = runOnce();
    elapsed.push(performance.now() - start);
  }
}
const meanMs = elapsed.reduce((sum, value) => sum + value, 0) / elapsed.length;
const perShotUs = shots === 0 ? 0 : (meanMs * 1000) / shots;
const sortedCorrectionSamplesMs = correctionSamplesMs.toSorted((left, right) => left - right);
const slowestCorrectionSamples = correctionSampleRecords
  .toSorted((left, right) => right.latency_ms - left.latency_ms)
  .slice(0, 10);
const correctionDistribution =
  options.mode === "correction"
    ? {
        sample_count: sortedCorrectionSamplesMs.length,
        p50_ms: percentile(sortedCorrectionSamplesMs, 0.5),
        p95_ms: percentile(sortedCorrectionSamplesMs, 0.95),
        p99_ms: percentile(sortedCorrectionSamplesMs, 0.99),
        max_ms: sortedCorrectionSamplesMs.at(-1) ?? null,
        under_50_ms_count: sortedCorrectionSamplesMs.filter((value) => value < 50).length,
        under_50_ms_fraction:
          sortedCorrectionSamplesMs.length === 0
            ? null
            : sortedCorrectionSamplesMs.filter((value) => value < 50).length / sortedCorrectionSamplesMs.length,
        at_or_above_50_ms: slowestCorrectionSamples.filter((sample) => sample.latency_ms >= 50),
        slowest_samples: slowestCorrectionSamples,
      }
    : null;
const cpu = os.cpus()[0]?.model ?? "unknown";

console.log(`Module: ${modulePath} (${bytes.length.toLocaleString()} bytes)`);
console.log(`Runtime: Node ${process.version}, ${process.platform}/${process.arch}`);
console.log(`CPU: ${cpu}`);
console.log(
  `Fixture: ${fixtureShots.toLocaleString()} shots; selected ${shots.toLocaleString()} from ${options.start} with stride ${options.stride}`,
);
console.log(`Mode: ${options.mode}; warmup ${options.warmup}; repeats ${options.repeat}`);
if (options.mode === "stream" || options.mode === "correction") {
  console.log(`Detector rounds per shot: ${options.rounds}`);
}
console.log(`Compile + instantiate: ${compileMs.toFixed(3)} ms`);
console.log(`Decoder init: ${initMs.toFixed(3)} ms`);
const timingLabel = options.mode === "correction" ? "Mean final-round-to-correction" : "Mean replay";
console.log(`${timingLabel}: ${meanMs.toFixed(3)} ms (${perShotUs.toFixed(3)} us/shot)`);
if (correctionDistribution !== null) {
  console.log(
    `Correction latency: p50 ${correctionDistribution.p50_ms.toFixed(3)} ms; ` +
      `p95 ${correctionDistribution.p95_ms.toFixed(3)} ms; ` +
      `p99 ${correctionDistribution.p99_ms.toFixed(3)} ms; max ${correctionDistribution.max_ms.toFixed(3)} ms`,
  );
  console.log(
    `Under 50 ms: ${correctionDistribution.under_50_ms_count}/${correctionDistribution.sample_count} ` +
      `(${(100 * correctionDistribution.under_50_ms_fraction).toFixed(2)}%)`,
  );
}
console.log(`Throughput: ${perShotUs === 0 ? "n/a" : `${(1e6 / perShotUs).toFixed(2)} shots/s`}`);
console.log(`Logical errors: ${errors}/${shots} (${shots === 0 ? "n/a" : `${((100 * errors) / shots).toFixed(4)}%`})`);
if (typeof wasm.frontier_replay_checksum === "function") {
  console.log(`Prediction checksum: ${wasm.frontier_replay_checksum() >>> 0}`);
}

if (options.json) {
  const report = {
    schema_version: 2,
    module: modulePath,
    wasm_bytes: bytes.length,
    runtime: `Node ${process.version}`,
    platform: `${process.platform}/${process.arch}`,
    cpu,
    fixture_shots: fixtureShots,
    selected_start: options.start,
    selected_shots: shots,
    selected_stride: options.stride,
    selected_final_index: shots === 0 ? null : finalSelectedIndex,
    mode: options.mode,
    warmup_passes: options.warmup,
    timed_repeats: options.repeat,
    detector_rounds: options.mode === "stream" || options.mode === "correction" ? options.rounds : null,
    timing_scope: options.mode === "correction" ? "final detector round push through correction result" : "whole shot",
    compile_instantiate_ms: compileMs,
    decoder_init_ms: initMs,
    mean_replay_ms: meanMs,
    microseconds_per_shot: perShotUs,
    shots_per_second: perShotUs === 0 ? null : 1e6 / perShotUs,
    correction_latency_distribution: correctionDistribution,
    logical_errors: errors,
    logical_error_rate: shots === 0 ? null : errors / shots,
    prediction_checksum:
      typeof wasm.frontier_replay_checksum === "function" ? wasm.frontier_replay_checksum() >>> 0 : null,
  };
  await writeFile(options.json, `${JSON.stringify(report, null, 2)}\n`);
}
