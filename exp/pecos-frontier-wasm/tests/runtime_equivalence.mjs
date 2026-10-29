// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0
// Correctness-only differential fixture; no latency or speed measurements.
import assert from "node:assert/strict";
import { readFile, writeFile } from "node:fs/promises";

const args = process.argv.slice(2);
if (args[0] === "--write-model" && [2, 3].includes(args.length)) {
  // Three detector words at the WASM ABI, two physical u64 detector words,
  // and optionally multiple logical words. Long edges retain competing states across rounds.
  const observables = Number(args[2] ?? 1);
  assert.ok(observables === 1 || observables === 65);
  const lines = [];
  for (let i = 0; i < 96; i++) {
    lines.push(`error(0.03) D${i} L${i % observables}`);
    if (i < 64) lines.push(`error(0.07) D${i} D${i + 32} L${(i + 1) % observables}`);
    if (i < 32) lines.push(`error(0.11) D${i} D${i + 64} L${(i + 2) % observables}`);
  }
  await writeFile(args[1], lines.join("\n") + "\n");
  process.exit(0);
}
if (args.length !== 2) {
  throw new Error("Usage: node runtime_equivalence.mjs --write-model MODEL.dem [1|65] | BASELINE.wasm CANDIDATE.wasm");
}
const modules = [];
for (const path of args) {
  const { instance } = await WebAssembly.instantiate(await readFile(path), {});
  modules.push(instance.exports);
}
function result(wasm) {
  return [wasm.frontier_status_i64(), ...[0, 1, 2, 3].map(i => wasm[`frontier_result_${i}_i64`]())];
}
function decode(wasm, words, mode, delayed) {
  // Reuse the decoder across shots, then exercise reset and mode changes.
  if (mode === "fixed") assert.equal(wasm.frontier_stream_fixed_lag_i64(3n, 2n), 0n);
  else assert.equal(wasm.frontier_stream_dependency_driven_i64(), 0n);
  if (mode === "batch") {
    wasm.frontier_decode(...words.map(w => w | 0), 0);
    return [result(wasm)];
  }
  wasm.frontier_stream_begin();
  const trace = [];
  if (delayed) {
    // Dependency-driven allows a complete syndrome in the terminal call.
    trace.push(wasm.frontier_stream_finish_round_i64(96n, ...words.map(BigInt), 0n));
    trace.push(result(wasm));
  } else {
    for (let round = 0; round < 3; round++) {
      const fn = round === 2 ? "frontier_stream_finish_round_i64" : "frontier_stream_push_i64";
      trace.push(wasm[fn](32n, BigInt(words[round]), 0n, 0n, 0n));
      trace.push(result(wasm));
    }
  }
  return trace;
}
for (const wasm of modules) {
  wasm.init();
  assert.equal(wasm.frontier_status_i64(), 0n);
}
let seed = 0x61c0ffee;
function randomWord() {
  seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
  return seed;
}
let comparisons = 0;
const shots = [[0, 0, 0], [0xffffffff, 0xffffffff, 0xffffffff]];
for (const bit of [0, 31, 32, 63, 64, 95]) {
  const words = [0, 0, 0];
  words[Math.floor(bit / 32)] = (2 ** (bit % 32)) >>> 0;
  shots.push(words);
}
for (let shot = 0; shot < 64; shot++) shots.push([randomWord(), randomWord(), randomWord()]);
for (const words of shots) {
  for (const [mode, delayed] of [["batch", false], ["dependency", false], ["dependency", true], ["fixed", false]]) {
    const expected = decode(modules[0], words, mode, delayed);
    const actual = decode(modules[1], words, mode, delayed);
    assert.deepEqual(actual, expected, `${mode}/${delayed}: ${words}`);
    // Every syndrome is supported by the single-detector mechanisms.
    assert.equal(actual.at(-1)[0], 0n, `${mode}/${delayed} fixture must decode successfully: ${words}`);
    comparisons++;
  }
}
console.log(`PASS: ${comparisons} baseline/candidate comparisons, including status, progress and all 128 correction bits`);
