// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {JSDOM, VirtualConsole} = require('jsdom');
const { test } = require('node:test');
const fixtures = process.env.PECOS_INSPECTOR_FIXTURES;
assert.ok(fixtures, 'Run npm test to export fresh Rust fixtures first');

function load(t, name) {
  const errors = [];
  const log = new VirtualConsole();
  log.on('jsdomError', error => errors.push(error.message));
  const dom = new JSDOM(fs.readFileSync(path.join(fixtures, name + '.html'), 'utf8'), {runScripts:'dangerously', virtualConsole:log});
  t.after(() => { try { assert.deepEqual(errors, []); } finally { dom.window.close(); } });
  const w = dom.window, d = w.document;
  const button = action => d.querySelector(`[data-action="${action}"]`);
  const select = d.querySelector('select');
  const search = d.querySelector('input[type=search]');
  const chosen = () => select.value;
  const choose = value => { select.value = String(value); select.dispatchEvent(new w.Event('change')); };
  const query = value => { search.value = value; search.dispatchEvent(new w.Event('input')); };
  assert.deepEqual(errors, []);
  return {dom, w, d, button, choose, query, chosen, errors};
}

// Same-origin stage traversal uses view order, independent of the highlight toggle.
test('trace stages preserve matching searches and explicitly reset excluded ones', t => {
  const {dom,w,d,button,choose,query,chosen,errors} = load(t, 'trace');
  assert.equal(d.querySelector('.trace-controls').hidden, true);
  choose(0);
  assert.match(d.querySelector('.stage-position').textContent, /1 of 2 in this view/);
  assert.equal(button('previous-stage').disabled, true);
  query('RZZ');
  assert.match(button('next-stage').textContent, /resets search/);
  button('link-trace').click();
  button('next-stage').click();
  assert.equal(chosen(), '1');
  assert.equal(d.querySelector('input[type=search]').value, '');
  assert.equal(d.querySelectorAll('[data-pecos-operation].related').length, 0);
  assert.match(d.querySelector('.stage-position').textContent, /2 of 2 in this view/);
  assert.equal(button('next-stage').disabled, true);
  query('q1'); // Both stages match: traversal must preserve this query.
  button('previous-stage').click();
  assert.equal(chosen(), '0');
  assert.equal(d.querySelector('input[type=search]').value, 'q1');
  button('clear').click();
  assert.equal(d.querySelector('.trace-controls').hidden, true);
  assert.equal(button('show-selection').disabled, true);
  assert.deepEqual(errors, []);
});
for (const name of ['trace-range', 'empty', 'no-symbol']) {
  test(`${name}: safe stage and unmeasured viewport navigation`, t => {
    const {dom,d,button,choose,errors} = load(t, name);
    if (name !== 'empty') choose(0);
    assert.equal(button('previous-stage').disabled, true);
    assert.equal(button('next-stage').disabled, true);
    if (name === 'trace-range') assert.match(d.querySelector('.stage-position').textContent, /1 of 1 in this view/);
    else assert.equal(button('show-selection').disabled, true);
    button('fit-width').click(); // Hidden/unmeasured viewport must not force zero scale.
    assert.equal(d.querySelector('[aria-label="Zoom level"]').textContent, '100%');
    assert.deepEqual(errors, []);
  });
}

// jsdom has no layout engine. Supply explicit measured boxes and viewport
// dimensions to verify the geometry algorithm with independently derived values.
test('viewport scrolling and zoom preserve anchors and enforce bounds', t => {
  const {dom,w,d,button,choose,errors} = load(t, 'wide');
  const pane = d.querySelector('.diagram'), svg = pane.querySelector('svg');
  const width = Number(svg.getAttribute('width')), height = Number(svg.getAttribute('height'));
  Object.defineProperties(pane, {
    clientWidth: {value: 432}, clientHeight: {value: 232}, clientLeft: {value: 1}, clientTop: {value: 1},
    scrollWidth: {get: () => Math.max(432, parseFloat(svg.style.width || width) + 32)},
    scrollHeight: {get: () => Math.max(232, parseFloat(svg.style.height || height) + 32)},
  });
  pane.getBoundingClientRect = () => ({left:100,top:50,right:534,bottom:284});
  const record = d.querySelector('article');
  const first = d.querySelector(`[data-pecos-operation="${record.dataset.pecosRecord}"]`);
  // Operation in diagram content coordinates x=900..930, y=300..320.
  first.getBoundingClientRect = () => ({
    left:101 + 900 - pane.scrollLeft, right:101 + 930 - pane.scrollLeft,
    top:51 + 300 - pane.scrollTop, bottom:51 + 320 - pane.scrollTop,
  });
  choose(0);
  // Right edge must become viewport inner right (101+432) minus 12 margin.
  assert.equal(pane.scrollLeft, 510);
  assert.equal(pane.scrollTop, Math.min(100, pane.scrollHeight - 232));
  assert.equal(d.body.scrollTop, 0);
  const saved = pane.scrollLeft;
  button('show-selection').click(); assert.equal(pane.scrollLeft, saved);
  pane.scrollLeft = 0;
  first.dispatchEvent(new w.MouseEvent('click'));
  assert.equal(pane.scrollLeft, 0, 'clicking visible glyph must not auto-pan');
  button('show-selection').click(); assert.equal(pane.scrollLeft, 510);

  // Zoom from 100% to 125%, preserving the viewport center relative to padding.
  pane.scrollLeft = 400; pane.scrollTop = 0;
  button('zoom-in').click();
  assert.equal(parseFloat(svg.style.width), width * 1.25);
  assert.equal(parseFloat(svg.style.height), height * 1.25);
  assert.equal(pane.scrollLeft, 550); // (400 + 216 - 16) * 1.25 + 16 - 216
  assert.equal(d.querySelector('[aria-label="Zoom level"]').textContent, '125%');
  button('zoom-out').click(); assert.equal(pane.scrollLeft, 400);
  button('fit-width').click();
  assert.equal(parseFloat(svg.style.width), 400); // client width minus two 16px paddings
  assert.equal(pane.scrollLeft, 0);
  button('actual-size').click(); assert.equal(parseFloat(svg.style.width), width);
  for (let i=0;i<50;i++) button('zoom-in').click();
  assert.equal(parseFloat(svg.style.width), width*4);
  assert.equal(button('zoom-in').disabled, true);
  for (let i=0;i<60;i++) button('zoom-out').click();
  assert.equal(parseFloat(svg.style.width), width*0.01);
  assert.equal(button('zoom-out').disabled, true);
  assert.equal(d.querySelector('[aria-label="Zoom level"]').textContent, '1%');
  assert.deepEqual(errors, []);
});

// Oversized multi-component batches align their combined upper-left extent.
test('oversized batches reveal the combined upper-left extent', t => {
  const {dom,d,button,choose,errors} = load(t, 'nested');
  const pane = d.querySelector('.diagram');
  Object.defineProperties(pane, {
    clientWidth:{value:200}, clientHeight:{value:100}, clientLeft:{value:0}, clientTop:{value:0},
    scrollWidth:{value:2000}, scrollHeight:{value:1000},
  });
  pane.getBoundingClientRect = () => ({left:0,top:0,right:200,bottom:100});
  const groups = [...d.querySelectorAll('[data-pecos-operation="tick:1:batch:0"]')];
  groups.forEach((group,i) => group.getBoundingClientRect = () => ({
    left:500+i*40-pane.scrollLeft, right:530+i*40-pane.scrollLeft,
    top:300+i*40-pane.scrollTop, bottom:330+i*40-pane.scrollTop,
  }));
  choose(1); assert.equal(pane.scrollLeft, 488); assert.equal(pane.scrollTop, 288);
  button('show-selection').click(); assert.equal(pane.scrollLeft, 488); assert.equal(pane.scrollTop, 288);
  assert.deepEqual(errors, []);
});
// Classic (non-overlay) scrollbar appears only after fitting enlarges the SVG.
test('fit width remeasures when a vertical scrollbar appears', t => {
  const {dom,d,button,errors} = load(t, 'trace');
  const pane = d.querySelector('.diagram'), svg = pane.querySelector('svg');
  const baseWidth = Number(svg.getAttribute('width'));
  const baseHeight = Number(svg.getAttribute('height'));
  // Cap at a height between the original and first fit heights, so fit adds a bar.
  const cap = (baseHeight + (468 / baseWidth) * baseHeight) / 2;
  assert.ok(468 > baseWidth);
  Object.defineProperties(pane, {
    clientWidth: {get: () => parseFloat(svg.style.height || baseHeight) + 32 > cap + 32 ? 485 : 500},
    clientHeight: {value: cap + 32},
    scrollWidth: {get: () => Math.max(pane.clientWidth, parseFloat(svg.style.width || baseWidth) + 32)},
    scrollHeight: {get: () => Math.max(cap+32, parseFloat(svg.style.height || baseHeight)+32)},
  });
  button('fit-width').click();
  assert.equal(parseFloat(svg.style.width), 453);
  assert.ok(pane.scrollWidth <= pane.clientWidth);
  assert.deepEqual(errors, []);
});
