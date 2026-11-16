// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { JSDOM, VirtualConsole } = require('jsdom');
const { test } = require('node:test');
const fixtures = process.env.PECOS_INSPECTOR_FIXTURES;
assert.ok(fixtures, 'Run npm test to export fresh Rust fixtures first');
const expectedCounts = {
  nested: [3, 14], dag: [10, 14], trace: [2, 4], 'trace-range': [1, 1],
  origins: [8, 8], empty: [0, 0], 'no-symbol': [1, 0], leakage: [3, 3],
};

for (const name of ['nested', 'dag', 'trace', 'trace-range', 'origins', 'empty', 'no-symbol', 'leakage']) {
  test(`${name}: selection, search, linking, escaping and no-script fallback`, t => {
    const html = fs.readFileSync(path.join(fixtures, name + '.html'), 'utf8');
    const errors = [];
    const log = new VirtualConsole();
    log.on('jsdomError', error => errors.push(error.message));
    const dom = new JSDOM(html, {runScripts: 'dangerously', virtualConsole: log});
    t.after(() => { try { assert.deepEqual(errors, []); } finally { dom.window.close(); } });
    const w = dom.window, d = w.document;
    const records = [...d.querySelectorAll('article')];
    const groups = [...d.querySelectorAll('[data-pecos-operation]')];
    assert.deepEqual([records.length, groups.length], expectedCounts[name]);
    assert.equal(d.querySelector('.controls').hidden, false);
    assert.equal(d.querySelector('.viewport-controls').hidden, false);
    const select = d.querySelector('select');
    const input = d.querySelector('input[type=search]');
    const button = action => d.querySelector(`[data-action="${action}"]`);
    const matching = id => groups.filter(g => g.dataset.pecosOperation === id);
    const chosen = () => [...d.querySelectorAll('article.selected')].map(r => records.indexOf(r));
    const filtered = () => [...select.options].slice(1).map(o => Number(o.value));
    function query(text) { input.value = text; input.dispatchEvent(new w.Event('input')); }
    function choose(index) { select.value = String(index); select.dispatchEvent(new w.Event('change')); }
    assert.deepEqual(errors, []);
    assert.equal(w.injected, undefined);
    assert.equal(button('next').disabled, !records.length);
    // Every operation remains navigable without a filter, including no-symbol records.
    for (let i = 0; i < records.length; i++) {
      if (name !== 'no-symbol') assert.ok(matching(records[i].dataset.pecosRecord).length > 0);
      button('next').click(); assert.deepEqual(chosen(), [i]);
      assert.deepEqual(groups.filter(g => g.classList.contains('selected')), matching(records[i].dataset.pecosRecord));
    }
    assert.equal(button('next').disabled, true);
    button('clear').click();
    query('Impossible-no-match ] [*');
    assert.deepEqual(filtered(), []);
    assert.equal(button('next').disabled, true);
    assert.equal(button('previous').disabled, true);
    assert.equal(select.disabled, true);
    assert.ok(groups.every(g => g.classList.contains('filtered-out')));
    for (const group of groups) {
      group.dispatchEvent(new w.MouseEvent('click'));
      group.dispatchEvent(new w.KeyboardEvent('keydown', {key:'Enter'}));
    }
    assert.deepEqual(chosen(), []);
    button('reset-filter').click();
    assert.equal(input.value, '');
    assert.equal(filtered().length, records.length);
    assert.ok(groups.every(g => !g.classList.contains('filtered-out')));
    if (name === 'dag') {
      assert.ok(records.every(record => record.dataset.pecosRecord.startsWith('node:')));
      query('CX q3');
      assert.equal(filtered().length, 1);
      const index = filtered()[0];
      button('next').click();
      assert.deepEqual(chosen(), [index]);
      assert.equal(matching(records[index].dataset.pecosRecord).length, 3);
      query('m7');
      assert.equal(filtered().length, 1);
      assert.deepEqual(chosen(), []);
    }
    if (name === 'nested') {
      const batch = matching(records[1].dataset.pecosRecord);
      assert.equal(batch[0].getAttribute('role'), 'button');
      assert.equal(batch[0].getAttribute('tabindex'), '0');
      for (const key of ['Enter', ' ']) {
        const event = new w.KeyboardEvent('keydown', {key, cancelable: true});
        batch[0].dispatchEvent(event);
        assert.equal(event.defaultPrevented, true);
        assert.deepEqual(chosen(), [1]);
        assert.equal(batch[0].getAttribute('aria-pressed'), 'true');
        assert.ok(batch.every(group => group.classList.contains('selected')));
        button('clear').click();
        assert.equal(batch[0].getAttribute('aria-pressed'), 'false');
      }
      batch[2].dispatchEvent(new w.MouseEvent('mouseenter'));
      assert.ok(batch.every(group => group.classList.contains('hovered')));
      batch[2].dispatchEvent(new w.MouseEvent('mouseleave'));
      assert.ok(batch.every(group => !group.classList.contains('hovered')));
      query('  cX   Q3  '); assert.deepEqual(filtered(), [1]);
      button('next').click(); assert.deepEqual(chosen(), [1]);
      assert.equal(button('previous').disabled, true);
      assert.equal(button('next').disabled, true);
      assert.equal(matching(records[1].dataset.pecosRecord).length, 6);
      matching(records[0].dataset.pecosRecord)[0].dispatchEvent(new w.MouseEvent('click'));
      assert.deepEqual(chosen(), [1]);
      query('m7'); assert.deepEqual(filtered(), [2]); assert.deepEqual(chosen(), []);
      button('next').click(); assert.deepEqual(chosen(), [2]);
      query('quoted'); assert.deepEqual(filtered(), [1]);
      query('<script>'); assert.deepEqual(filtered(), [1]);
      query('q0'); button('next').click(); button('next').click(); button('previous').click();
      assert.deepEqual(chosen(), [0]);
    }
    if (name === 'trace') {
      choose(0); assert.equal(d.querySelectorAll('[data-pecos-operation].related').length, 1);
      assert.match(d.querySelector('.selection-status').textContent, /1 other stored operation/);
      query('RZZ'); assert.deepEqual(filtered(), [0]); assert.deepEqual(chosen(), [0]);
      const other = matching(records[1].dataset.pecosRecord)[0];
      assert.equal(other.classList.contains('related'), true);
      assert.equal(other.classList.contains('filtered-out'), true);
      assert.equal(w.getComputedStyle(other).opacity, '1');
      assert.equal(other.getAttribute('tabindex'), '-1');
      assert.match(d.querySelector('.selection-status').textContent, /1 are outside the search/);
      button('link-trace').click(); assert.equal(other.classList.contains('related'), false);
      assert.equal(w.getComputedStyle(other).opacity, '0.2');
      button('link-trace').click(); assert.equal(other.classList.contains('related'), true);
      button('clear').click(); assert.equal(d.querySelectorAll('.related').length, 0);
      button('reset-filter').click(); choose(1);
      assert.equal(d.querySelectorAll('[data-pecos-operation].related').length, 3);
      assert.equal(records[0].dataset.pecosTraceOrigin, 'qis:18446744073709551615:0:0:0');
    }
    if (name === 'trace-range') {
      choose(0); assert.equal(button('link-trace').disabled, true);
      assert.match(d.querySelector('.selection-status').textContent, /0 other stored operation/);
      assert.equal(d.querySelectorAll('.related').length, 0);
    }
    if (name === 'origins') {
      choose(0); assert.equal(d.querySelectorAll('[data-pecos-operation].related').length, 1);
      for (let i = 2; i < records.length; i++) {
        choose(i); assert.equal(d.querySelectorAll('[data-pecos-operation].related').length, 0);
      }
      assert.notEqual(records[0].dataset.pecosTraceOrigin, records[2].dataset.pecosTraceOrigin);
      assert.equal(records[6].dataset.pecosTraceOrigin, undefined);
      assert.equal(records[7].dataset.pecosTraceOrigin, undefined);
    }
    if (name === 'no-symbol') {
      query('Crosstalk'); button('next').click(); assert.deepEqual(chosen(), [0]);
      assert.match(d.querySelector('.selection-status').textContent, /no visible symbol/);
    }
    if (name === 'leakage') {
      query('MZ_L m7'); assert.deepEqual(filtered(), [1]);
      query('MeasureLeaked'); assert.deepEqual(filtered(), [1]);
    }
    const fallback = new JSDOM(html);
    t.after(() => fallback.window.close());
    assert.equal(fallback.window.document.querySelector('.controls').hidden, true);
    assert.equal(fallback.window.document.querySelector('.viewport-controls').hidden, true);
    for (const record of fallback.window.document.querySelectorAll('article')) {
      assert.equal(fallback.window.getComputedStyle(record).display, 'block');
    }
    assert.deepEqual(errors, []);
  });
}
