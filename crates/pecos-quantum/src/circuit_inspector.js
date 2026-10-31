/* Copyright 2026 The PECOS Developers. Licensed under Apache-2.0. */
(() => {
  "use strict";
  const root = document.currentScript.closest(".pecos-inspector");
  const records = Array.from(root.querySelectorAll("[data-pecos-record]"));
  const select = root.querySelector("select");
  const search = root.querySelector('input[type="search"]');
  const filterStatus = root.querySelector(".filter-status");
  const status = root.querySelector(".selection-status");
  const previous = root.querySelector('[data-action="previous"]');
  const next = root.querySelector('[data-action="next"]');
  const clear = root.querySelector('[data-action="clear"]');
  const reset = root.querySelector('[data-action="reset-filter"]');
  const linkTrace = root.querySelector('[data-action="link-trace"]');
  const viewport = root.querySelector(".diagram");
  const svg = viewport.querySelector("svg");
  const zoomIn = root.querySelector('[data-action="zoom-in"]');
  const zoomOut = root.querySelector('[data-action="zoom-out"]');
  const zoomLevel = root.querySelector('[aria-label="Zoom level"]');
  const showSelection = root.querySelector('[data-action="show-selection"]');
  const traceControls = root.querySelector(".trace-controls");
  const previousStage = root.querySelector('[data-action="previous-stage"]');
  const nextStage = root.querySelector('[data-action="next-stage"]');
  const labels = Array.from(select.options).slice(1).map(option => option.textContent);
  const groups = new Map(records.map(record => [record.dataset.pecosRecord, []]));
  const origins = new Map();
  records.forEach((record, index) => {
    const origin = record.dataset.pecosTraceOrigin;
    if (origin) {
      if (!origins.has(origin)) origins.set(origin, []);
      origins.get(origin).push(index);
    }
  });
  for (const component of root.querySelectorAll("[data-pecos-operation]")) {
    groups.get(component.dataset.pecosOperation)?.push(component);
  }
  // Include displayed symbols (e.g. MZ_L) as well as canonical gate names.
  // Spaces between fields prevent accidental matches spanning adjacent elements.
  const searchable = records.map((record, index) => [labels[index], record.dataset.pecosSearchIds || "",
    ...Array.from(record.querySelectorAll("h2, dt, dd")).map(item => item.textContent),
    ...groups.get(record.dataset.pecosRecord).map(component => component.textContent)
  ].join(" ").toLowerCase());
  let selected = -1;
  let matches = records.map((_, index) => index);
  let allowed = new Set(matches);
  let highlighted = [];
  const components = index => groups.get(records[index].dataset.pecosRecord);
  const baseWidth = Number(svg.getAttribute("width"));
  const baseHeight = Number(svg.getAttribute("height"));
  let zoom = 1;
  function scrollDiagram(left, top) {
    // Scroll only the diagram viewport, never the containing notebook or page.
    viewport.scrollLeft = Math.max(0, Math.min(left, viewport.scrollWidth - viewport.clientWidth));
    viewport.scrollTop = Math.max(0, Math.min(top, viewport.scrollHeight - viewport.clientHeight));
  }
  function revealSelection() {
    if (selected < 0 || !components(selected).length || viewport.clientWidth <= 0 || viewport.clientHeight <= 0) return;
    // A batch can contain many instances: accumulate without spreading an
    // unbounded array into a function's argument list.
    let left = Infinity, right = -Infinity, top = Infinity, bottom = -Infinity;
    for (const component of components(selected)) {
      const box = component.getBoundingClientRect();
      left = Math.min(left, box.left);
      right = Math.max(right, box.right);
      top = Math.min(top, box.top);
      bottom = Math.max(bottom, box.bottom);
    }
    const bounds = viewport.getBoundingClientRect();
    const x = bounds.left + viewport.clientLeft;
    const y = bounds.top + viewport.clientTop;
    const margin = 12;
    const delta = (start, end, low, high) => {
      if (end - start > high - low || start < low) return start - low;
      return end > high ? end - high : 0;
    };
    scrollDiagram(viewport.scrollLeft + delta(left, right, x + margin, x + viewport.clientWidth - margin),
      viewport.scrollTop + delta(top, bottom, y + margin, y + viewport.clientHeight - margin));
  }
  function setZoom(value) {
    if (!(baseWidth > 0 && baseHeight > 0 && Number.isFinite(value))) return;
    const style = getComputedStyle(viewport);
    const paddingLeft = parseFloat(style.paddingLeft) || 0;
    const paddingTop = parseFloat(style.paddingTop) || 0;
    const centerX = (viewport.scrollLeft + viewport.clientWidth / 2 - paddingLeft) / zoom;
    const centerY = (viewport.scrollTop + viewport.clientHeight / 2 - paddingTop) / zoom;
    zoom = Math.max(0.01, Math.min(value, 4));
    svg.style.width = `${baseWidth * zoom}px`;
    svg.style.height = `${baseHeight * zoom}px`;
    zoomLevel.textContent = `${Number((zoom * 100).toFixed(1))}%`;
    zoomIn.disabled = zoom >= 4;
    zoomOut.disabled = zoom <= 0.01;
    scrollDiagram(centerX * zoom + paddingLeft - viewport.clientWidth / 2,
      centerY * zoom + paddingTop - viewport.clientHeight / 2);
  }
  function stageIndices() {
    return selected < 0 ? [] : origins.get(records[selected].dataset.pecosTraceOrigin) || [];
  }
  function choose(index, reveal = false) {
    if (index >= 0 && !allowed.has(index)) return;
    if (selected >= 0) {
      records[selected].classList.remove("selected");
      components(selected)[0]?.setAttribute("aria-pressed", "false");
    }
    for (const component of highlighted) component.classList.remove("selected", "related");
    highlighted = [];
    selected = index;
    select.value = index < 0 ? "" : String(index);
    if (index >= 0) {
      records[index].classList.add("selected");
      for (const component of components(index)) {
        component.classList.add("selected");
        highlighted.push(component);
      }
      components(index)[0]?.setAttribute("aria-pressed", "true");
      status.textContent = components(index).length
        ? `${labels[index]} — all components of this stored operation are highlighted.`
        : `${labels[index]} — this operation has no visible symbol in the diagram.`;
      const origin = records[index].dataset.pecosTraceOrigin;
      if (origin && linkTrace.checked) {
        const related = origins.get(origin).filter(other => other !== index);
        for (const other of related) {
          for (const component of components(other)) {
            component.classList.add("related");
            highlighted.push(component);
          }
        }
        const excluded = related.filter(other => !allowed.has(other)).length;
        status.textContent += ` ${related.length} other stored operation(s) in this view share the recorded trace origin.`;
        if (excluded) status.textContent += ` ${excluded} are outside the search matches and highlighted for context.`;
      }
    } else {
      status.textContent = !records.length ? "Empty inspection." : !matches.length
        ? "No operations match this search. Reset search to see all operations."
        : "Select an operation to see its details.";
    }
    const position = matches.indexOf(index);
    previous.disabled = position <= 0;
    next.disabled = position >= matches.length - 1;
    clear.disabled = index < 0;
    showSelection.disabled = index < 0 || !components(index).length;
    const stages = stageIndices();
    const stage = stages.indexOf(index);
    traceControls.hidden = !stages.length;
    previousStage.disabled = stage <= 0;
    nextStage.disabled = stage >= stages.length - 1;
    root.querySelector(".stage-position").textContent = stages.length
      ? `Trace stage ${stage + 1} of ${stages.length} in this view (stored operation order).` : "";
    const previousExcluded = stage > 0 && !allowed.has(stages[stage - 1]);
    const nextExcluded = stage >= 0 && stage < stages.length - 1 && !allowed.has(stages[stage + 1]);
    previousStage.textContent = previousExcluded ? "Previous trace stage (resets search)" : "Previous trace stage";
    nextStage.textContent = nextExcluded ? "Next trace stage (resets search)" : "Next trace stage";
    root.querySelector(".stage-hint").textContent = previousExcluded || nextExcluded
      ? "Moving to a stage outside the search clears the search." : "";
    if (reveal) revealSelection();
  }
  function filter() {
    const terms = search.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    matches = records.map((_, index) => index).filter(index => terms.every(term => searchable[index].includes(term)));
    allowed = new Set(matches);
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = matches.length ? "Choose an operation" : "No matching operations";
    select.replaceChildren(placeholder);
    for (const index of matches) {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = labels[index];
      select.append(option);
    }
    records.forEach((_, index) => {
      for (const component of components(index)) {
        component.classList.toggle("filtered-out", !allowed.has(index));
        component.classList.remove("hovered");
      }
      components(index)[0]?.setAttribute("tabindex", allowed.has(index) ? "0" : "-1");
    });
    select.disabled = matches.length === 0;
    reset.disabled = search.value.length === 0;
    filterStatus.textContent = `${matches.length} of ${records.length} stored operations match.`;
    choose(allowed.has(selected) ? selected : -1);
  }
  records.forEach((record, index) => {
    const matching = components(index);
    matching.forEach(component => {
      component.addEventListener("click", () => choose(index));
      component.addEventListener("mouseenter", () => {
        if (allowed.has(index)) matching.forEach(item => item.classList.add("hovered"));
      });
      component.addEventListener("mouseleave", () => matching.forEach(item => item.classList.remove("hovered")));
    });
    if (matching.length) {
      const first = matching[0];
      first.setAttribute("role", "button");
      first.setAttribute("aria-label", labels[index]);
      first.setAttribute("aria-pressed", "false");
      first.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          choose(index);
        }
      });
    }
  });
  select.addEventListener("change", () => choose(select.value === "" ? -1 : Number(select.value), true));
  search.addEventListener("input", filter);
  reset.addEventListener("click", () => { search.value = ""; filter(); search.focus(); });
  previous.addEventListener("click", () => {
    const position = matches.indexOf(selected);
    if (position > 0) choose(matches[position - 1], true);
  });
  next.addEventListener("click", () => {
    const position = matches.indexOf(selected);
    if (position < matches.length - 1) choose(matches[position + 1], true);
  });
  clear.addEventListener("click", () => choose(-1));
  linkTrace.addEventListener("change", () => choose(selected));
  function moveStage(step) {
    const stages = stageIndices();
    const target = stages[stages.indexOf(selected) + step];
    if (target === undefined) return;
    if (!allowed.has(target)) { search.value = ""; filter(); }
    choose(target, true);
  }
  previousStage.addEventListener("click", () => moveStage(-1));
  nextStage.addEventListener("click", () => moveStage(1));
  showSelection.addEventListener("click", revealSelection);
  zoomIn.addEventListener("click", () => setZoom(zoom * 1.25));
  zoomOut.addEventListener("click", () => setZoom(zoom / 1.25));
  root.querySelector('[data-action="actual-size"]').addEventListener("click", () => setZoom(1));
  root.querySelector('[data-action="fit-width"]').addEventListener("click", () => {
    const style = getComputedStyle(viewport);
    const availableWidth = () => viewport.clientWidth - (parseFloat(style.paddingLeft) || 0) - (parseFloat(style.paddingRight) || 0);
    const width = availableWidth();
    if (width <= 0) return;
    setZoom(width / baseWidth);
    // Enlarging a short diagram can introduce a vertical scrollbar. Account
    // for its width, without oscillating if shrinking removes it again.
    const narrowedWidth = availableWidth();
    if (narrowedWidth > 0 && narrowedWidth < width) setZoom(narrowedWidth / baseWidth);
    scrollDiagram(0, viewport.scrollTop);
  });
  linkTrace.disabled = !Array.from(origins.values()).some(indices => indices.length > 1);
  root.classList.add("interactive");
  root.querySelector(".controls").hidden = false;
  root.querySelector(".viewport-controls").hidden = false;
  filter();
})();
