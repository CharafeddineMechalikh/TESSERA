(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const body = document.body;
  const SVG_NS = "http://www.w3.org/2000/svg";

  const state = {
    limits: null,
    authRequired: body.dataset.authRequired === "true",
    refreshing: false,
    run: null,
    summary: null,
    summaryFor: null,
    aggregate: null,
    aggregateRequest: null,
    lastSeq: 0,
    stream: null,
    netCount: 0,
    activeTab: "pipeline",
    activeView: "benchmark",
    startedAt: null,
    selectedUnit: 0,
    elapsedTimer: null,
    pollTimer: null,
    reconnectTimer: null,
  };

  // ---------------------------------------------------------------- utils
  function fmtError(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
    const number = Number(value);
    if (number === 0) return "0";
    const magnitude = Math.abs(number);
    if (magnitude < 1e-3 || magnitude >= 1e4) return number.toExponential(2);
    if (magnitude < 1) return number.toFixed(5);
    return number.toFixed(3);
  }

  function fmtSeconds(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
    const number = Number(value);
    if (number < 60) return `${number.toFixed(1)}s`;
    const minutes = Math.floor(number / 60);
    const seconds = Math.round(number - minutes * 60);
    return `${minutes}m ${seconds}s`;
  }

  function fmtElapsed(startedIso) {
    if (!startedIso) return "--";
    const elapsed = Math.max(0, (Date.now() - Date.parse(startedIso)) / 1000);
    return fmtClock(elapsed);
  }

  function fmtClock(seconds) {
    const total = Math.max(0, Math.floor(seconds));
    const minutes = String(Math.floor(total / 60)).padStart(2, "0");
    const secs = String(total % 60).padStart(2, "0");
    return `${minutes}:${secs}`;
  }

  function fmtInt(value) {
    if (value === null || value === undefined) return "--";
    return Number(value).toLocaleString("en-US");
  }

  function fmtTime(iso) {
    if (!iso) return "";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString("en-US", { hour12: false });
  }

  async function api(path, options = {}) {
    const config = {
      method: options.method || "GET",
      credentials: "same-origin",
      headers: { ...(options.headers || {}) },
    };
    if (options.body !== undefined) {
      config.headers["Content-Type"] = "application/json";
      config.body = JSON.stringify(options.body);
    }
    const response = await fetch(path, config);
    const text = await response.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch { data = { detail: text }; }
    }
    if (!response.ok) {
      const error = new Error((data && data.detail) || `HTTP ${response.status}`);
      error.status = response.status;
      throw error;
    }
    return data;
  }

  function showBanner(message, kind = "error") {
    const banner = $("banner");
    banner.textContent = message;
    banner.className = kind === "info" ? "banner info" : "banner";
    banner.hidden = false;
  }

  function hideBanner() {
    $("banner").hidden = true;
  }

  function setStatusPill(status) {
    const pill = $("serverState");
    const known = ["idle", "running", "completed", "failed", "cancelled"];
    const normalized = known.includes(status) ? status : "idle";
    pill.className = `pill pill-${normalized === "completed" ? "done" : normalized}`;
    pill.textContent = normalized;
  }

  // ---------------------------------------------------------------- logs
  function addLog(target, kind, message, source) {
    const container = $(target);
    const line = document.createElement("div");
    line.className = `log-line ${source && source.level ? source.level : ""}`;
    const time = document.createElement("span");
    time.className = "log-time";
    time.textContent = fmtTime(source && source.ts);
    const label = document.createElement("span");
    label.className = "log-kind";
    label.textContent = kind;
    const text = document.createElement("span");
    text.className = "log-msg";
    text.textContent = message;
    line.append(time, label, text);
    const nearBottom = container.scrollHeight - container.scrollTop - container.clientHeight < 40;
    container.appendChild(line);
    while (container.childElementCount > 1200) container.removeChild(container.firstChild);
    if (nearBottom) container.scrollTop = container.scrollHeight;
  }

  function addPipeline(kind, message, source) {
    addLog("pipelineLog", kind, message, source);
  }

  function addNetwork(message, source) {
    state.netCount += 1;
    $("netCount").textContent = String(state.netCount);
    addLog("networkLog", "net", message, source);
  }

  function clearLogs() {
    $("pipelineLog").replaceChildren();
    $("networkLog").replaceChildren();
    state.netCount = 0;
    $("netCount").textContent = "0";
  }

  function switchTab(name) {
    state.activeTab = name;
    $("tabPipeline").classList.toggle("active", name === "pipeline");
    $("tabNetwork").classList.toggle("active", name === "network");
    $("pipelineLog").hidden = name !== "pipeline";
    $("networkLog").hidden = name !== "network";
  }

  // ---------------------------------------------------------------- views
  const VIEWS = ["benchmark", "generation", "theory"];

  function switchView(name) {
    if (!VIEWS.includes(name)) return;
    state.activeView = name;
    VIEWS.forEach((view) => {
      const panel = $(`tab${view[0].toUpperCase()}${view.slice(1)}`);
      if (panel) panel.classList.toggle("active", view === name);
    });
    document.querySelectorAll(".nav-button[data-view]").forEach((button) => {
      button.classList.toggle("active", button.dataset.view === name);
    });
    try {
      window.sessionStorage.setItem("tessera.view", name);
    } catch (error) {
      // A blocked sessionStorage only costs the remembered view, not the switch itself.
    }
  }

  function restoreView() {
    let saved = null;
    try {
      saved = window.sessionStorage.getItem("tessera.view");
    } catch (error) {
      saved = null;
    }
    switchView(VIEWS.includes(saved) ? saved : "benchmark");
  }

  // ---------------------------------------------------------------- progress
  function setProgress(percent, text) {
    const bounded = Math.max(0, Math.min(100, percent));
    $("progressBar").style.width = `${bounded}%`;
    $("progressPct").textContent = `${Math.round(bounded)}%`;
    if (text) $("progressText").textContent = text;
  }

  function setPhase(text) {
    $("metaPhase").textContent = text;
  }

  function renderRunMeta() {
    const run = state.run;
    if (!run) {
      $("metaRun").textContent = "none";
      $("metaWorkers").textContent = "--";
      $("metaTokens").textContent = "--";
      $("metaElapsed").textContent = "--";
      return;
    }
    $("metaRun").textContent = run.run_id;
    const config = run.config || {};
    $("metaWorkers").textContent = config.workers ? `K=${config.workers}` : "--";
    $("metaTokens").textContent = config.max_tokens ? `<= ${config.max_tokens}` : "--";
    const duration = state.summary ? Number(state.summary.duration_s) : NaN;
    const finished =
      run.status !== "running" &&
      state.summary &&
      state.summary.run_id === run.run_id &&
      state.summary.duration_s != null &&
      Number.isFinite(duration);
    $("metaElapsed").textContent = finished
      ? fmtClock(duration)
      : fmtElapsed(run.started_utc || state.startedAt);
  }

  // ---------------------------------------------------------------- charts
  function svgEl(tag, attrs = {}, text = null) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
      node.setAttribute(key, String(value));
    }
    if (text !== null) node.textContent = text;
    return node;
  }

  function chartFrame(svg, title) {
    svg.replaceChildren();
    const width = 420;
    const height = 240;
    svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    const margin = { left: 48, right: 16, top: 18, bottom: 30 };
    const plot = {
      left: margin.left,
      top: margin.top,
      right: width - margin.right,
      bottom: height - margin.bottom,
      width: width - margin.left - margin.right,
      height: height - margin.top - margin.bottom,
    };
    svg.appendChild(
      svgEl("line", {
        x1: plot.left, y1: plot.bottom, x2: plot.right, y2: plot.bottom,
        class: "axis-line",
      })
    );
    svg.appendChild(
      svgEl("line", {
        x1: plot.left, y1: plot.top, x2: plot.left, y2: plot.bottom,
        class: "axis-line",
      })
    );
    return { svg, plot, title };
  }

  function emptyChart(svg, message) {
    svg.replaceChildren();
    svg.appendChild(
      svgEl("text", { x: 210, y: 120, "text-anchor": "middle", class: "empty" }, message)
    );
  }

  function yAxis(svg, plot, yFor, ticks = 4) {
    for (let index = 0; index <= ticks; index += 1) {
      const fraction = index / ticks;
      const y = plot.bottom - fraction * plot.height;
      svg.appendChild(
        svgEl("line", {
          x1: plot.left, y1: y, x2: plot.right, y2: y, class: "grid-line",
        })
      );
      svg.appendChild(
        svgEl("text", {
          x: plot.left - 6, y: y + 3, "text-anchor": "end",
        }, yFor(fraction))
      );
    }
  }

  function renderLoadChart(positions) {
    const svg = $("chartLoad");
    if (!Array.isArray(positions) || !positions.length) {
      emptyChart(svg, "Run a benchmark to populate this chart.");
      return;
    }
    const { plot } = chartFrame(svg);
    const max = Math.max(...positions, 1);
    yAxis(svg, plot, (fraction) => fmtInt(Math.round(max * fraction)));
    const slot = plot.width / positions.length;
    const barWidth = Math.min(48, slot * 0.62);
    positions.forEach((value, index) => {
      const height = (value / max) * plot.height;
      const x = plot.left + slot * index + (slot - barWidth) / 2;
      svg.appendChild(
        svgEl("rect", {
          x, y: plot.bottom - height, width: barWidth, height: Math.max(1, height),
          rx: 3, class: "bar",
        })
      );
      svg.appendChild(
        svgEl("text", {
          x: plot.left + slot * index + slot / 2,
          y: plot.bottom + 15, "text-anchor": "middle",
        }, `W${index}`)
      );
      svg.appendChild(
        svgEl("text", {
          x: plot.left + slot * index + slot / 2,
          y: plot.bottom - height - 5, "text-anchor": "middle",
        }, fmtInt(value))
      );
    });
  }

  function renderBoundsChart(unit) {
    const svg = $("chartBounds");
    if (!unit) {
      emptyChart(svg, "Run a benchmark to populate this chart.");
      return;
    }
    const bounds = unit.block_bounds || [];
    if (!bounds.length) {
      emptyChart(svg, "No block bounds recorded.");
      return;
    }
    const { plot } = chartFrame(svg);
    const finite = bounds.map((value) => (Number.isFinite(value) ? value : 0));
    const max = Math.max(...finite, 1e-9);
    const min = Math.min(0, ...finite);
    yAxis(svg, plot, (fraction) => {
      const value = min + (max - min) * fraction;
      return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(1);
    });
    const retained = new Set(unit.retained_workers || []);
    const slot = plot.width / finite.length;
    const barWidth = Math.min(46, slot * 0.62);
    finite.forEach((value, index) => {
      const range = max - min || 1;
      const height = ((value - min) / range) * plot.height;
      const x = plot.left + slot * index + (slot - barWidth) / 2;
      svg.appendChild(
        svgEl("rect", {
          x, y: plot.bottom - height, width: barWidth, height: Math.max(1, height),
          rx: 3, class: `bar ${retained.has(index) ? "retained" : "pruned"}`,
        })
      );
      svg.appendChild(
        svgEl("text", {
          x: plot.left + slot * index + slot / 2,
          y: plot.bottom + 15, "text-anchor": "middle",
        }, `W${index}`)
      );
    });
    svg.appendChild(
      svgEl("text", {
        x: plot.right, y: plot.top - 5, "text-anchor": "end",
      }, "max over Q heads")
    );
  }

  function renderCurveChart(unit, epsilon) {
    const svg = $("chartCurve");
    if (!unit || !Array.isArray(unit.retention_curve) || !unit.retention_curve.length) {
      emptyChart(svg, "Run a benchmark to populate this chart.");
      return;
    }
    const curve = unit.retention_curve;
    const { plot } = chartFrame(svg);
    const floor = Math.max(1e-16, epsilon * 1e-6);
    const values = [];
    for (const point of curve) {
      values.push(Math.max(point.max_certified_error, floor));
      values.push(Math.max(point.max_relative_error, floor));
    }
    values.push(Math.max(epsilon, floor));
    const top = Math.max(...values) * 5;
    const logFloor = Math.log10(floor);
    const logTop = Math.log10(top);
    const yFor = (value) => {
      const clamped = Math.max(floor, Math.min(top, value));
      const fraction = (Math.log10(clamped) - logFloor) / (logTop - logFloor);
      return plot.bottom - fraction * plot.height;
    };
    yAxis(svg, plot, (fraction) => {
      const value = 10 ** (logFloor + fraction * (logTop - logFloor));
      return value.toExponential(0);
    }, 4);

    if (epsilon >= floor && epsilon <= top) {
      const y = yFor(epsilon);
      svg.appendChild(
        svgEl("line", {
          x1: plot.left, y1: y, x2: plot.right, y2: y, class: "eps-line",
        })
      );
      svg.appendChild(
        svgEl("text", { x: plot.right, y: y - 4, "text-anchor": "end" }, "epsilon")
      );
    }

    const maxWorkers = Math.max(...curve.map((point) => point.retained_workers), 1);
    const xFor = (workers) => {
      if (maxWorkers <= 1) return plot.left + plot.width / 2;
      return plot.left + ((workers - 1) / (maxWorkers - 1)) * plot.width;
    };

    function polyline(key, className) {
      const points = curve
        .map((point) => `${xFor(point.retained_workers)},${yFor(point[key])}`)
        .join(" ");
      svg.appendChild(svgEl("polyline", { points, class: className }));
      curve.forEach((point) => {
        svg.appendChild(
          svgEl("circle", {
            cx: xFor(point.retained_workers),
            cy: yFor(point[key]),
            r: 3,
            class: className === "line-certified" ? "dot-certified" : "dot-actual",
          })
        );
      });
    }

    polyline("max_certified_error", "line-certified");
    polyline("max_relative_error", "line-actual");

    curve.forEach((point) => {
      if (maxWorkers <= 8 || point.retained_workers === 1 || point.retained_workers === maxWorkers) {
        svg.appendChild(
          svgEl("text", {
            x: xFor(point.retained_workers),
            y: plot.bottom + 15,
            "text-anchor": "middle",
          }, String(point.retained_workers))
        );
      }
    });
    svg.appendChild(
      svgEl("text", { x: plot.right, y: plot.bottom + 27, "text-anchor": "end" }, "retained workers")
    );

    const last = curve[curve.length - 1];
    svg.appendChild(
      svgEl("text", {
        x: xFor(last.retained_workers) - 6,
        y: yFor(last.max_certified_error) - 7,
        "text-anchor": "end",
      }, "certified")
    );
    svg.appendChild(
      svgEl("text", {
        x: xFor(last.retained_workers) - 6,
        y: yFor(last.max_relative_error) + 13,
        "text-anchor": "end",
      }, "actual")
    );
  }

  function populateUnits(dashboard) {
    const select = $("unitSelect");
    select.replaceChildren();
    const units = (dashboard && dashboard.units) || [];
    units.forEach((unit, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = `layer ${unit.layer} / kv ${unit.kv_head}`;
      select.appendChild(option);
    });
    select.disabled = units.length === 0;
    state.selectedUnit = 0;
    select.value = "0";
    if (units.length) {
      renderBoundsChart(units[0]);
      renderCurveChart(units[0], epsilonOf());
    }
  }

  function epsilonOf() {
    return Number(state.summary?.request?.error_tol ?? $("errorTol").value);
  }

  function renderCharts() {
    const dashboard = state.summary?.dashboard;
    if (!dashboard) {
      renderLoadChart(null);
      renderBoundsChart(null);
      renderCurveChart(null, epsilonOf());
      return;
    }
    renderLoadChart(dashboard.positions_per_worker);
    populateUnits(dashboard);
  }

  function resetMetrics() {
    $("metricExact").textContent = "--";
    $("metricExactSub").textContent = "max relative error across units";
    $("metricCertified").textContent = "--";
    $("metricCertifiedSub").textContent = `epsilon ${Number($("errorTol").value).toExponential(0)}`;
    $("metricRetained").textContent = "--";
    $("metricRetainedSub").textContent = "mean retained token fraction";
    $("metricFlops").textContent = "--";
    $("metricFlopsSub").textContent = "positive means fewer FLOPs than dense";
    $("statMass").textContent = "--";
    $("statEntropy").textContent = "--";
    $("statCapture").textContent = "--";
    $("statBenchmark").textContent = "--";
    $("statCalls").textContent = "--";
    $("statRetries").textContent = "--";
    $("accBaseline").textContent = "--";
    $("accOnline").textContent = "--";
    $("accPreproc").textContent = "--";
    $("accCharged").textContent = "--";
    $("accTotal").textContent = "--";
    $("accOnlineSavings").textContent = "--";
    $("accChargedSavings").textContent = "--";
    $("accBreakEven").textContent = "--";
    $("accountingNote").textContent = ACCOUNTING_NOTE_DEFAULT;
    $("overlapMode").textContent = "no dual ownership";
    $("ovCopied").textContent = "--";
    $("ovCopies").textContent = "--";
    $("ovExtra").textContent = "--";
    $("ovFlops").textContent = "--";
    $("ovControl").textContent = "--";
    $("overlapNote").textContent = OVERLAP_NOTE_DEFAULT;
    $("includePreproc").disabled = false;
    updateReuseLabel();
    $("preprocReuse").disabled = !$("includePreproc").checked;
    renderLoadChart(null);
    renderBoundsChart(null);
    renderCurveChart(null, Number($("errorTol").value));
    $("unitSelect").replaceChildren();
    $("unitSelect").disabled = true;
  }

  // ---------------------------------------------------------------- metrics
  const ACCOUNTING_NOTE_DEFAULT =
    "Run a benchmark to see the bounds, softmax, merge and preprocessing terms.";
  const OVERLAP_NOTE_DEFAULT =
    "Set overlap above 0 to exercise dual ownership. 15% matches the original " +
    "prototype; the control merges every copy with its log-multiplicity " +
    "correction and compares the result against dense attention.";

  function fmtPct(value) {
    if (value === null || value === undefined || Number.isNaN(Number(value))) return "--";
    const number = Number(value);
    return `${number >= 0 ? "+" : ""}${number.toFixed(2)}%`;
  }

  function updateReuseLabel() {
    const reuse = Math.max(1, Math.floor(Number($("preprocReuse").value) || 1));
    $("reuseOut").textContent = fmtInt(reuse);
  }

  function renderAccounting(aggregate, request) {
    if (!aggregate) return;
    const include = $("includePreproc").checked;
    const reuse = Math.max(1, Math.floor(Number($("preprocReuse").value) || 1));
    const baseline = Number(aggregate.baseline_flops || 0);
    const online = Number(aggregate.distributed_flops || 0);
    const hasPreproc =
      aggregate.preprocessing_flops !== undefined && aggregate.preprocessing_flops !== null;
    const preprocessing = hasPreproc ? Number(aggregate.preprocessing_flops) : 0;
    const charged = include ? preprocessing / reuse : 0;
    const total = online + charged;
    const onlineSavings = baseline > 0 ? 100 * (1 - online / baseline) : null;
    const chargedSavings = baseline > 0 ? 100 * (1 - total / baseline) : null;

    $("includePreproc").disabled = !hasPreproc;
    $("preprocReuse").disabled = !hasPreproc || !include;
    $("reuseOut").textContent = fmtInt(reuse);
    $("accBaseline").textContent = fmtInt(baseline);
    $("accOnline").textContent = fmtInt(online);
    $("accOnlineSavings").textContent = fmtPct(onlineSavings);

    if (!hasPreproc) {
      $("accPreproc").textContent = "not recorded";
      $("accCharged").textContent = "not recorded";
      $("accTotal").textContent = "not recorded";
      $("accBreakEven").textContent = "not recorded";
      $("accChargedSavings").textContent = "--";
      $("metricFlops").textContent = fmtPct(onlineSavings);
      $("metricFlopsSub").textContent = "this run predates the preprocessing accounting";
      $("accountingNote").textContent =
        "This run's summary has no bounds-pass term. Start a new run to record it.";
      return;
    }

    $("accPreproc").textContent = fmtInt(preprocessing);
    $("accCharged").textContent = include ? fmtInt(Math.round(charged)) : "not charged";
    $("accTotal").textContent = fmtInt(Math.round(total));
    $("accChargedSavings").textContent = fmtPct(chargedSavings);

    let breakEven = "not needed";
    if (aggregate.preprocessing_break_even_runs !== null &&
        aggregate.preprocessing_break_even_runs !== undefined) {
      const runs = Number(aggregate.preprocessing_break_even_runs);
      breakEven = `${fmtInt(runs)} ${runs === 1 ? "run" : "runs"}`;
    } else if (online >= baseline) {
      breakEven = "never";
    }
    $("accBreakEven").textContent = breakEven;

    $("metricFlops").textContent = fmtPct(chargedSavings);
    $("metricFlopsSub").textContent = include
      ? `online ${fmtPct(onlineSavings)} · bounds ${fmtInt(preprocessing)} ÷ ${fmtInt(reuse)}`
      : `online only · bounds ${fmtInt(preprocessing)} excluded`;
    $("accountingNote").textContent = include
      ? (reuse === 1
          ? "The bounds pass is charged once on this run. Cached centroids and radii would let one pass serve several runs."
          : `The bounds pass is charged once and shared by ${fmtInt(reuse)} runs, so each run pays 1/${fmtInt(reuse)} of it.`)
      : "The bounds pass is excluded, which reproduces the online saving. Turn the charge on to include the cost of computing centroids and radii.";
  }

  function renderOverlapAccounting(aggregate, request) {
    if (!aggregate) return;
    const overlap = Number(request?.overlap_pct ?? $("overlap").value ?? 0);
    const hasOverlap =
      aggregate.extra_copies !== undefined && aggregate.extra_copies !== null;
    $("overlapMode").textContent =
      overlap > 0 ? `${overlap.toFixed(0)}% dual ownership` : "no dual ownership";
    $("ovCopies").textContent = Number(aggregate.mean_copies_per_token || 1).toFixed(3);

    if (!hasOverlap) {
      $("ovCopied").textContent = "not recorded";
      $("ovExtra").textContent = "not recorded";
      $("ovFlops").textContent = "not recorded";
      $("ovControl").textContent = "not recorded";
      $("overlapNote").textContent =
        "This run predates the overlap breakdown. Start a new run to record copies and the overlap control.";
      return;
    }

    const copied = Number(aggregate.copied_token_fraction || 0);
    const extra = Number(aggregate.extra_copies || 0);
    const extraFlops = Number(aggregate.overlap_stats_extra_flops || 0);
    const extraPct = Number(aggregate.overlap_stats_extra_pct || 0);
    const control = aggregate.max_overlap_control_rel_error;

    $("ovCopied").textContent = `${(copied * 100).toFixed(1)}%`;
    $("ovExtra").textContent = fmtInt(extra);
    $("ovFlops").textContent =
      extra > 0 ? `${fmtInt(extraFlops)} (${extraPct.toFixed(2)}%)` : "0";
    $("ovControl").textContent = control === null || control === undefined
      ? "not run"
      : fmtError(control);
    $("overlapNote").textContent = overlap > 0
      ? "Every duplicated position is corrected by log(c) before the merge. The overlap control queries every copy and must match dense attention to 1e-8."
      : OVERLAP_NOTE_DEFAULT;
  }

  function refreshAccounting() {
    updateReuseLabel();
    if (state.aggregate) {
      renderAccounting(state.aggregate, state.aggregateRequest);
      renderOverlapAccounting(state.aggregate, state.aggregateRequest);
    }
  }

  function renderAggregate(aggregate, request) {
    if (!aggregate) return;
    state.aggregate = aggregate;
    state.aggregateRequest = request || null;
    const epsilon = Number(request?.error_tol ?? $("errorTol").value);
    $("metricExact").textContent = fmtError(aggregate.max_relative_error);
    $("metricExactSub").textContent =
      `disjoint control ${fmtError(aggregate.max_disjoint_control_rel_error)}`;
    $("metricCertified").textContent = fmtError(aggregate.max_certified_error);
    $("metricCertifiedSub").textContent =
      `epsilon ${epsilon.toExponential(0)} · ${aggregate.tolerance_satisfied ? "satisfied" : "not reached"}`;
    const workers = Number(request?.workers ?? 0);
    $("metricRetained").textContent =
      workers > 0 ? `${Number(aggregate.mean_retained_workers).toFixed(2)} / ${workers}` : "--";
    $("metricRetainedSub").textContent =
      `retained tokens ${(Number(aggregate.mean_retained_fraction) * 100).toFixed(1)}%`;
    renderAccounting(aggregate, request);
    renderOverlapAccounting(aggregate, request);
    $("statMass").textContent = `${Number(aggregate.peak_attention_mass_pct).toFixed(1)}%`;
    $("statEntropy").textContent = `${Number(aggregate.attention_entropy_bits).toFixed(2)} bits`;
  }

  function renderTimings(summary) {
    const timings = summary.timings || {};
    $("statCapture").textContent = fmtSeconds(timings.capture_s);
    $("statBenchmark").textContent = fmtSeconds(timings.benchmark_s);
    const cluster = summary.cluster || {};
    $("statCalls").textContent = fmtInt(cluster.requests);
    $("statRetries").textContent = fmtInt(cluster.retries);
    if (summary.status === "completed" || summary.status === "failed" || summary.status === "cancelled") {
      $("metaElapsed").textContent = fmtClock(Number(summary.duration_s || 0));
    }
  }

  function renderGeneration(generation) {
    const container = $("generationBody");
    container.replaceChildren();
    if (!generation || generation.status === "disabled") {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = "Generation was not requested for this run.";
      container.appendChild(note);
      return;
    }
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = generation.status === "ok"
      ? "exact causal decode, KV cache on the workers"
      : generation.status === "not_run"
        ? "distributed generation did not run"
        : "distributed generation failed";
    container.appendChild(tag);
    const output = document.createElement("pre");
    output.className = "generation-text";
    if (generation.status === "ok") {
      output.textContent = generation.text || "";
    } else if (generation.status === "not_run") {
      output.textContent = `Generation did not run: ${generation.message || "the run failed earlier"}`;
    } else {
      output.textContent = `Generation failed: ${generation.message || generation.error || "unknown error"}`;
    }
    container.appendChild(output);
    if (generation.status === "ok" && !String(generation.text || "").trim()) {
      const note = document.createElement("p");
      note.className = "muted";
      note.textContent = "The generated text contains only whitespace.";
      container.appendChild(note);
    }
    if (generation.status === "ok") {
      const tokens = generation.token_count ?? generation.generated_token_count;
      const details = document.createElement("dl");
      details.className = "meta";
      const add = (label, value) => {
        const dt = document.createElement("dt");
        dt.textContent = label;
        const dd = document.createElement("dd");
        dd.textContent = value;
        details.append(dt, dd);
      };
      if (tokens != null) add("Tokens", fmtInt(tokens));
      if (generation.stop_reason) add("Stop", generation.stop_reason);
      const timing = generation.timing_s ?? generation.timings?.total_s;
      if (timing != null) add("Wall time", fmtSeconds(timing));
      if (generation.workers != null) add("Workers", fmtInt(generation.workers));
      container.appendChild(details);
      const provenance = generation.computed_by || generation.provenance;
      if (provenance) {
        const note = document.createElement("p");
        note.className = "muted";
        note.textContent = provenance;
        container.appendChild(note);
      }
    }
  }

  function renderArtifacts(runId, names) {
    const list = $("artifactList");
    list.replaceChildren();
    if (!runId || !names || !names.length) {
      const item = document.createElement("li");
      item.className = "muted";
      item.textContent = "No artifacts yet.";
      list.appendChild(item);
      return;
    }
    for (const entry of names) {
      const name = typeof entry === "string" ? entry : entry.name;
      if (!name) continue;
      const item = document.createElement("li");
      const link = document.createElement("a");
      link.href = `/api/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(name)}`;
      link.textContent = typeof entry === "string" ? entry : `${entry.name} (${fmtBytes(entry.bytes)})`;
      link.download = name;
      item.appendChild(link);
      list.appendChild(item);
    }
  }

  function fmtBytes(bytes) {
    if (bytes === undefined || bytes === null) return "";
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function renderSummary(summary) {
    if (!summary) return;
    state.summary = summary;
    state.summaryFor = summary.run_id;
    renderAggregate(summary.aggregate, summary.request);
    renderTimings(summary);
    renderCharts();
    renderArtifacts(
      summary.run_id,
      summary.artifacts && summary.artifacts.length
        ? summary.artifacts
        : ["summary.json", "manifest.json"]
    );
    $("curveEps").textContent =
      `epsilon ${Number(summary.request?.error_tol ?? 0).toExponential(0)}`;
    if (summary.status !== "completed") {
      const reason = summary.error?.message || summary.error || summary.status;
      showBanner(`Run ${summary.status}: ${reason}`);
    }
  }

  async function loadSummary(runId) {
    if (!runId || state.summaryFor === runId) return;
    try {
      const summary = await api(`/api/runs/${encodeURIComponent(runId)}/result`);
      if (state.run) {
        state.run.status = summary.status;
        setStatusPill(summary.status);
        renderRunMeta();
      }
      renderSummary(summary);
      renderGeneration(summary.generation || null);
      setProgress(100, `run ${summary.status}`);
    } catch (error) {
      if (error.status === 409) {
        setTimeout(() => loadSummary(runId), 600);
        return;
      }
      if (error.status === 401) {
        showLogin();
        return;
      }
      showBanner(`Could not load the run summary: ${error.message}`);
    }
    loadArtifacts(runId);
  }

  async function loadArtifacts(runId) {
    try {
      const data = await api(`/api/runs/${encodeURIComponent(runId)}/artifacts`);
      if (data && Array.isArray(data.artifacts)) {
        renderArtifacts(runId, data.artifacts);
      }
    } catch {
      /* the summary already listed the expected artifacts */
    }
  }

  // ---------------------------------------------------------------- events
  function attachRun(run) {
    if (!run || !run.run_id) return;
    state.run = run;
    state.summary = null;
    state.summaryFor = null;
    state.aggregate = null;
    state.aggregateRequest = null;
    state.lastSeq = 0;
    state.netCount = 0;
    state.startedAt = run.started_utc || null;
    clearLogs();
    hideBanner();
    resetMetrics();
    renderGeneration(null);
    setStatusPill(run.status);
    renderRunMeta();
    $("metaTokens").textContent = run.config?.max_tokens ? `<= ${run.config.max_tokens}` : "--";
    const running = run.status === "running";
    $("runBtn").disabled = running;
    $("cancelBtn").hidden = !running;
    setProgress(running ? 3 : 100, running ? "starting" : `run ${run.status}`);
    setPhase(running ? "starting" : run.status);
    openStream(run.run_id);
    if (!running) loadSummary(run.run_id);
    startElapsedTimer();
    startStatusPolling();
  }

  function openStream(runId) {
    if (state.stream) {
      state.stream.close();
      state.stream = null;
    }
    if (state.reconnectTimer) {
      clearTimeout(state.reconnectTimer);
      state.reconnectTimer = null;
    }
    const source = new EventSource(
      `/api/runs/${encodeURIComponent(runId)}/events?after=${state.lastSeq}`
    );
    state.stream = source;

    source.onmessage = (event) => {
      let payload;
      try { payload = JSON.parse(event.data); } catch { return; }
      if (typeof payload.seq === "number") {
        if (payload.seq <= state.lastSeq) return;
        state.lastSeq = payload.seq;
      }
      handleEvent(payload);
    };

    source.addEventListener("end", () => {
      source.close();
      state.stream = null;
      finishRun();
    });

    source.onerror = () => {
      if (source !== state.stream) return;
      source.close();
      state.stream = null;
      if (state.run && state.run.status === "running") {
        state.reconnectTimer = setTimeout(() => {
          if (state.run && state.run.status === "running") openStream(runId);
        }, 1500);
      }
    };
  }

  function handleEvent(event) {
    switch (event.type) {
      case "run":
        break;
      case "info":
        addPipeline("info", event.msg || "", event);
        break;
      case "stage":
        handleStage(event);
        break;
      case "comm":
        addNetwork(
          `${event.msg || event.event || "worker call"}${event.ms ? ` [${Number(event.ms).toFixed(1)} ms]` : ""}`,
          event
        );
        break;
      case "generation":
        addPipeline(
          event.status === "ok" ? "gen" : "warn",
          event.status === "ok"
            ? `distributed decode produced ${fmtInt(event.token_count)} tokens (${event.stop_reason || "done"})`
            : `distributed generation failed: ${event.message || "unknown error"}`,
          event
        );
        renderGeneration(event);
        break;
      case "error":
        addPipeline("error", `run failed: ${event.message || event.error_type || "unknown"}`, event);
        showBanner(event.message || "run failed");
        setStatusPill("failed");
        setPhase("failed");
        break;
      case "done": {
        const status = event.status || "completed";
        setStatusPill(status);
        setPhase(status);
        if (status === "completed") setProgress(100, "complete");
        addPipeline(
          status === "completed" ? "done" : status,
          `run ${status}`,
          event
        );
        finishRun(status);
        break;
      }
      default:
        addPipeline("info", JSON.stringify(event), event);
    }
  }

  function handleStage(event) {
    switch (event.stage) {
      case "capture": {
        if (event.phase === "starting") {
          setPhase("capturing attention");
          setProgress(5, `capturing from ${event.model_id || "model"}`);
          addPipeline("cap", `capture starting (limit ${event.max_tokens} tokens)`, event);
        } else if (event.phase === "done") {
          setProgress(46, "capture complete");
          addPipeline(
            "cap",
            `capture complete: ${fmtInt(event.token_count)} tokens, ${event.layers} layers, ${event.kv_heads} KV heads in ${fmtSeconds(event.timing_s)}`,
            event
          );
        } else if (typeof event.position === "number" && event.selected_layers) {
          const fraction = Math.min(1, event.position / event.selected_layers);
          setProgress(5 + 41 * fraction, `capturing layer ${event.position}/${event.selected_layers}`);
          if (event.position === event.selected_layers || event.position % 7 === 0) {
            addPipeline(
              "cap",
              `layer ${event.position}/${event.selected_layers} captured, RSS ${event.rss_mb ?? "?"} MB`,
              event
            );
          }
        }
        break;
      }
      case "cluster": {
        if (event.phase === "spawning") {
          setPhase("starting workers");
          setProgress(48, `spawning ${event.workers} isolated workers`);
          addPipeline("cluster", `spawning ${event.workers} isolated HTTP workers`, event);
        } else if (event.phase === "ready") {
          setProgress(50, "workers ready");
          setPhase("workers ready");
          addPipeline(
            "cluster",
            `worker cluster ready (session ${event.session}, ${Number(event.free_ram_gb).toFixed(2)} GB free)`,
            event
          );
        }
        break;
      }
      case "partition": {
        setProgress(52, "replaying captured tensors");
        addPipeline(
          "bench",
          `partition: K=${event.workers}, N=${event.tokens}, mean copies ${Number(event.mean_copies).toFixed(3)}`,
          event
        );
        break;
      }
      case "benchmark": {
        const fraction = event.total ? event.position / event.total : 0;
        setPhase("certified selection");
        setProgress(52 + 42 * fraction, `layer ${event.layer} / kv ${event.kv_head}`);
        if (
          event.position === 1 ||
          event.position === event.total ||
          event.position % 8 === 0
        ) {
          addPipeline(
            "bench",
            `layer ${event.layer} kv ${event.kv_head}: retained ${event.retained}/${event.workers}, certified ${fmtError(event.certified)}, actual ${fmtError(event.relative_error)}, FLOPs ${Number(event.savings_pct).toFixed(2)}%`,
            event
          );
        }
        break;
      }
      case "benchmark_done": {
        setProgress(95, "benchmark complete");
        setPhase("generation");
        addPipeline("bench", "distributed attention benchmark complete", event);
        renderAggregate(event.aggregate, state.run?.config);
        if (event.aggregate) {
          $("statMass").textContent =
            `${Number(event.aggregate.peak_attention_mass_pct).toFixed(1)}%`;
          $("statEntropy").textContent =
            `${Number(event.aggregate.attention_entropy_bits).toFixed(2)} bits`;
        }
        break;
      }
      case "generation": {
        if (event.phase === "starting") {
          setPhase("generation starting");
          setProgress(96, `decoding ${event.max_new_tokens} tokens across the workers`);
          addPipeline(
            "gen",
            `distributed decoder starting: ${fmtInt(event.prompt_tokens)} prompt tokens, ${event.max_new_tokens} new tokens`,
            event
          );
        } else if (event.phase === "prefill_start") {
          setPhase("distributed prefill");
          setProgress(96, `prefill ${fmtInt(event.prompt_tokens)} tokens`);
          addPipeline(
            "gen",
            `prefill across ${event.workers} workers, context ${event.max_positions}`,
            event
          );
        } else if (event.phase === "prefill") {
          const total = event.layers || 1;
          setProgress(96 + 2 * (event.layer / total), `prefill layer ${event.layer + 1}/${total}`);
        } else if (event.phase === "decode") {
          const total = event.layers || 1;
          setProgress(
            98,
            `decode position ${event.position}, layer ${event.layer + 1}/${total}`
          );
        } else if (event.phase === "sampling") {
          setProgress(98, `sampling token ${event.step}/${event.of}`);
        }
        break;
      }
      case "generation_cluster": {
        if (event.phase === "spawning") {
          setPhase("starting generation workers");
          setProgress(95, `spawning ${event.workers} KV workers`);
          addPipeline(
            "cluster",
            `spawning ${event.workers} workers for the generation KV cache`,
            event
          );
        } else if (event.phase === "ready") {
          setPhase("generation workers ready");
          setProgress(96, "generation workers ready");
          addPipeline(
            "cluster",
            `generation cluster ready (session ${event.session}, ${Number(event.free_ram_gb).toFixed(2)} GB free, needs ${Number(event.needed_ram_gb).toFixed(2)} GB)`,
            event
          );
        }
        break;
      }
      default:
        addPipeline("stage", JSON.stringify(event), event);
    }
  }

  function finishRun(statusOverride) {
    if (!state.run) return;
    const status = statusOverride || state.run.status || "completed";
    state.run.status = status;
    setStatusPill(status);
    $("runBtn").disabled = false;
    $("cancelBtn").hidden = true;
    stopStatusPolling();
    stopElapsedTimer();
    if (status === "completed") setProgress(100, "complete");
    if (state.stream) {
      state.stream.close();
      state.stream = null;
    }
    loadSummary(state.run.run_id);
    refreshStatus();
  }

  // ---------------------------------------------------------------- timers
  function startElapsedTimer() {
    stopElapsedTimer();
    state.elapsedTimer = setInterval(() => {
      if (!state.run) return;
      if (state.run.status === "running") {
        $("metaElapsed").textContent = fmtElapsed(state.run.started_utc || state.startedAt);
      }
    }, 1000);
  }

  function stopElapsedTimer() {
    if (state.elapsedTimer) {
      clearInterval(state.elapsedTimer);
      state.elapsedTimer = null;
    }
  }

  function startStatusPolling() {
    stopStatusPolling();
    state.pollTimer = setInterval(async () => {
      if (!state.run || state.run.status !== "running") {
        stopStatusPolling();
        return;
      }
      try {
        const data = await api(`/api/runs/${encodeURIComponent(state.run.run_id)}`);
        if (!data.run) return;
        if (data.run.status !== "running") {
          state.run.status = data.run.status;
          finishRun(data.run.status);
        }
      } catch (error) {
        if (error.status === 401) showLogin();
      }
    }, 2000);
  }

  function stopStatusPolling() {
    if (state.pollTimer) {
      clearInterval(state.pollTimer);
      state.pollTimer = null;
    }
  }

  // ---------------------------------------------------------------- forms
  function updateSliderLabels() {
    $("workersOut").textContent = $("workers").value;
    $("tokensOut").textContent = fmtInt($("maxTokens").value);
    $("overlapOut").textContent = `${$("overlap").value}%`;
    $("promptCount").textContent = fmtInt($("prompt").value.length);
  }

  function currentPayload() {
    return {
      prompt: $("prompt").value,
      workers: Number($("workers").value),
      max_tokens: Number($("maxTokens").value),
      overlap_pct: Number($("overlap").value),
      strategy: $("strategy").value,
      seed: Number($("seed").value),
      error_tol: Number($("errorTol").value),
      generate: $("generate").checked,
      max_new_tokens: Number($("maxNewTokens").value),
    };
  }

  function setButtonLabel(id, text) {
    const button = $(id);
    const label = button.querySelector(".btn-label");
    if (label) {
      label.textContent = text;
      return;
    }
    button.textContent = text;
  }

  async function startRun() {
    hideBanner();
    const payload = currentPayload();
    if (!payload.prompt.trim()) {
      showBanner("Enter a prompt before starting a run.");
      return;
    }
    $("runBtn").disabled = true;
    setButtonLabel("runBtn", "Starting...");
    try {
      const data = await api("/api/runs", { method: "POST", body: payload });
      if (data && data.run) {
        attachRun(data.run);
        addPipeline("run", `run ${data.run.run_id} accepted`, {
          ts: data.run.started_utc,
        });
      }
    } catch (error) {
      if (error.status === 401) {
        showLogin();
      } else if (error.status === 409) {
        showBanner("A run is already active. Attaching to it.");
        await attachCurrentRun();
      } else {
        showBanner(error.message);
      }
      $("runBtn").disabled = false;
    } finally {
      setButtonLabel("runBtn", "Run benchmark");
    }
  }

  async function cancelRun() {
    if (!state.run) return;
    $("cancelBtn").disabled = true;
    try {
      const data = await api(
        `/api/runs/${encodeURIComponent(state.run.run_id)}/cancel`,
        { method: "POST" }
      );
      addPipeline("warn", data.cancelled ? "cancellation requested" : "run is not active", {});
    } catch (error) {
      if (error.status === 401) showLogin();
      else showBanner(error.message);
    } finally {
      $("cancelBtn").disabled = false;
    }
  }

  async function attachCurrentRun() {
    try {
      const data = await api("/api/runs/current");
      if (data && data.run) {
        attachRun(data.run);
      } else {
        setProgress(0, "no run yet");
      }
    } catch (error) {
      if (error.status === 401) showLogin();
    }
  }

  async function refreshStatus() {
    try {
      const data = await api("/api/status");
      state.authRequired = Boolean(data.auth_required);
      $("logoutBtn").hidden = !state.authRequired;
      if (data.current_run) {
        state.run = data.current_run;
        renderRunMeta();
        setStatusPill(data.current_run.status);
      }
    } catch (error) {
      if (error.status === 401) showLogin();
    }
  }

  // ---------------------------------------------------------------- auth
  function revealKeyForm() {
    $("loginTitle").textContent = "API key required";
    $("loginHelp").hidden = false;
    $("apiKeyField").hidden = false;
    $("loginOverlay").hidden = false;
    $("apiKey").focus();
  }

  async function refreshSession() {
    // Keyless local mode never asks for a secret: an expired cookie is simply
    // re-issued. The guard keeps a failed bootstrap from recursing here.
    if (state.refreshing) return false;
    state.refreshing = true;
    try {
      await api("/api/session", { method: "POST", body: { api_key: "" } });
      await bootstrap();
      return true;
    } catch {
      return false;
    } finally {
      state.refreshing = false;
    }
  }

  async function showLogin() {
    if (state.stream) {
      state.stream.close();
      state.stream = null;
    }
    if (!state.authRequired && (await refreshSession())) {
      return;
    }
    revealKeyForm();
  }

  function hideLogin() {
    $("loginOverlay").hidden = true;
    $("loginError").hidden = true;
    $("apiKey").value = "";
  }

  async function submitLogin(event) {
    event.preventDefault();
    const key = $("apiKey").value;
    if (!key) return;
    $("loginError").hidden = true;
    try {
      await api("/api/session", { method: "POST", body: { api_key: key } });
      hideLogin();
      await bootstrap();
    } catch (error) {
      const target = $("loginError");
      target.textContent = error.status === 429
        ? "Too many attempts. Wait before trying again."
        : "That key was rejected.";
      target.hidden = false;
    }
  }

  async function logout() {
    try {
      await api("/api/logout", { method: "POST" });
    } catch {
      /* cookie is cleared client side by the server response when it succeeds */
    }
    window.location.reload();
  }

  // ---------------------------------------------------------------- boot
  async function bootstrap() {
    try {
      const config = await api("/api/public-config");
      state.limits = config.limits;
      state.authRequired = Boolean(config.auth_required);
      $("promptLimit").textContent = fmtInt(config.limits.max_prompt_chars);
      if ($("maxNewTokens")) {
        $("maxNewTokens").max = config.limits.max_generation_tokens;
        $("maxNewTokens").value = config.limits.default_max_new_tokens;
      }
      if (state.authRequired) {
        $("logoutBtn").hidden = false;
      }
    } catch (error) {
      showBanner(`Server configuration is unavailable: ${error.message}`);
      return;
    }

    try {
      const status = await api("/api/status");
      hideLogin();
      state.authRequired = Boolean(status.auth_required);
      $("logoutBtn").hidden = !state.authRequired;
      if (status.metrics) {
        const rss = status.metrics.process_rss_mb;
        if (rss && rss > 0) {
          addPipeline("sys", `server RSS ${Number(rss).toFixed(0)} MB`, {});
        }
      }
      if (status.current_run) {
        attachRun(status.current_run);
      } else {
        setProgress(0, "no run yet");
      }
    } catch (error) {
      if (error.status === 401) {
        if (!state.refreshing) await showLogin();
      } else {
        showBanner(`Server status is unavailable: ${error.message}`);
      }
    }
  }

  function bind() {
    $("prompt").addEventListener("input", updateSliderLabels);
    $("workers").addEventListener("input", updateSliderLabels);
    $("maxTokens").addEventListener("input", updateSliderLabels);
    $("overlap").addEventListener("input", updateSliderLabels);
    $("overlapPreset").addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      $("overlap").value = "15";
      updateSliderLabels();
    });
    $("includePreproc").addEventListener("change", refreshAccounting);
    $("preprocReuse").addEventListener("input", refreshAccounting);
    $("runBtn").addEventListener("click", startRun);
    $("cancelBtn").addEventListener("click", cancelRun);
    $("clearLogs").addEventListener("click", clearLogs);
    $("tabPipeline").addEventListener("click", () => switchTab("pipeline"));
    $("tabNetwork").addEventListener("click", () => switchTab("network"));
    document.querySelectorAll(".nav-button[data-view]").forEach((button) => {
      button.addEventListener("click", () => switchView(button.dataset.view));
    });
    $("logoutBtn").addEventListener("click", logout);
    $("loginForm").addEventListener("submit", submitLogin);
    $("unitSelect").addEventListener("change", (event) => {
      state.selectedUnit = Number(event.target.value);
      const units = state.summary?.dashboard?.units || [];
      const unit = units[state.selectedUnit];
      if (unit) {
        renderBoundsChart(unit);
        renderCurveChart(unit, epsilonOf());
      }
    });
  }

  bind();
  restoreView();
  updateSliderLabels();
  updateReuseLabel();
  renderLoadChart(null);
  renderBoundsChart(null);
  renderCurveChart(null, Number($("errorTol").value));
  bootstrap();
})();
