# Benchmarks

One point per commit on `main`, from the `benchmark` workflow. Hover a point
for the commit and the measured spread; click it to open the commit. Smaller
is better for every metric.

<div id="benchmark-charts">
  <p id="benchmark-status">Loading benchmark data…</p>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.js"></script>
<script src="_static/benchmarks.js"></script>
<script src="dev/bench/data.js" onerror="benchmarksUnavailable()"></script>
<script>window.addEventListener("load", renderBenchmarks);</script>

The raw data is `dev/bench/data.js` next to this site, written by
[github-action-benchmark](https://github.com/benchmark-action/github-action-benchmark)
to the `gh-pages` branch and copied into the docs at build time. The
action's own dashboard is at
<https://e11bio.github.io/volara/dev/bench/>.
