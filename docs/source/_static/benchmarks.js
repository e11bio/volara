// Renders one line chart per benchmark from window.BENCHMARK_DATA, the file
// github-action-benchmark writes to gh-pages:dev/bench/data.js.

function benchmarksUnavailable() {
  const status = document.getElementById("benchmark-status");
  if (status) {
    status.textContent =
      "No benchmark data in this build. Data appears once the benchmark " +
      "workflow has run on main and the docs have been rebuilt.";
  }
}

function renderBenchmarks() {
  const data = window.BENCHMARK_DATA;
  const container = document.getElementById("benchmark-charts");
  if (!data || !container) {
    benchmarksUnavailable();
    return;
  }
  const status = document.getElementById("benchmark-status");
  if (status) {
    status.remove();
  }
  const dark = document.documentElement.dataset.theme === "dark";
  const text = dark ? "#d0d0d0" : "#333";
  const grid = dark ? "rgba(255,255,255,0.12)" : "rgba(0,0,0,0.1)";

  for (const [suite, entries] of Object.entries(data.entries)) {
    const names = [];
    for (const entry of entries) {
      for (const bench of entry.benches) {
        if (!names.includes(bench.name)) {
          names.push(bench.name);
        }
      }
    }
    for (const name of names) {
      const points = entries
        .map((entry) => {
          const bench = entry.benches.find((b) => b.name === name);
          return bench ? { entry, bench } : null;
        })
        .filter(Boolean);
      const unit = points[0].bench.unit;

      const heading = document.createElement("h2");
      heading.textContent = entries.length > 1 && Object.keys(data.entries).length > 1
        ? `${suite}: ${name}`
        : name;
      container.appendChild(heading);
      const canvas = document.createElement("canvas");
      canvas.height = 90;
      container.appendChild(canvas);

      const chart = new Chart(canvas, {
        type: "line",
        data: {
          labels: points.map((p) => p.entry.commit.id.slice(0, 7)),
          datasets: [
            {
              label: `${name} (${unit})`,
              data: points.map((p) => p.bench.value),
              borderColor: "#2a6fdb",
              backgroundColor: "#2a6fdb",
              tension: 0.1,
              pointRadius: 4,
              pointHoverRadius: 6,
            },
          ],
        },
        options: {
          animation: false,
          scales: {
            x: { title: { display: true, text: "commit", color: text }, ticks: { color: text }, grid: { color: grid } },
            y: { beginAtZero: true, title: { display: true, text: unit, color: text }, ticks: { color: text }, grid: { color: grid } },
          },
          plugins: {
            legend: { display: false },
            tooltip: {
              callbacks: {
                title: (items) => {
                  const p = points[items[0].dataIndex];
                  return p.entry.commit.message.split("\n")[0];
                },
                label: (item) => {
                  const p = points[item.dataIndex];
                  const spread = p.bench.range ? ` ${p.bench.range}` : "";
                  return `${p.bench.value} ${unit}${spread}`;
                },
                afterLabel: (item) => {
                  const p = points[item.dataIndex];
                  return `${p.entry.commit.id.slice(0, 7)} · ${new Date(p.entry.commit.timestamp).toLocaleString()}`;
                },
              },
            },
          },
          onClick: (_event, elements) => {
            if (elements.length) {
              window.open(points[elements[0].index].entry.commit.url, "_blank");
            }
          },
        },
      });
      chart.canvas.style.cursor = "pointer";
    }
  }
}
