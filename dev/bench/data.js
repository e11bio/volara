window.BENCHMARK_DATA = {
  "lastUpdate": 1793850593147,
  "repoUrl": "https://github.com/e11bio/volara",
  "entries": {
    "volara": [
      {
        "commit": {
          "author": {
            "name": "William Patton",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "228b2b55c16780578944037d9a48605530f3bdc0",
          "message": "Run coverage from pytest addopts (#96)\n\n* Run coverage from pytest addopts so a bare `uv run pytest` is the CI command\n\n`[tool.pytest.ini_options] addopts` now carries `--cov --cov-report=term-missing`,\nso the README's `uv run pytest` prints the coverage table and enforces\n`[tool.coverage.report] fail_under` the same way CI does. The CI step drops the\nflags it no longer needs. A single-file run can skip the floor with `--no-cov`.\n\nCo-Authored-By: Claude Mythos 5.1 <noreply@anthropic.com>\n\n* Track import time and test duration across commits\n\nThe tests job times its own `uv run pytest`, measures the cold import of\nthe package's headline modules (benchmarks/run.py), and uploads the two\nnumbers as an artifact from one matrix cell. The benchmark workflow runs\nwhen a tests run on main completes, downloads that artifact, creates the\ngh-pages branch if it is missing, and publishes with github-action-benchmark\nto gh-pages:dev/bench, with a comparison table in the job summary and a\ncommit comment when a metric gets 1.5x worse. The docs gain a Benchmarks\npage that charts every metric from that data, one line per metric with the\ncommit on hover and click; the docs workflow now runs after each benchmark\nrun and copies dev/bench from gh-pages into the deployed site, so the\ncharts carry the newest point and the action's own dashboard keeps its URL.\nNothing runs the test suite twice.\n\nCo-Authored-By: Claude Mythos 5.1 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Mythos 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-11-05T03:35:31Z",
          "url": "https://github.com/e11bio/volara/commit/228b2b55c16780578944037d9a48605530f3bdc0"
        },
        "date": 1793850256250,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara.blockwise, volara.datasets (cold)",
            "value": 1657.221,
            "range": "± 133.354",
            "unit": "ms",
            "extra": "interpreter startup 16.7 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 46.471,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Will",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "committer": {
            "name": "Will",
            "username": "pattonw",
            "email": "wllmpttn24@gmail.com"
          },
          "id": "e2e558f56698ac46e4dc99cc52a2423e8b2f682b",
          "message": "Keep the task out of the worker spawn closure\n\n`BlockwiseTask.worker_func` built its spawn function over `self`, so daisy\npickled the whole task into every driver-side worker helper and each\nhelper imported the task's module graph: about 1 GB of driver RAM per\nworker slot on a Predict task, 100 GB for 130 slurm workers, while the\nhelper only runs `sbatch --wait`. The closure now captures the worker\nconfig, the config file path and the task name, which is all the command\nneeds; the cluster job still rebuilds the task from the config file. A\nregression test checks the closure's free variables, and the SlurmWorker\ndocstring says why `num_workers` should match what the queue can grant.\n\nFixes #95\n\nCo-Authored-By: Claude Mythos 5.1 <noreply@anthropic.com>",
          "timestamp": "2026-11-05T03:43:48Z",
          "url": "https://github.com/e11bio/volara/commit/e2e558f56698ac46e4dc99cc52a2423e8b2f682b"
        },
        "date": 1793850590047,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara.blockwise, volara.datasets (cold)",
            "value": 2281.174,
            "range": "± 37.662",
            "unit": "ms",
            "extra": "interpreter startup 25.9 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 58.957,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      }
    ]
  }
}