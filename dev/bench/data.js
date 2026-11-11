window.BENCHMARK_DATA = {
  "lastUpdate": 1794424633440,
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
      },
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
          "id": "e41b26c02113bc24c7c281a7aea50402ebc5922b",
          "message": "Build docs on pull requests and deploy them on version tags (#98)\n\n* Build docs on pull requests and deploy them on version tags\n\nThe docs workflow ran after every benchmark run on main and deployed each\ntime. It now builds on every pull request without deploying, deploys on a\nv* tag push, and can be redeployed by hand, so the site describes the\nreleased version. The benchmark charts are still copied from gh-pages at\ndeploy time, so they refresh at each release.\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>\n\n* Run the lowest-direct test leg against the declared floors\n\nThe resolution mode was passed to `uv sync` only. The next step, a plain\n`uv run pytest`, saw a lockfile written in another mode, re-resolved to the\nnewest versions and replaced the environment, so the leg named\nlowest-direct tested unlocked newest versions instead. The mode is now set\nonce as UV_RESOLUTION for the whole job.\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>\n\n---------\n\nCo-authored-by: Claude Fable 5.2 <noreply@anthropic.com>",
          "timestamp": "2026-11-10T23:50:00Z",
          "url": "https://github.com/e11bio/volara/commit/e41b26c02113bc24c7c281a7aea50402ebc5922b"
        },
        "date": 1794354992866,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara.blockwise, volara.datasets (cold)",
            "value": 1570.189,
            "range": "± 22.58",
            "unit": "ms",
            "extra": "interpreter startup 18.4 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 47.903,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      },
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
          "id": "b545593a8458bfd0b26c850e9293b93c17e7e762",
          "message": "Bump version to 2.1.0 (#99)\n\nCo-authored-by: Claude Fable 5.2 <noreply@anthropic.com>",
          "timestamp": "2026-11-10T23:58:55Z",
          "url": "https://github.com/e11bio/volara/commit/b545593a8458bfd0b26c850e9293b93c17e7e762"
        },
        "date": 1794355570412,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara.blockwise, volara.datasets (cold)",
            "value": 1713.885,
            "range": "± 30.667",
            "unit": "ms",
            "extra": "interpreter startup 19.6 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 47.293,
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
          "id": "1d6c739922a133b7a4dedb8bdcbd774f0d1c423b",
          "message": "Draw the benchmark charts on pages that load require.js\n\njupyter-sphinx puts require.js on every docs page. With an AMD loader\npresent, the Chart.js bundle registers as an AMD module and never sets\nthe global Chart, so benchmarks.js failed at new Chart() and the page\nstayed empty although the data was deployed. Hide define while Chart.js\nloads and restore it afterwards.\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>",
          "timestamp": "2026-11-11T16:17:17Z",
          "url": "https://github.com/e11bio/volara/commit/1d6c739922a133b7a4dedb8bdcbd774f0d1c423b"
        },
        "date": 1794414243110,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara.blockwise, volara.datasets (cold)",
            "value": 1541.987,
            "range": "± 102.594",
            "unit": "ms",
            "extra": "interpreter startup 15.2 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 39.417,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      },
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
          "id": "6adc55a273350ee0b1d02c42181fef88c6f04f5b",
          "message": "Load cloud-volume on first use and exclude its broken 12.16.0 release (#102)\n\ncloud-volume 12.16.0 imports crackle at module level without requiring\ncrackle-codec, so `import volara` failed on a fresh install. Only\nCloudVolumeWrapper needs cloud-volume: it is now loaded with lazy_loader\non first use, so importing volara no longer imports it, and 12.16.0 is\nexcluded from the dependency range.\n\nCo-authored-by: Claude Fable 5.2 <noreply@anthropic.com>",
          "timestamp": "2026-11-11T19:03:27Z",
          "url": "https://github.com/e11bio/volara/commit/6adc55a273350ee0b1d02c42181fef88c6f04f5b"
        },
        "date": 1794424318211,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara.blockwise, volara.datasets (cold)",
            "value": 1039.68,
            "range": "± 35.136",
            "unit": "ms",
            "extra": "interpreter startup 16.1 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 35.926,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      },
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
          "id": "96081936afc8ee6fadc09baaea3ccab9207f018a",
          "message": "Deploy docs on version tags and by hand only (#101)\n\nThe docs workflow now runs on a v* tag push and on workflow_dispatch, and\nnot on every push or pull request commit. A tag or a manual run on main\ndeploys; a manual run on any other branch builds without deploying, which\nis how a pull request's docs are checked once it is ready to merge.\n\nCo-authored-by: Claude Fable 5.2 <noreply@anthropic.com>",
          "timestamp": "2026-11-11T19:04:21Z",
          "url": "https://github.com/e11bio/volara/commit/96081936afc8ee6fadc09baaea3ccab9207f018a"
        },
        "date": 1794424394683,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara.blockwise, volara.datasets (cold)",
            "value": 1494.298,
            "range": "± 21.533",
            "unit": "ms",
            "extra": "interpreter startup 23.5 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 51.401,
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
          "id": "f7cbcdd6552d7dc8dded9dc23a855f1563fafd19",
          "message": "Bump version to 2.1.1\n\nCo-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>",
          "timestamp": "2026-11-11T19:08:59Z",
          "url": "https://github.com/e11bio/volara/commit/f7cbcdd6552d7dc8dded9dc23a855f1563fafd19"
        },
        "date": 1794424631663,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "import volara.blockwise, volara.datasets (cold)",
            "value": 1380.76,
            "range": "± 11.682",
            "unit": "ms",
            "extra": "interpreter startup 21.4 ms subtracted; n=10"
          },
          {
            "name": "pytest wall time",
            "value": 47.88,
            "unit": "s",
            "extra": "`uv run pytest --cov-report=xml` in the tests job"
          }
        ]
      }
    ]
  }
}