"""Regenerate notebooks/kaggle_run.ipynb from readable cell sources.

Editing notebook JSON by hand is miserable, so the Kaggle runner is kept here as plain
text and the .ipynb is generated:

    python scripts/make_kaggle_notebook.py

Edit this file, never the notebook - a regeneration overwrites it.
"""

import json
from pathlib import Path

CELLS = []


def _source(text):
    """nbformat wants one string per line, each keeping its newline but the last."""
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text):
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": _source(text)})


def code(text):
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {"trusted": True},
            "outputs": [],
            "source": _source(text),
        }
    )


md(r"""
# Thesis — full study on Kaggle

**Run all** drives the entire pipeline: environment, data, every specification step from
4 to 18, then a results bundle you can download or attach to the next session.

## Before you press Run All

| Setting | Value | Why |
|---|---|---|
| **Accelerator** | GPU T4 ×2 or P100 | The study is not viable on CPU |
| **Internet** | On | Needed to clone the repo and pip-install |
| **Persistence** | Files only | Keeps `/kaggle/working` between sessions |
| **Add-ons → Secrets** | `GITHUB_TOKEN` | Only if the repo is private |
| **Input datasets** | see below | Faster and works with Internet off |

Attach these two datasets (**+ Add Input → Datasets**), or leave Internet on and the
notebook will download them:

- `mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images` — the primary set
- `ashkhagan/figshare-brain-tumor-dataset` — external validation, Step 17

## Pick a profile

| Profile | What it does | Rough cost |
|---|---|---|
| `smoke` | 1 epoch, 3 batches per stage. Proves every stage runs and every checkpoint hand-off works. | ~30–45 min |
| `fast` | One seed, 8 epochs. Real numbers, wrong protocol. | most of one session |
| `full` | Three seeds, the fixed protocol. **The only reportable profile.** | several sessions |

Two things dominate the cost and neither is the GPU. Feature extraction and every
quantum stage — Step 12, `baseline_fixed_qcnn`, arms 7 and 8 — run their circuits on a
CPU simulator, so they take the same time on a T4 as on a laptop. Measured here: about
0.09 s per image through all three branches. Step 12 across three seeds is the single
largest item in the study.

**Run `smoke` first.** It catches every wiring problem for the price of one coffee,
before you spend GPU quota. Then switch `PROFILE` to `full` and run again — the two keep
separate output trees (`logs/_smoke/` vs `logs/`) and separate feature caches, so neither
can be mistaken for the other and smoke's completion markers cannot make `full` skip a
stage.

`full` will not finish in one 12-hour session, and it does not need to. Every stage
writes a completion marker, so re-running this notebook in a new session skips what is
already done and picks up training runs mid-epoch from `last.ckpt`. Cell 9 tells you how
to hand one session's work to the next.

## Where the results go

Everything lands under `/kaggle/working/thesis/logs/`, which survives the session once
you **Save Version**. The last cells also write a compact `thesis_results_*.zip`
(tables, figures, summaries, configs — no checkpoints) and can push it to a
`kaggle-results` branch on GitHub.
""")

md("## 1 · Settings")

code(r'''
# ---------------------------------------------------------------------------- settings
PROFILE        = "smoke"      # "smoke" | "fast" | "full"
BUDGET_HOURS   = 11.0          # stop and save before Kaggle's 12h cut-off
NUM_WORKERS    = 2             # dataloader workers; Kaggle GPU boxes have 4 cores

GITHUB_USER    = "Biswadev-9"
REPO_NAME      = "thesis"
BRANCH         = "main"
PRIVATE_REPO   = False         # if you make the repo private, set True and add a
                               # GITHUB_TOKEN secret under Add-ons -> Secrets

RUN_TESTS      = True          # ~305 tests, ~3 min; catches a broken environment early
PUSH_RESULTS   = False         # push the results bundle to a kaggle-results branch
PRUNE_CHECKPOINTS = False      # see cell 11 before turning this on

EXTRA_ARGS     = []            # e.g. ["--skip", "step11"] or ["--only", "step16_internal"]
# ---------------------------------------------------------------------------------------

WORK = "/kaggle/working"
PROJECT = f"{WORK}/{REPO_NAME}"
print(f"profile={PROFILE}  budget={BUDGET_HOURS}h  project={PROJECT}")
''')

md("## 2 · What machine did we get?")

code(r'''
import os, shutil, subprocess, sys, textwrap
from pathlib import Path

print(sys.version)
print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout or
      "no GPU visible -- set Accelerator to GPU in the sidebar, or expect this to crawl")
print(f"cpus={os.cpu_count()}  working disk free={shutil.disk_usage(WORK).free/1e9:.0f} GB")

for d in sorted(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").is_dir() else []:
    print("input:", d)
''')

md("""## 3 · Clone the repository

Cloned fresh every session, so the code always matches `main`. Results are *not* stored
in the repo — they come back from the previous session's output in cell 6.""")

code(r'''
import os, shutil, subprocess
from pathlib import Path

token = ""
if PRIVATE_REPO:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")

url = (f"https://{token}@github.com/{GITHUB_USER}/{REPO_NAME}.git" if token
       else f"https://github.com/{GITHUB_USER}/{REPO_NAME}.git")

if Path(PROJECT, ".git").is_dir():
    print("Repository already cloned; pulling instead.")
    subprocess.run(["git", "-C", PROJECT, "pull", "--ff-only"], check=False)
else:
    shutil.rmtree(PROJECT, ignore_errors=True)
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, PROJECT], check=True)

os.chdir(PROJECT)
print(subprocess.run(["git", "log", "-1", "--oneline"], capture_output=True, text=True).stdout)
''')

md("""## 4 · Install what Kaggle is missing

Kaggle already ships CUDA-enabled torch, torchvision, pandas, scikit-learn, scipy,
matplotlib, seaborn and Pillow. Reinstalling torch from `requirements.txt` would replace
a working CUDA build with whatever pip resolves, so only the genuinely missing packages
are installed here.""")

code(r'''
# hydra-optuna-sweeper is in requirements.txt but nothing in the pipeline sweeps
# hyperparameters, and it drags in optuna. Left out on purpose.
MISSING = [
    "lightning>=2.0.0", "torchmetrics>=0.11.4",
    "hydra-core==1.3.2", "hydra-colorlog==1.2.0",
    "rootutils", "rich",
    "pennylane>=0.40",
    "SimpleITK", "opencv-python-headless", "scikit-image", "h5py",
]
packages = " ".join(f'"{p}"' for p in MISSING)
!pip install -q {packages}
''')

code(r'''
import importlib, torch, lightning, pennylane, hydra, cv2, SimpleITK, h5py, rootutils
print("torch      ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("lightning  ", lightning.__version__)
print("pennylane  ", pennylane.__version__)
print("hydra      ", hydra.__version__)

# The quantum branches stand or fall on TorchLayer's backward pass, so prove it here
# rather than four hours into a training run.
import pennylane as qml
dev = qml.device("default.qubit", wires=2)
@qml.qnode(dev, interface="torch", diff_method="backprop")
def circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(2))
    qml.BasicEntanglerLayers(weights, wires=range(2))
    return [qml.expval(qml.PauliZ(i)) for i in range(2)]
layer = qml.qnn.TorchLayer(circuit, {"weights": (1, 2)})
out = layer(torch.rand(4, 2, requires_grad=True)).sum()
out.backward()
print("pennylane TorchLayer fwd+bwd: ok")
''')

md("""## 5 · Wire up the data

The datamodule wants `data/raw/bt_mri/**/{Training,Testing}/<class>/` and, for Step 17,
`.mat` files under `data/raw/figshare/`. Attached Kaggle inputs are symlinked into place —
no copying, no disk cost. If nothing is attached and Internet is on, they are downloaded
instead.""")

code(r'''
import os, subprocess
from pathlib import Path

RAW = Path(PROJECT, "data", "raw")
RAW.mkdir(parents=True, exist_ok=True)
INPUT = Path("/kaggle/input")


def find_primary():
    """A directory holding both Training/ and Testing/ somewhere beneath it."""
    for base in sorted(INPUT.glob("*")) if INPUT.is_dir() else []:
        for training in list(base.rglob("Training"))[:8]:
            if training.is_dir() and (training.parent / "Testing").is_dir():
                return base
    return None


def find_figshare():
    for base in sorted(INPUT.glob("*")) if INPUT.is_dir() else []:
        if next(base.rglob("*.mat"), None) is not None:
            return base
    return None


def link(source, name):
    target = RAW / name
    if target.is_symlink() or target.exists():
        print(f"{name}: already present at {target}")
        return
    target.symlink_to(source, target_is_directory=True)
    print(f"{name}: linked -> {source}")


primary, figshare = find_primary(), find_figshare()

if primary:
    link(primary, "bt_mri")
else:
    print("Primary dataset not attached; trying to download it instead.")
    if subprocess.run(["bash", "scripts/download_data.sh"]).returncode != 0:
        raise SystemExit(
            "Could not obtain the primary dataset. Add it as an input:\n"
            "  + Add Input -> Datasets -> "
            "mohamadabouali1/mri-brain-tumor-dataset-4-class-7023-images"
        )

if figshare:
    link(figshare, "figshare")
else:
    print("Figshare not attached -- Step 17 will be skipped unless it downloads below.")
    subprocess.run(["kaggle", "datasets", "download", "-d",
                    "ashkhagan/figshare-brain-tumor-dataset",
                    "-p", str(RAW / "figshare"), "--unzip"], check=False)

n_images = sum(1 for _ in (RAW / "bt_mri").rglob("*.jpg"))
print(f"\n{n_images} jpg files visible under data/raw/bt_mri")
''')

md("""## 6 · Carry the previous session forward

Kaggle mounts a notebook's saved output at `/kaggle/input/<slug>/`. Attach the previous
version's output as an input and this cell copies its `logs/`, split table, materialised
recipes and feature caches back in — so the pipeline resumes instead of starting over.

Nothing to do on the first run; it just prints that it found nothing.""")

code(r'''
from pathlib import Path

RESTORE_FROM = None
for manifest in Path("/kaggle/input").glob(f"*/{REPO_NAME}/logs/pipeline/manifest.json"):
    RESTORE_FROM = str(manifest.parents[2])
    break

print(f"restoring from: {RESTORE_FROM}" if RESTORE_FROM else
      "no previous session attached -- starting from scratch")
''')

md("""## 7 · Sanity tests

Three minutes against ten hours of GPU time. These cover leakage, the corrected focal
loss, protocol conformance, the gate's per-pixel softmax, the quantum mixture arithmetic
and the once-only test lock — exactly the things that break silently.

Expect around 300 to pass. `test_train_resume` is flaky and may fail: it is boilerplate
inherited from the project template and asserts that epoch 1 beats epoch 0, which holds
on MNIST but not on 1 % of a brain-MRI split. It does not block the run. Anything *else*
failing means the environment is wrong — fix that before spending GPU quota.""")

code(r'''
if RUN_TESTS:
    # No -x: one flaky template test must not stop the notebook.
    !python -m pytest tests/ -q --no-header
else:
    print("skipped")
''')

md("""## 8 · Run the study

One command. It works out the stage order, pins each run's output directory, feeds the
Step 6 / 8 / 13 / 14 selections into the stages downstream, skips anything already
finished, and caps training runs so they stop and save rather than getting killed by the
session limit.

Watch the `[run] / [ok] / [cached] / [skip]` lines — progress bars are off because sixty
runs of Rich redrawing would overflow the notebook's output buffer. Per-run detail goes to
each stage's `stage.log`.""")

code(r'''
import shlex, subprocess, sys

cmd = [sys.executable, "scripts/kaggle_pipeline.py",
       "--profile", PROFILE,
       "--budget-hours", str(BUDGET_HOURS),
       "--num-workers", str(NUM_WORKERS),
       "--keep-going"]
if RESTORE_FROM:
    cmd += ["--restore-from", RESTORE_FROM]
cmd += EXTRA_ARGS

print(" ".join(shlex.quote(c) for c in cmd), "\n")
returncode = subprocess.run(cmd).returncode

# 0 finished · 1 a required stage failed · 2 out of time, re-run to continue
print(f"\nexit code: {returncode}")
''')

md("## 9 · What happened")

code(r'''
from pathlib import Path
from IPython.display import Markdown, display

report = Path(PROJECT, "logs", "pipeline", "REPORT.md")
display(Markdown(report.read_text(encoding="utf-8") if report.is_file() else "_no report_"))
''')

code(r'''
# Anything left to do? Re-running this notebook picks up exactly here.
!python scripts/kaggle_pipeline.py --list --profile {PROFILE} | tail -25
''')

md("""## 10 · Keep the results

Three places, in increasing order of durability:

1. **Save Version** (top right) — persists all of `/kaggle/working`, checkpoints
   included. Attach that output as an input next session to resume.
2. **The zip below** — tables, figures, summaries and resolved configs. Small enough to
   download from the Output tab.
3. **The GitHub push** — set `PUSH_RESULTS = True` in cell 1. Lands on a
   `kaggle-results` branch, so it outlives your Kaggle account.""")

code(r'''
from pathlib import Path

bundles = sorted(Path(WORK).glob("thesis_results_*.zip"))
for b in bundles:
    print(f"{b}  {b.stat().st_size/1e6:.1f} MB  -- download it from the Output tab")

if not bundles:
    print("No bundle found. Cell 8 writes one on every exit; if it is missing, that cell "
          "did not get far enough. Re-run it, or build one now:")
    print("  !python scripts/kaggle_pipeline.py --list --profile", PROFILE)
''')

code(r'''
import subprocess, time
from pathlib import Path

if PUSH_RESULTS:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    branch = "kaggle-results"
    out = Path(PROJECT, "results", f"kaggle_{stamp}")
    out.mkdir(parents=True, exist_ok=True)

    # Text and figures only. Checkpoints do not belong in git.
    keep = {".json", ".csv", ".png", ".md", ".yaml", ".log"}
    logs = Path(PROJECT, "logs")
    for path in logs.rglob("*"):
        if path.is_file() and path.suffix in keep and path.stat().st_size < 20_000_000:
            dest = out / path.relative_to(logs)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(path.read_bytes())

    run = lambda *a: subprocess.run(["git", "-C", PROJECT, *a], check=False)
    run("config", "user.email", "kaggle@example.com")
    run("config", "user.name", "kaggle-runner")
    run("checkout", "-B", branch)
    run("add", "-f", "results")
    run("commit", "-m", f"Kaggle {PROFILE} run {stamp}")
    run("push", "-u", "origin", branch)
    print(f"pushed to {branch}")
else:
    print("PUSH_RESULTS is off")
''')

md("""## 11 · Optional: shrink the saved output

A `full` run leaves roughly 5–8 GB of checkpoints, which makes Save Version slow. This
deletes the ones nothing downstream reads — every Step 9 baseline except EfficientNet-B0
and ViT, and every Step 11 arm.

**It is not free.** Steps 19–23 (explainability, quantum-advantage analysis) are not
implemented yet and will want those checkpoints when they are. Leave this off unless disk
is actually a problem.""")

code(r'''
from pathlib import Path

if PRUNE_CHECKPOINTS:
    keep_markers = ("step10_classical", "step12_adaptive_quantum", "step15_final",
                    "baseline_efficientnet_b0", "baseline_vit")
    freed = 0
    for ckpt in Path(PROJECT, "logs", "train", "runs").rglob("*.ckpt"):
        if any(marker in ckpt.as_posix() for marker in keep_markers):
            continue
        freed += ckpt.stat().st_size
        ckpt.unlink()
    print(f"freed {freed/1e9:.2f} GB")
else:
    print("PRUNE_CHECKPOINTS is off -- nothing deleted")
''')

md("""---

## Resuming in a new session

1. **Save Version** on the finished session and wait for it to commit.
2. Open the notebook again → **+ Add Input → Notebook Output** → pick that version.
3. Run all. Cell 6 finds it, cell 8 skips every finished stage.

Repeat until `--list` shows every stage ticked.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `exit code: 2` | Hit the time budget | Expected on `full`. Save Version and resume. |
| `no GPU visible` | Accelerator not set | Sidebar → Accelerator → GPU. |
| Step 17 skipped | Figshare dataset absent | Attach `ashkhagan/figshare-brain-tumor-dataset`. |
| `already been evaluated on the internal test set` | Step 16 ran before | That is the point — it runs once. `EXTRA_ARGS = ["--force-test"]` overrides it, and the summary records that it was forced. |
| A stage keeps failing | Genuine bug | `EXTRA_ARGS = ["--only", "<stage id>", "--progress"]` to see it in isolation. |
| Out of disk | Checkpoints | Cell 11. |
""")

nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "isGpuEnabled": True,
            "isInternetEnabled": True,
            "language": "python",
            "sourceType": "notebook",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

target = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_run.ipynb"
target.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"wrote {target} ({len(CELLS)} cells)")
