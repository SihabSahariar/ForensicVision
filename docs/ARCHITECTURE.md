# Architecture

The guiding constraint: **the restoration engine must be reusable without the
GUI**. Nothing under `core/`, `analysis/`, `restoration/`, `forensic/`,
`database/`, `reports/`, `ocr/` or `detection/` imports PyQt5, with one
deliberate exception (`core/image_utils.py`, which exists solely to convert
numpy arrays to `QImage`). That makes a CLI, an HTTP service or a video
pipeline a matter of writing a new front end.

```
main.py                     entry point, argument parsing, --check self-test
│
├── app/                    configuration, constants, paths, logging, version
│
├── core/                   framework-agnostic foundations
│   ├── image_io.py         ImageData; load/save preserving bit depth + alpha
│   ├── image_utils.py      numpy <-> QImage  (the only Qt import outside gui/)
│   ├── device.py           CUDA discovery, VRAM reporting
│   ├── case_manager.py     case lifecycle, evidence import
│   └── exceptions.py       one exception hierarchy for the whole application
│
├── analysis/               heuristic degradation indicators
│   ├── base.py             MetricResult, score mapping helpers
│   ├── blur|noise|jpeg|exposure|contrast|resolution|haze.py
│   ├── analyzer.py         orchestrator -> AnalysisReport
│   └── visualizations.py   histograms, ELA, clipping maps, spectra
│
├── restoration/            the engine
│   ├── base.py             RestorationModel, ModelInfo, ParamSpec, WeightSpec
│   ├── registry.py         name -> factory; lazy instantiation
│   ├── torch_base.py       device/precision/tiling plumbing for neural models
│   ├── tiling.py           overlapping tiles, feathered blend, OOM backoff
│   ├── weights.py          explicit, digest-verified weight installation
│   ├── pipeline.py         Pipeline, PipelineRunner, per-step hashing
│   ├── auto_engine.py      analysis -> recommended pipeline + reasons
│   ├── classical/          deterministic operators
│   └── realesrgan|swinir|restormer|nafnet|dncnn|fbcnn/  arch.py + model.py
│
├── forensic/               hashing, metadata, provenance, safe mode
├── database/               SQLAlchemy models, engine, repository
├── workers/                QThread wrappers - the only threading in the app
├── reports/                ReportLab PDF generation
├── ocr/ detection/         optional integrations
└── gui/                    PyQt5 - window, inspector, dialogs, viewers
    ├── main_window.py      menus, toolbar, actions, worker orchestration
    ├── inspector.py        the single tabbed dock hosting every panel
    ├── image_viewer.py     QGraphicsView canvas; emits contextMenuRequested
    └── <panel>.py          case / analysis / restoration / history / log
```

---

## Key decisions

### Models register factories, not instances

`ModelRegistry.register(info, factory)` stores a `ModelInfo` plus a zero-arg
callable. Listing every model - which the Model Manager does at start-up -
therefore costs nothing; PyTorch is imported only when a neural model is
actually instantiated. This is what keeps start-up fast on a machine with no
GPU, and what lets the application start with no ML stack at all.

### The repository owns every database session

The GUI never holds a live ORM session. `CaseRepository` opens a short-lived
session per call and returns detached objects. Worker threads write results
through the same repository, so there is no cross-thread session sharing - the
most common source of bugs when background work persists results.

### One persistence path for derivatives

`workers/restoration_worker.py::persist_restoration` is the only function that
writes a derivative. The interactive path and the batch path both call it, so a
derivative on disk always has a matching sidecar, derivative row, per-step
history and audit entry. There is no second code path that could produce a file
without its record.

### Nothing long-running touches the GUI thread

Every operation that can exceed a frame runs in a `BaseWorker` (a `QThread`
subclass) emitting `progress`, `status`, `finished`, `error` and `cancelled`.
`MainWindow._run_worker` wires those to the progress bar and status line and
handles cleanup uniformly, so no individual handler can leak a running thread.

Batch processing runs sequentially on one worker rather than in parallel: the
bottleneck is GPU inference, which serialises anyway, and parallelism would
only raise peak VRAM and the chance of an out-of-memory failure.

### Hashes chain through the pipeline

Each `StepResult` records the digest of exactly what entered and left that
step, so step *N*'s output digest equals step *N+1*'s input digest. The chain
from original evidence to final derivative is verifiable link by link, not just
end to end. `tests/test_integration.py` asserts this property.

### One dock, and a context menu

Six separate docks consumed roughly 600 px of width and 220 px of height,
squeezing the image into a small central box and clipping the taller panels -
the analysis dock showed five of its nine indicators. They are now one tabbed
inspector (`gui/inspector.py`), and most actions are reachable from the image
viewer's context menu, so the panels can stay closed entirely.

The viewer does not build that menu itself. It emits `contextMenuRequested`
with a global position and `MainWindow` populates a `QMenu` from the same
`QAction` objects the menu bar and toolbar use, so enabled/disabled state and
shortcuts stay consistent across all three without duplication.

### Two operator classes, everywhere

`ModelKind.CLASSICAL` versus `ModelKind.NEURAL` and the `may_synthesise` flag
propagate from `ModelInfo` into the restoration panel, the pipeline editor's
warning banner, the confirmation dialog, the case explorer's colouring, the
provenance record, the derivative database row and the PDF report. The
distinction is declared once and surfaces everywhere it matters.

---

## Extension points

### A new analysis indicator

1. Add `analysis/<name>.py` returning a `MetricResult` with its raw
   `measurements` and a `method` string naming the estimator.
2. Add the key to `DegradationKey` and `DEGRADATION_LABELS`.
3. Register it in `DegradationAnalyzer.analyze`'s step list.

The analysis panel, detail dialog and report pick it up automatically.

### A new restoration model

1. `restoration/<name>/arch.py` - the network, with upstream layer naming so
   official checkpoints load.
2. `restoration/<name>/model.py` - subclass `TorchRestorationModel` (or
   `RestorationModel` for a classical operator) and populate `ModelInfo`.
   `license_name`, `repository`, `method` and `may_synthesise` are surfaced to
   the investigator and printed in reports, so they are not optional in
   practice.
3. Add `register_<name>()` and list it in `register_all_models`.

Declared `ParamSpec` entries generate the GUI controls; there is no per-model
UI code.

### A new front end

```python
from restoration import register_all_models
from restoration.auto_engine import AutoRestorationEngine
from restoration.pipeline import PipelineRunner
from analysis import analyze_image
from core.image_io import load_image

register_all_models()
image = load_image("frame.jpg")
report = analyze_image(image)
pipeline = AutoRestorationEngine().recommend(report).pipeline
result = PipelineRunner(device="auto").run(image, pipeline)
```

No Qt import is required for any of that.

---

## Data model

```
Case ──┬── Evidence ──┬── Derivative ──┬── Derivative      (parent_derivative_id)
       │              │                └── AnalysisRecord
       │              ├── AnalysisRecord
       │              └── ProcessingStep
       ├── ReportRecord
       └── AuditEvent
```

One SQLite file per case, inside the case directory, so a case folder is
self-contained and can be archived or handed over as a unit. `case.json` mirrors
the essentials in human-readable form.

Derivatives form a tree through `parent_derivative_id`, which is what the
Processing History panel renders and what lets a derivative-of-a-derivative
trace back to the original.

---

## Threading model

| Thread | Work |
|---|---|
| GUI | Widgets, painting, signal handling. Never blocks. |
| Worker (`BaseWorker`) | Import, analysis, inference, reports, batch, downloads. One at a time. |
| Qt internals | Repaint and event delivery. |

`MainWindow` permits one worker at a time and disables the actions that would
start another. Cancellation is cooperative: workers poll `is_cancelled()` and
pass it into engine calls that accept a cancel predicate, so a cancelled
operation stops at a defined point rather than being killed mid-write.
