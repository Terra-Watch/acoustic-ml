# acoustic-ml

A sound classifier for the fixed TerraWatch forest stations. A station listens through a microphone and tells the Control Room when it hears a chainsaw or a blast. It is a "check" item for a person, never a confirmed event. Fire sound is logged and does not raise an alert.

This repository holds the data work, the training and evaluation code, the speed test on the station's chip, and the cloud job scripts. The firmware that runs the model is in the TerraWatch node firmware (`node-firmware/`, the sound classifier section of its README). Context: `docs/13-lookout-stations.md` and `docs/14-field-firmware-plan.md` in the TerraWatch repository.

## Classes and rules
Six classes: chainsaw, blast (gunshot and fireworks together), fire (any open flame), wood noise, wind, background. Chainsaw raises an alert when at least 2 of the last 3 one-second windows within 60 s are above the threshold; blast raises one when a single window is. Fire, wood and wind are logged only (wind also marks the microphone as blind). Reasoning and open questions: `DESIGN.md`. Phases, decision rules and pass targets: `PLAN.md`.

## Data
The corpus is built from public sound collections (ESC-50, FSD50K, FSC22, Freesound, a Belize forest soundscape set and several smaller sets) plus targeted Freesound searches. `manifests/SUMMARY.md` has the counts: 43,715 items listed, 43,055 usable after removing duplicates, label conflicts and unreadable files, with minutes per class and split. Splits are by recording source, so a held-out test source is never seen in training (`test_in` is the in-domain test, `test_ext` the external one). Wood and wind have no external test source, and fire has three real wildfire recordings.

**No audio is stored in this repository.** The licences differ per source (open, unknown, restricted). `manifests/` lists every item with its source and licence class; `scripts/` fetches and packs the audio on your machine (`data/` is git-ignored). Licence class of the usable items matters for any product use: use open-licence items only for that.

## What has been done
- Corpus, manifest and splits (`scripts/01` to `12`, `40`); a notebook from raw clips to processed features (`notebooks/`).
- Floor baselines: spectrum features with logistic regression, and a small depthwise CNN (`src/baseline.py`, `runs/`, `runs_cloud/dscnn_*`).
- Student sweep on cloud GPUs, a large pretrained upper bound (PaSST and PANNs, `runs_cloud/upper`), and fine-tuning of an AudioSet-pretrained MobileNet family model (`src/finetune.py`, `runs_cloud/vk*`).
- int8 export and a check against the float model (`src/export_onnx.py`, `src/tflite_eval.py`, `runs_cloud/export`). The model deployed on the station is `mn10` in int8 (4.3 MB, input 106 mel bands by 100 frames), kept in the node firmware repository.
- Speed test of int8 models on the ESP32-S3 with ESP-NN: `board-bench/RESULTS.md`. A C++ mel front end checked against the Python reference (`src/frontend_ref.py`, `src/frontend_check.py`, `board-bench/`).

## Results so far (validation and in-domain test, not field results)
| model | macro-F1 val | macro-F1 test_in | note |
|---|---|---|---|
| logistic regression on spectrum features | 0.354 | 0.332 | floor |
| small depthwise CNN, 3 seeds | 0.60 to 0.63 | 0.57 to 0.59 | `runs_cloud/dscnn_*` |
| PaSST fitted as the large-model upper bound | 0.686 | | `runs_cloud/upper` |
| fine-tuned MobileNet-family `mn10`, float | 0.771 | 0.733 | `runs_cloud/export/mn10b.log` |
| the same, int8 (deployed) | 0.733 | 0.702 | agreement with float 0.95 (val), 0.91 (test_in) |

Open problems, stated plainly:
- On sources the model has not seen (`test_ext`) the scores are clearly lower: the small CNN drops from 0.58 to 0.32 to 0.37.
- False alarms are far above the target of at most 1 per hour: measured on held-out forest background the rate is in the hundreds per hour for the models listed in `runs_cloud/*/eval.json`. Thresholds in the firmware are starting points, not calibrated in the field.
- A published forest gunshot model lost much precision at an unseen site; a field test with the forest department (chainsaw at 50 to 400 m, a prescribed burn, per-site thresholds) has not been done.
- Chainsaw has 634 training items from four sources; fire audio is mostly campfire-type.

## Layout
| Path | Content |
|---|---|
| `docs/design.md`, `docs/plan.md` | design, classes, data and licences, evaluation, phases and go/no-go gates |
| `scripts/` | numbered scripts: `01`-`06` download metadata and sets, `10` manifest, `11` pack corpus, `12` targeted Freesound, `20` notebook, `21` 8 kHz cache, `30`/`31` launch and run jobs on cloud VMs, `40` soundscapes, `50` front-end header for the firmware |
| `src/` | data loading, features, models, training, fine-tuning, evaluation with false alarms per hour, export to ONNX and TFLite, sliding-window and triggered evaluation |
| `tests/` | `test_core.py` |
| `manifests/` | item lists, licence classes, summary, listening-check lists (no audio) |
| `runs/`, `runs_cloud/` | evaluation outputs and logs of the runs (checkpoints are not kept in git) |
| `board-bench/` | ESP32-S3 speed test and its results; `RESULTS.md` |
| `cloud/` | the job script for the GPU VMs and the notes on them (`VMS.md`) |
| `notebooks/`, `reports/` | the raw-to-processed notebook and its figures |

## Run
```
python3 -m venv .venv && . .venv/bin/activate
pip install numpy scipy soundfile librosa torch torchaudio timm scikit-learn pandas pyyaml tqdm
python scripts/10_build_manifest.py        # needs the fetched audio under data/
python src/train.py --model student_ir --seed 0 --out runs/student_ir_s0
python src/evaluate.py --help
```
The cloud scripts (`scripts/30_launch.py`, `31_fleet.py`) read the project-to-account map from a local file outside git (`~/.config/terrawatch/gcp_projects.md`, or the path in `GCP_PROJECT_MAP`; one line per project: `| PROJECT_ID | ACCOUNT |`) and are not meant to run on their own. The package list above is the one the scripts import; there is no lock file yet.
