# Station sound model: plan to the end (2026-11-02)

Definition of done: a station that, from a live microphone, raises chainsaw and blast "check" items to the Control Room over LoRa, logs fire without alerting, with a measured false-alarm rate on held-out forest audio, and everything documented including what does not work. Estimates are mine and unmeasured. Status: phase 0 done.

## Phases
| # | Phase | Where | Est. | Passes when |
|---|---|---|---|---|
| 0 | Data: manifest, splits, corpus (21,219 items, 32.1 h), board speed test | laptop, board | DONE | |
| 1 | Training and evaluation code; floor baselines (spectrum + logistic regression, old patrol CNN); tested on a slice | laptop (CPU) | 0.5 to 1 day | code runs end to end; baselines scored on the fixed splits |
| 2 | Cloud job A: choose the one large pretrained model (zero-shot on held-out sources, rule below) and fit it as the upper bound | 1 GPU VM | 2 to 4 GPU h | table of 4 candidates, one chosen by the rule |
| 3 | Cloud job B: student sweep: CP-Mobile and depthwise control, 3 seeds each, 4 kHz band; ablations (8 kHz, no mixing, no mic simulation) | 1 to 2 GPU VMs in parallel | 6 to 10 GPU h | results per source with intervals |
| 4 | Distillation, only if the best student is clearly below the upper bound (more than the confidence interval) | 1 GPU VM | 3 to 6 GPU h | kept only if it beats the plain student on held-out sources |
| 5 | Freeze and score once: settings and thresholds fixed first, then held-out sources and false alarms per hour; QAT to int8; int8 vs float check | VM + laptop | 0.5 day | targets below, or the failure is reported |
| 6 | Firmware: C++ mel front end matching Python (test vectors), ESP_TF + ESP-NN classifier, event logic (chainsaw k of n windows, blast single, fire log only, wind = blind flag), event packet | laptop, board | 2 to 3 days | on-device and Python agree on at least 95 percent of test windows |
| 7 | Bench with the live microphone: play recordings at the mic at several loudness levels; hours of quiet-room and outdoor false-alarm logging; latency and power | board | 1 to 2 days | measured rates written down |
| 8 | Integration: gateway decodes events, serial-to-backend bridge, station registry, "check" item (source sensor), Stations page; relay hop demo; PMS5003 added (docs/14 phases 2 to 4) | laptop, website | 3 to 5 days | an event on the board appears as a check item on the map |
| 9 | Field test with the department: chainsaw at 50, 100, 200 and 400 m; recording at a prescribed burn; per-site threshold calibration | outdoors | depends on access | measured range and false alarms on site |
| 10 | Write-up: model card, results tables, limitations | laptop | 1 day | |

## Decision rules fixed now
- Large pretrained model (phase 2): candidates PaSST, EfficientAT MobileNet, PANNs CNN14, BEATs; chosen by best macro-F1 on sources held out of training; a tie goes to the smaller model.
- Student (phase 3): chosen by macro-F1 and false alarms per hour on held-out sources; a tie goes to fewer operations. Channels in multiples of 16; at most about 30 M operations (board: about 11 ns per operation, estimate 0.3 to 0.5 s per window).
- Distillation (phase 4) and the 8 kHz band (phase 3 ablation) only if the data says they help.
- Pass targets, to confirm with the owner: at most 1 false alarm per hour on held-out forest background at a threshold keeping 80 percent of chainsaw and blast windows; every class result reported with intervals, including bad ones.
- Product variant: items with an open licence only (chainsaw's open data is the new Freesound set); research variant: everything.

## Cloud and token budget
- GPU VMs: three projects with one L4 each; Spot (about 0.45 USD per hour); one VM per job, started only when its input is on it, stopped the moment the job ends; only VMs I created are stopped, by exact name. Upper estimate for phases 2 to 5: 15 to 25 GPU hours, roughly 7 to 12 USD (unmeasured).
- Claude: scripts run once and the summary read once; agents only for narrow jobs (the board test alone used about 800k tokens).

## Risks
- Chainsaw has 634 training items from four sources; fire is campfire-type audio with only three real wildfire recordings held out; wood and wind have no held-out source. Expect weak held-out results in places; they will be reported as they are.
- A new forest site can cost a lot of precision (a published forest gunshot model went from 0.85 to about 0.57 precision at an unseen site): phase 9 and per-site calibration matter.
- Spot GPUs can be preempted or out of stock; jobs save results as they go and resume.
- Fireworks and gunshots share the blast class by assumption; a separate-classes test is a small ablation if wanted.
- Not covered by this task: number plates are not blurred, and the image checks were dropped (camera-only reports).
