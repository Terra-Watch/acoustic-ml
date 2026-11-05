# ESP32-S3 speed test for int8 models (measured 2026-11-02)

Board: ESP32-S3 (QFN56, rev v0.2), 240 MHz, 16 MB flash, 8 MB **octal** PSRAM, on the UART USB port (/dev/cu.usbmodem5B5E0265961). Input 1x40x101x1 int8, output 1x6 int8, random weights, per-tensor int8, ops ADD / CONV_2D / DEPTHWISE_CONV_2D / MEAN / RESHAPE (all five allocated and ran in both libraries, so they are supported on the board). Time is `micros()` around `Invoke()`; n = number of timed invokes (the big models take seconds, so n is small; repeat runs of the same case differed by under 1 percent).

## Results
| model | params | MACs | int8 KB | channels | library and kernels | arena KB (used) | ms per inference | ns per MAC |
|---|---|---|---|---|---|---|---|---|
| m5 | 16,754 | 4.97 M | 33.1 | 4 to 32, odd widths | TanakaMasayuki 1.0.0, reference | 139 | 3,317 | 668 |
| m30 | 107,740 | 29.31 M | 122.3 | 11 to 88, odd widths | same | 371 | 17,600 (n=1) | 600 |
| m100 | 373,890 | 98.64 M | 382.7 | 21 to 168, odd widths | same | 702 | 56,931 (n=1) | 577 |
| m5 | 16,754 | 4.97 M | 33.1 | odd | ESP_TF 2.1.1 + ESP-NN (S3 assembly) | 170 | 318 (PSRAM arena), 291 (internal arena) | 64 / 59 |
| a8 | 59,134 | 16.44 M | 74.7 | all multiples of 8 | same | 460 | 285 (n=3) | 17.4 |
| m30 | 107,740 | 29.31 M | 122.3 | odd | same | 442 | 1,139 (n=1, twice) | 38.9 |
| a16 | 220,790 | 58.89 M | 232.9 | all multiples of 16 | same | 739 | 650 (n=2) | 11.0 |
| m100 | 373,890 | 98.64 M | 382.7 | odd | same | 790 | 2,269 (n=1, twice) | 23.0 |

- **Speed-up from ESP-NN: 10x (m5), 15x (m30), 25x (m100)** against the old library's reference kernels. The libraries differ (old and new TFLite-Micro), so this is "library A reference vs library B ESP-NN", not a clean on/off of the same code. An ESP_TF build without ESP-NN is set up as env `bench_esptf_ref` but was not run.
- **Channel counts matter as much as MACs.** a8 (16 M MACs) takes 285 ms and a16 (59 M MACs) 650 ms, both faster than m30 (29 M MACs, odd widths) at 1,139 ms. ns per MAC falls from 39 (odd widths) to 11 to 17 (multiples of 8 or 16). Cause not isolated: I did not test it separately from depth and size, but it fits the ESP-NN fast paths that want channels in multiples of 4 or 8.
- **Memory placement hardly matters:** with the reference kernels flash weights, RAM weights, PSRAM arena and internal arena all gave 3.31 to 3.32 s for m5. With ESP-NN an internal arena was 9 percent faster (291 vs 318 ms).
- **Arena:** models with more than about 170 KB of arena need PSRAM (about 370 KB of internal RAM was free at boot, and the node firmware uses some).

## Reading (estimates, not measurements, are marked)
- The fastest measured point is 285 ms (a8, 16 M MACs); nothing was measured below that, so no measured model fits 100 ms or 250 ms. Estimate from a8 and a16: time is about 144 ms plus 8.6 ms per M MACs for this 33-layer structure. That puts 250 ms at roughly 12 M MACs and 500 ms at roughly 41 M MACs; 100 ms is below the fixed cost of this depth (a shallower net would be needed).
- A CP-Mobile-style student (the DCASE baseline: 61k parameters, 29.4 M MACs), built with channels in multiples of 8 and run with ESP-NN: about 400 ms by this fit (estimate). With the old reference-kernel library: about 17.6 s (measured for the 29.3 M-MAC m30). So the library choice (ESP-NN) decides whether any student of that size is usable in a 1 s window; the patrol project's tanakamasayuki library cannot run it.
- Not measured: the DCASE architecture itself, int8 accuracy, power draw, or timing together with the audio front end (mel features) and LoRa.

## Setup facts found on the way (matter for the node firmware too)
- This board's serial is UART0 on the UART USB port: build with `-D ARDUINO_USB_CDC_ON_BOOT=0` (with 1, nothing reaches this port).
- PSRAM is octal: `board_build.arduino.memory_type = qio_opi` and `-D BOARD_HAS_PSRAM`. The default quad setting (also `board_build.psram = enabled`, which the node firmware uses) logs "PSRAM ID read error" and finds none, so the node firmware has no PSRAM.
- The old TFLite-Micro crashes if you delete an interpreter after a failed AllocateTensors.

## Files and how to rerun
build_models.py (models/*.tflite, src/models_data.h), src/main.cpp, platformio.ini (envs bench, bench_esptf_ref, bench_esptf_nn), read_results.py.
`pio run -e bench_esptf_nn -t upload --upload-port /dev/cu.usbmodem5B5E0265961`, then `~/.platformio/penv/bin/python read_results.py`. The older env `bench` (old library) runs every model with the weights and arena variants; `-DONLY_M100` for a separate m100 boot.
