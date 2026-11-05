// Times the real int8 mn10_as exports (per-tensor and per-channel) on the ESP32-S3 with ESP_TF + ESP-NN, arena in PSRAM, weights in flash, and prints the 6 output values for a fixed input
// so they can be compared with the same model run on a computer.
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <ESP_TF.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "models_mnreal.h"

struct Model { const char* name; const unsigned char* data; size_t len; };
static const Model kModels[] = {{"mn10real_pt", model_mn10real_pt, model_mn10real_pt_len}};
static String results[8]; static int nResults = 0;

static void bench(const Model& m0, size_t arenaBytes, int nTimed, bool weightsInPsram) {
    char line[360]; Model m = m0;
    if (weightsInPsram) { uint8_t* w = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, m0.len, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)); if (!w) { Serial.println("WEIGHTS_ALLOC_FAILED"); return; } memcpy(w, m0.data, m0.len); m.data = w; }
    uint8_t* arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, arenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!arena) { snprintf(line, sizeof(line), "BENCH %s ARENA_ALLOC_FAILED", m.name); results[nResults++] = line; Serial.println(line); return; }
    static tflite::MicroMutableOpResolver<12> resolver; static bool added = false;
    if (!added) { resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddAdd(); resolver.AddMul(); resolver.AddMean(); resolver.AddReshape(); resolver.AddHardSwish();
                  resolver.AddFullyConnected(); resolver.AddLogistic(); resolver.AddPad(); resolver.AddReduceMax(); added = true; }
    tflite::MicroInterpreter* interp = new tflite::MicroInterpreter(tflite::GetModel(m.data), resolver, arena, arenaBytes);
    if (interp->AllocateTensors() != kTfLiteOk) { snprintf(line, sizeof(line), "BENCH %s ALLOCATE_TENSORS_FAILED (arena too small or an op is missing)", m.name); results[nResults++] = line; Serial.println(line); return; }
    TfLiteTensor* in = interp->input(0);
    for (size_t i = 0; i < in->bytes; ++i) in->data.int8[i] = static_cast<int8_t>((i * 37) % 255 - 127);
    uint32_t mn = 0xFFFFFFFF, mx = 0; uint64_t sum = 0; bool ok = true;
    for (int i = 0; i < nTimed; ++i) { const uint32_t t0 = micros(); if (interp->Invoke() != kTfLiteOk) { ok = false; break; } const uint32_t dt = micros() - t0; sum += dt; if (dt < mn) mn = dt; if (dt > mx) mx = dt; }
    TfLiteTensor* out = interp->output(0);
    snprintf(line, sizeof(line), "BENCH %s weights=%s %s n=%d min=%.1fms mean=%.1fms max=%.1fms arena_used=%uB model=%uB out=[%d,%d,%d,%d,%d,%d]", m.name, weightsInPsram ? "psram" : "flash", ok ? "OK" : "INVOKE_FAILED", nTimed, mn / 1000.0, sum / (double)nTimed / 1000.0, mx / 1000.0,
             (unsigned)interp->arena_used_bytes(), (unsigned)m.len, out->data.int8[0], out->data.int8[1], out->data.int8[2], out->data.int8[3], out->data.int8[4], out->data.int8[5]);
    results[nResults++] = line; Serial.println(line);
}

void setup() {
    Serial.begin(115200); delay(3000); Serial.println("SETUP_START");
    results[nResults++] = String("INFO cpu_mhz=") + getCpuFrequencyMhz() + " psram=" + (psramFound() ? "yes" : "no") + " free_psram=" + heap_caps_get_free_size(MALLOC_CAP_SPIRAM); Serial.println(results[0]);
    const size_t kArena = 3 * 1024 * 1024;
    bench(kModels[0], kArena, 3, true);      // weights copied to PSRAM
    bench(kModels[0], 700 * 1024, 3, false); // weights read from flash, for comparison
}
void loop() { for (int i = 0; i < nResults; ++i) Serial.println(results[i]); Serial.println("BENCH_DONE"); delay(8000); }
