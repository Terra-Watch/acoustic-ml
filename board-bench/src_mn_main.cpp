// Times the MobileNetV3 proxies (see build_mn.py) on the ESP32-S3 with ESP_TF + ESP-NN. Arena in PSRAM, weights in flash. Results repeat every 8 s so a late serial monitor still sees them.
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <ESP_TF.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "models_mn.h"

struct Model { const char* name; const unsigned char* data; size_t len; };
#ifdef ONLY_MN04
static const Model kModels[] = {{"mn04p", model_mn04p, model_mn04p_len}};
#elif defined(ONLY_MN10)
static const Model kModels[] = {{"mn10p", model_mn10p, model_mn10p_len}};
#else
static const Model kModels[] = {{"mn04p", model_mn04p, model_mn04p_len}, {"mn10p", model_mn10p, model_mn10p_len}};
#endif
static String results[8]; static int nResults = 0;

static void bench(const Model& m, size_t arenaBytes, int nTimed) {
    char line[300];
    uint8_t* arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, arenaBytes, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!arena) { snprintf(line, sizeof(line), "BENCH %s ARENA_ALLOC_FAILED (%u bytes of PSRAM)", m.name, (unsigned)arenaBytes); results[nResults++] = line; Serial.println(line); return; }
    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddAdd(); resolver.AddMul(); resolver.AddMean(); resolver.AddReshape(); resolver.AddHardSwish();
    tflite::MicroInterpreter* interp = new tflite::MicroInterpreter(tflite::GetModel(m.data), resolver, arena, arenaBytes);
    if (interp->AllocateTensors() != kTfLiteOk) { snprintf(line, sizeof(line), "BENCH %s ALLOCATE_TENSORS_FAILED (arena %u bytes too small or an op is missing)", m.name, (unsigned)arenaBytes); results[nResults++] = line; Serial.println(line); return; }
    TfLiteTensor* in = interp->input(0);
    for (size_t i = 0; i < in->bytes; ++i) in->data.int8[i] = static_cast<int8_t>((i * 37) % 255 - 127);
    uint32_t mn = 0xFFFFFFFF, mx = 0; uint64_t sum = 0; bool ok = true;
    for (int i = 0; i < nTimed; ++i) { const uint32_t t0 = micros(); if (interp->Invoke() != kTfLiteOk) { ok = false; break; } const uint32_t dt = micros() - t0; sum += dt; if (dt < mn) mn = dt; if (dt > mx) mx = dt; }
    snprintf(line, sizeof(line), "BENCH %s %s n=%d min=%.1fms mean=%.1fms max=%.1fms arena_used=%uB model=%uB", m.name, ok ? "OK" : "INVOKE_FAILED", nTimed, mn / 1000.0, sum / (double)nTimed / 1000.0, mx / 1000.0, (unsigned)interp->arena_used_bytes(), (unsigned)m.len);
    results[nResults++] = line; Serial.println(line);
}

void setup() {
    Serial.begin(115200); delay(3000); Serial.println("SETUP_START");
    results[nResults++] = String("INFO cpu_mhz=") + getCpuFrequencyMhz() + " psram=" + (psramFound() ? "yes" : "no") + " psram_size=" + ESP.getPsramSize() + " free_psram=" + heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    Serial.println(results[0]);
    const size_t kArena = 3 * 1024 * 1024;
    for (const Model& m : kModels) bench(m, kArena, 3);
}

void loop() { for (int i = 0; i < nResults; ++i) Serial.println(results[i]); Serial.println("BENCH_DONE"); delay(8000); }
