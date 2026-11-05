// Times three int8 models on the ESP32-S3 with the same TFLite-Micro library the patrol project used. Results repeat every 8 s so a late serial monitor still sees them.
#include <Arduino.h>
#include <esp_heap_caps.h>

#ifdef USE_ESP_TF                     // nickjgniklu/ESP_TF: newer TFLite-Micro with Espressif's ESP-NN kernels (when built with -DESP_NN ...)
#include <ESP_TF.h>
#else                                 // tanakamasayuki/TensorFlowLite_ESP32 1.0.0: older TFLite-Micro, reference kernels only
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/micro_error_reporter.h"
#endif
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "models_data.h"

struct Model { const char* name; const unsigned char* data; size_t len; };
static const Model kModels[] = {{"m5", model_m5, model_m5_len}, {"m30", model_m30, model_m30_len}, {"m100", model_m100, model_m100_len}, {"a8", model_a8, model_a8_len}, {"a16", model_a16, model_a16_len}};
static String results[20];
static int nResults = 0;

// weights: 0 = read from flash (as the model array is), 1 = copied to internal RAM, 2 = copied to PSRAM. Returns the arena bytes the model needs (0 on failure).
// Nothing is freed: this TFLM version crashed in the destructor after a failed AllocateTensors, and it is a one-shot bench.
static size_t bench(const Model& m, bool arenaPsram, int weights, size_t arenaBytes, int nTimed) {
    const uint32_t caps = (arenaPsram ? MALLOC_CAP_SPIRAM : MALLOC_CAP_INTERNAL) | MALLOC_CAP_8BIT;
    char line[360]; const char* wname = weights == 0 ? "flash" : weights == 1 ? "int_ram" : "psram";
    const size_t intBefore = heap_caps_get_free_size(MALLOC_CAP_INTERNAL), psBefore = heap_caps_get_free_size(MALLOC_CAP_SPIRAM);
    const unsigned char* data = m.data;
    if (weights) {
        void* w = heap_caps_aligned_alloc(16, m.len, (weights == 1 ? MALLOC_CAP_INTERNAL : MALLOC_CAP_SPIRAM) | MALLOC_CAP_8BIT);
        if (!w) { snprintf(line, sizeof(line), "BENCH %s weights=%s WEIGHT_ALLOC_FAILED", m.name, wname); results[nResults++] = line; Serial.println(line); return 0; }
        memcpy(w, m.data, m.len); data = static_cast<const unsigned char*>(w);
    }
    uint8_t* arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, arenaBytes, caps));
    if (!arena) { snprintf(line, sizeof(line), "BENCH %s arena=%s weights=%s ARENA_ALLOC_FAILED", m.name, arenaPsram ? "psram" : "internal", wname); results[nResults++] = line; Serial.println(line); return 0; }
    tflite::MicroMutableOpResolver<5> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddAdd(); resolver.AddMean(); resolver.AddReshape();
#ifdef USE_ESP_TF
    tflite::MicroInterpreter* interp = new tflite::MicroInterpreter(tflite::GetModel(data), resolver, arena, arenaBytes);
#else
    static tflite::MicroErrorReporter reporter;
    tflite::MicroInterpreter* interp = new tflite::MicroInterpreter(tflite::GetModel(data), resolver, arena, arenaBytes, &reporter);
#endif
    if (interp->AllocateTensors() != kTfLiteOk) {
        snprintf(line, sizeof(line), "BENCH %s arena=%s weights=%s ALLOCATE_TENSORS_FAILED (arena %u bytes too small or op missing)", m.name, arenaPsram ? "psram" : "internal", wname, (unsigned)arenaBytes);
        results[nResults++] = line; Serial.println(line); return 0;
    }
    TfLiteTensor* in = interp->input(0);
    for (int i = 0; i < in->bytes; ++i) in->data.int8[i] = static_cast<int8_t>((i * 37) % 255 - 127);
    uint32_t mn = 0xFFFFFFFF, mx = 0; uint64_t sum = 0; bool ok = true;
    for (int i = 0; i < nTimed; ++i) {   // no warm-up: the first invoke is as slow as the rest (no cache effects worth a separate run at these times)
        const uint32_t t0 = micros();
        if (interp->Invoke() != kTfLiteOk) { ok = false; break; }
        const uint32_t dt = micros() - t0; sum += dt; if (dt < mn) mn = dt; if (dt > mx) mx = dt;
    }
    snprintf(line, sizeof(line), "BENCH %s arena=%s weights=%s %s n=%d min=%.2fms mean=%.2fms max=%.2fms arena_used=%uB model=%uB int_free_before=%u int_free_after=%u psram_free_before=%u psram_free_after=%u",
             m.name, arenaPsram ? "psram" : "internal", wname, ok ? "OK" : "INVOKE_FAILED", nTimed, mn / 1000.0, sum / (double)nTimed / 1000.0, mx / 1000.0, (unsigned)interp->arena_used_bytes(), (unsigned)m.len,
             (unsigned)intBefore, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)psBefore, (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    results[nResults++] = line; Serial.println(line);
    return ok ? interp->arena_used_bytes() : 0;
}

void setup() {
    Serial.begin(115200);
    delay(3000);
    Serial.println("SETUP_START");
#if defined(ESP_NN) && defined(ARCH_ESP32_S3) && defined(CONFIG_NN_OPTIMIZED)
    const char* nn = "ESP_NN+S3 assembly flags defined";
#elif defined(ESP_NN)
    const char* nn = "ESP_NN (portable C) defined";
#else
    const char* nn = "ESP_NN not defined";
#endif
    results[nResults++] = String("INFO cpu_mhz=") + getCpuFrequencyMhz() + " " + nn + " psram=" + (psramFound() ? "yes" : "no") + " psram_size=" + ESP.getPsramSize() + " flash_size=" + ESP.getFlashChipSize() + " chip=" + ESP.getChipModel() + " rev=" + ESP.getChipRevision() + " tflite_schema=" + TFLITE_SCHEMA_VERSION;
    const size_t kPs = 1536 * 1024;
#ifdef ONLY_M100   // a separate boot for the big model: earlier runs leave too little PSRAM
    bench(kModels[2], true, 0, kPs, 1);
    return;
#endif
    bench(kModels[0], false, 1, 200 * 1024, 3);    // m5 first, while internal RAM is empty: arena and weights both in internal RAM
    bench(kModels[0], true, 0, kPs, 3);            // arena in PSRAM, weights in flash from here on
    bench(kModels[3], true, 0, kPs, 3);            // a8: channels all multiples of 8
    bench(kModels[1], true, 0, kPs, 1);            // m30: odd channel counts
    bench(kModels[4], true, 0, kPs, 2);            // a16: channels multiples of 16
    bench(kModels[2], true, 0, kPs, 1);            // m100: odd channel counts
}

void loop() {
    for (int i = 0; i < nResults; ++i) Serial.println(results[i]);
    Serial.println("BENCH_DONE");
    delay(8000);
}
