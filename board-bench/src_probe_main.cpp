// For each operation output of the per-channel mn10 model, runs the whole model on the fixed input and prints the sum and first 8 values of that tensor, so they can be compared with the computer's (probe_expected.json).
#include <Arduino.h>
#include <esp_heap_caps.h>
#include <ESP_TF.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "models_mnreal.h"
#include "probe_data.h"
#ifdef PROBE_PT
#define MODEL_DATA model_mn10real_pt
#define MODEL_LEN model_mn10real_pt_len
#else
#define MODEL_DATA model_mn10real_pc
#define MODEL_LEN model_mn10real_pc_len
#endif

void setup() {
    Serial.begin(115200); delay(3000); Serial.println("SETUP_START");
    uint8_t* model = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, MODEL_LEN, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    const size_t kArena = 700 * 1024; uint8_t* arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(16, kArena, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT));
    if (!model || !arena) { Serial.println("ALLOC_FAILED"); return; }
    memcpy(model, MODEL_DATA, MODEL_LEN);
    static tflite::MicroMutableOpResolver<12> resolver;
    resolver.AddConv2D(); resolver.AddDepthwiseConv2D(); resolver.AddAdd(); resolver.AddMul(); resolver.AddMean(); resolver.AddReshape(); resolver.AddHardSwish();
    resolver.AddFullyConnected(); resolver.AddLogistic(); resolver.AddPad(); resolver.AddReduceMax();
    for (int p = 0; p < kNProbes; ++p) {
        const uint32_t t = kProbes[p]; memcpy(model + kOutPos, &t, 4);
        tflite::MicroInterpreter* interp = new tflite::MicroInterpreter(tflite::GetModel(model), resolver, arena, kArena);
        if (interp->AllocateTensors() != kTfLiteOk) { Serial.printf("PROBE %u ALLOCATE_FAILED\n", (unsigned)t); continue; }
        TfLiteTensor* in = interp->input(0);
        for (size_t i = 0; i < in->bytes; ++i) in->data.int8[i] = static_cast<int8_t>((i * 37) % 255 - 127);
        if (interp->Invoke() != kTfLiteOk) { Serial.printf("PROBE %u INVOKE_FAILED\n", (unsigned)t); continue; }
        TfLiteTensor* out = interp->output(0); long long sum = 0; for (size_t i = 0; i < out->bytes; ++i) sum += out->data.int8[i];
        Serial.printf("PROBE %u n=%u sum=%lld first8=", (unsigned)t, (unsigned)out->bytes, sum);
        for (int i = 0; i < 8 && i < (int)out->bytes; ++i) Serial.printf("%d%c", out->data.int8[i], i < 7 ? ',' : ' ');
        Serial.println();
    }
    Serial.println("PROBE_DONE");
}
void loop() { delay(1000); }
