// The whole classifier path on the node board (front end + int8 model), on the same 8 windows as the front end test; prints the class probabilities for comparison with the computer.
#include <Arduino.h>
#include "../node-firmware-link/MelFrontend.cpp"
#include "../node-firmware-link/AcousticClassifier.cpp"
#include "frontend_vectors.h"

extern const uint8_t kModelStart[] asm("_binary_model_acoustic_mn10_int8_tflite_start");
extern const uint8_t kModelEnd[] asm("_binary_model_acoustic_mn10_int8_tflite_end");

void setup() {
    Serial.begin(115200); delay(3000); Serial.println("SETUP_START");
    terrawatch::acoustic::AcousticClassifier c;
    if (!c.begin(kModelStart, kModelEnd - kModelStart, nullptr)) { Serial.println("CLASSIFIER_BEGIN_FAILED"); return; }
    for (int v = 0; v < kNVectors; ++v) {
        terrawatch::acoustic::ClassifierResult r{};
        if (!c.classify(kVecPcm[v], r)) { Serial.printf("CLS %d FAILED\n", v); continue; }
        Serial.printf("CLS %d probs=%.4f,%.4f,%.4f,%.4f,%.4f,%.4f frontend=%.0fms model=%.0fms\n", v, r.probs[0], r.probs[1], r.probs[2], r.probs[3], r.probs[4], r.probs[5], r.frontendMs, r.modelMs);
    }
    Serial.println("CLS_DONE");
}
void loop() { delay(1000); }
