// Checks the node's audio front end (node-firmware/lib/Acoustic/src/MelFrontend.cpp) on the board against the numpy reference, on 8 real windows, and times it.
#include <Arduino.h>
#include "../node-firmware-link/MelFrontend.cpp"      // compiled here so this environment has its own object file
#include "frontend_vectors.h"

void setup() {
    Serial.begin(115200); delay(3000); Serial.println("SETUP_START");
    terrawatch::acoustic::MelFrontend fe;
    if (!fe.begin()) { Serial.println("FRONTEND_BEGIN_FAILED (out of memory)"); return; }
    float* out = static_cast<float*>(heap_caps_malloc(sizeof(float) * 106 * 100, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT)); if (!out) { Serial.println("ALLOC_FAILED"); return; }
    double worst = 0; 
    for (int v = 0; v < kNVectors; ++v) {
        const uint32_t t0 = micros(); fe.compute(kVecPcm[v], out); const uint32_t dt = micros() - t0;
        double mx = 0, sum = 0; for (int i = 0; i < 106 * 100; ++i) { const double d = fabs(out[i] - kVecPatch[v][i]); if (d > mx) mx = d; sum += d; }
        if (mx > worst) worst = mx;
        Serial.printf("FRONTEND window=%d max_abs=%.6f mean_abs=%.7f time=%.1fms\n", v, mx, sum / (106 * 100), dt / 1000.0);
    }
    Serial.printf("FRONTEND_DONE worst max_abs=%.6f free_internal=%u free_psram=%u\n", worst, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
}
void loop() { delay(1000); }
