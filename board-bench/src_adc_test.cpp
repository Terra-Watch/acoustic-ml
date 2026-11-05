// MAX4466 on GPIO2 through the ADC digital controller, nothing else running: does it deliver blocks?
#include <Arduino.h>
#include "../node-firmware-link/Max4466.cpp"

void setup() {
    Serial.begin(115200); delay(3000); Serial.println("SETUP_START");
    static terrawatch::sensors::Max4466 mic;
    if (!mic.begin(2, 8000, 128)) { Serial.println("ADC_BEGIN_FAILED"); return; }
    Serial.println("ADC_STARTED");
    for (int i = 0; i < 20; ++i) {
        terrawatch::sensors::BlockStats st;
        const bool ok = mic.readBlock(st, 200);
        if (ok) Serial.printf("ADC block %d rms=%.1f mean=%.0f min=%d max=%d\n", i, st.rms, st.mean, st.min, st.max); else Serial.printf("ADC block %d NO DATA\n", i);
    }
    Serial.println("ADC_DONE");
}
void loop() { delay(1000); }
