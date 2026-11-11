# Acoustic classifier for the forest stations: design (draft, 2026-11-02)

Facts are marked [V] verified from a page or file, [C] claim from a source I could not check, [E] my estimate, [?] open.

## 1. Purpose
A sound module on a station gives the Control Room a "check" item, never a confirmed event. About 95% of Indian forest fires are human-caused [C: secondary reports of a Forest Survey of India study; I did not reach the report], so sound is useful as evidence of human activity or an open flame near fire-prone forest. A human verifies every item.

## 2. Classes
| Class | Alerts? | Reason |
|---|---|---|
| chainsaw | yes (needs repeat detections) | logging, human presence |
| blast (gunshot + fireworks, merged) | yes (single event) | poaching, ignition source; a 1 s window may not separate the two [E] |
| fire (any open flame of any size: campfire, bonfire, burning residue, wildfire) | log only until validated | every wildfire starts small; legal campfires will also trigger it, so the operator checks whether it is attended |
| wood noise (tree fall, branch crack, axe, handsaw) | no | the usual false-alarm source for chainsaw and gunshot |
| wind | no | also reports "audio is blind now" |
| background (birds, insects, rain, thunder, frogs) | no | |
Left out: vehicle engine (confusable, thin data), voices and footsteps (privacy). All audio stays on the station; only class, confidence and counts are sent.

## 3. Evidence behind the choices
- Detection distance depends mostly on ambient noise and varies up to 5x over day and season; attenuation is linear in frequency and distance [V: Haupert et al., arXiv 2211.16077].
- A forest gunshot CNN: recall 0.95 / precision 0.85 at trained sites; precision about 0.57 (recall about 0.80) at an unseen site [V: Katsis et al. 2022, via eprints.soton.ac.uk/468267]. Evaluate on held-out sites; calibrate each site at installation.
- AudioMoth gunshot units logged 129 h of non-gunshot triggers (dogs, leaves in wind, birds) [V: Mongabay coverage].
- Wind makes low-frequency noise peaks that classifiers mistake for events; windscreens help but do not remove it [V: unito 2024 study, BYU]. Ridge placement for LoRa is the worst place for a microphone: put it on a short cable below the ridge top with a foam and fur windscreen [E].
- Fire crackle has energy at 6 to 15 kHz in lab burns [V: Yedinak], which attenuates fastest in forest. The only range figure is a company claim: about 45 m for a one-square-yard fire [C: FireWave]. Fire is a near-field class.
- Chainsaw range claims: 1 km and 3 km [C: Rainforest Connection]; 400 m or less is a wildlife-impact estimate, not a detection range [C]. Our own field test must give the number.

## 4. Data
| Source | Use | Licence | Notes |
|---|---|---|---|
| Existing Secure Forest Patrol audio (C3GD, FSC22, Rodopi, ESC-50, FrugalAI, Sensing the Forest) | chainsaw, gunshot, wood, wind, fire (FSC22 Fire 75, ESC-50 crackling fire 40), background | mixed; ESC-50 and FrugalAI BY-NC; FSC22 per-clip, unverified | on disk, read in place |
| Belize forest gunshot set (Mendeley x48cwz364j) | gunshot, forest background | CC BY 4.0 [V] | 749 gunshots, over 35,000 background clips, 20 sites, 8 kHz; file download not yet verified |
| FSD50K, 18 listened-and-confirmed open-fire clips (manifests/fsd50k_fire_listening.csv) | fire class | 9 CC0, 8 CC BY, 1 Sampling+ (leave out of the product set) | about 5 min in total; two pairs look like duplicates (Bonfire Flames / Bonfire 2, FireBurning / FireBurning_v2): split as one recording |
| FSD50K metadata survey (this folder) | fire hard cases, wood, wind, rain | CC0 / BY / BY-NC per clip [V] | Fire label: 509 clips; outdoor keywords 20, unknown 115, household 374. Crackle label is mostly vinyl and plastic: not used. Audio is about 25 GB in split zips; disk has 25 GB free |
| Real wildfire recordings | held-out fire test only | mostly BY-NC [V] | Tim Kahn, Oregon, 13 min (freesound 253770, BY-NC); Cerrado, Brazil, 1.25 min (752028, BY-NC, possibly distant); Yellowstone about 15 ft (CC0 on Freesound 519520, BY-NC on Acoustic Atlas 2367 and 2369, probably the same recording: treat as BY-NC) |
| Not usable | Zenodo 3746895 (audio inside a fire, camera-mounted); Zenodo 22634889 Forest Fire Audio (restricted, request only); Kaggle "Forest Wild Fire Sound Dataset" (page unreadable) | | |
No open-licence real wildfire audio exists in what I found. A product variant would have no real fire data; our own burn recording is the fix.

## 5. Evaluation
Main metric: false alarms per hour of forest background on held-out sites (Belize sites plus Sensing the Forest, 20,000+ background windows already local). Also per-class precision and recall with intervals, a campfire-to-wildfire transfer test, and a noise-robustness test (the old model fell to about 2% accuracy at 5 dB). Splits by recording and by site. Two model variants: all data (research) and open-licence only (product).

## 6. Models to compare on held-out sites
A) small CNN from scratch (old model: 1,315 parameters, 0.38 M multiply-adds). B) reduced-width MobileNet fine-tuned from a pretrained model (width 0.25 at 40x101 input: 218,400 parameters, about 213 KB int8, 2.65 M multiply-adds [V: computed here]). C) pretrained embeddings plus a head, as a ceiling only. Inference time on the S3 is an estimate, not measured.

## 7. Gates
1. Offline model meets an agreed false-alarm rate on held-out sites.
2. Bench test: ESP32-S3 with real INMP441 audio matches the Python model (old project never ran on a real microphone).
3. Field test: chainsaw at 50, 100, 200 and 400 m; prescribed-burn recording together with the PMS5003 and BME680.
4. Pilot with per-site calibration. Nothing alerts before gate 4.

## 8. Open items
- Listen to the 20 outdoor and 115 unknown FSD50K fire clips; fetch only chosen clips (range requests into the split zips) [?].
- Belize file download; Freesound licences are verified only for the 3 recordings above.
- Audio privacy law in India (not researched); blocks pilot clip logging to SD card.
- Power draw of INMP441, MAX4466 and continuous ADC sampling (not measured).
- Whether the ESP32-S3 low-power coprocessor can sample while the CPU sleeps (not researched).
- A prescribed burn with a forest department (not researched whether available).
- Pin plan: audio pins 1, 6, 7, 8 do not clash with LoRa (9 to 14); the node config puts I2C clock on GPIO 20, the USB D+ pin, which the BME680 wiring must move off.
