"""Step 2: convert the ONNX model to an int8 TFLite model with onnx2tf (per-tensor, calibrated on real validation patches), then score it and the float model on the same patches.
Run in a Python environment with tensorflow and onnx2tf: python src/tflite_eval.py <run>/export"""
import subprocess, sys
from pathlib import Path

import numpy as np
import pandas as pd

exp = Path(sys.argv[1]); qt = sys.argv[2] if len(sys.argv) > 2 else "per-tensor"; out = exp / ("tf" if qt == "per-tensor" else "tf_" + qt.replace("-", "_"))
subprocess.run(["onnx2tf", "-i", str(exp / "model.onnx"), "-o", str(out), "-oiqt", "-qt", qt, "-cind", "mel", str(exp / "calib_nhwc.npy"), "[[[[0.0]]]]", "[[[[1.0]]]]", "-b", "1"], check=True)
import tensorflow as tf
f = sorted(out.glob("*_integer_quant.tflite"))[0]; print("int8 model", f, f.stat().st_size, "bytes", flush=True)
it = tf.lite.Interpreter(model_path=str(f)); it.allocate_tensors(); i, o = it.get_input_details()[0], it.get_output_details()[0]
P = np.load(exp / "patches.npy"); R = np.load(exp / "float_logits.npy"); meta = pd.read_csv(exp / "meta.csv")
si, zi = i["quantization"]; so, zo = o["quantization"]; print("input", i["shape"], i["dtype"], i["quantization"], "output", o["shape"], o["dtype"], o["quantization"], flush=True)
Q = []
for k, p in enumerate(P):
    x = p[None, ..., None] if tuple(i["shape"])[-1] == 1 else p[None, None]
    if i["dtype"] == np.int8: x = np.clip(np.round(x / si + zi), -128, 127).astype(np.int8)
    it.set_tensor(i["index"], x.astype(i["dtype"])); it.invoke(); y = it.get_tensor(o["index"])[0].astype("float32")
    Q.append((y - zo) * so if o["dtype"] == np.int8 else y)
    if k % 500 == 0: print(k, flush=True)
Q = np.array(Q); np.save(exp / ("int8_logits.npy" if qt == "per-tensor" else "int8_logits_" + qt.replace("-", "_") + ".npy"), Q)
def macro_f1(y, p):
    f = []
    for c in range(6):
        tp = ((y == c) & (p == c)).sum(); fp = ((y != c) & (p == c)).sum(); fn = ((y == c) & (p != c)).sum()
        if (y == c).sum(): f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f))
y = meta.label.to_numpy()
for s in ("val", "test_in"):
    m = (meta.split == s).to_numpy(); print(s, "float macro-F1 %.3f | int8 macro-F1 %.3f | agreement %.3f" % (macro_f1(y[m], R[m].argmax(1)), macro_f1(y[m], Q[m].argmax(1)), (R[m].argmax(1) == Q[m].argmax(1)).mean()), flush=True)
