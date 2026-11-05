#!/bin/bash
# Runs ON a GPU VM: get the cache from the bucket, smoke-test, train and evaluate each model:seed given as arguments, copy results to the bucket, then power the VM off.
# Usage: bash vm_job.sh student_ir:0 student_ir:1 ...      (log: ~/acoustic/job.log)
exec > ~/acoustic/job.log 2>&1
cd ~/acoustic; B=gs://kireeti-wildfire-507012/acoustic_v1; NAME=$(hostname)
trap 'gcloud storage cp ~/acoustic/job.log "$B/runs/$NAME/job_$(date +%s).log" >/dev/null 2>&1' EXIT   # failures leave a log in the bucket before the VM is gone
echo "start $(date -u +%FT%TZ) on $NAME"; nvidia-smi --query-gpu=name --format=csv,noheader
pip install --user -q librosa soundfile scikit-learn pandas scipy 2>&1 | tail -1
mkdir -p data/cache8k runs scripts
# the laptop uploads corpus_v1 slowly; the DONE marker is written when the upload ends and the VM is launched only after that, so this waits at most 5 min
for i in $(seq 5); do gcloud storage ls "$B/corpus_v1/DONE" >/dev/null 2>&1 && break; sleep 60; done
gcloud storage ls "$B/corpus_v1/DONE" >/dev/null 2>&1 || { echo "CORPUS MISSING"; sudo shutdown -h +1; exit 1; }
if [ ! -f data/cache8k/audio.npy ]; then
  mkdir -p data/corpus_v1 && gcloud storage rsync -r "$B/corpus_v1" data/corpus_v1 && python3 scripts/21_cache_8k.py || { echo "CACHE BUILD FAILED"; sudo shutdown -h +1; exit 1; }
fi
python3 - <<'P' || { echo "SMOKE FAILED"; exit 1; }
import torch; print("torch", torch.__version__, "cuda", torch.cuda.is_available())
P
echo ok | gcloud storage cp - "$B/runs/$NAME/write_test" >/dev/null 2>&1 || echo "bucket not writable from this VM: results stay in ~/acoustic/runs and are fetched with scp"   # the VM account may only read the bucket
for spec in "$@"; do
  IFS=: read -r m s tag xargs <<< "$spec"; run=runs/${m}_s${s}${tag:+_$tag}   # spec = model:seed[:tag:extra+train+args]
  if [ "$m" = upper ]; then      # spec upper:<models comma separated>[::extra+args]: pretrained upper bound, see src/upper_bound.py
    pip install --user -q panns-inference hear21passt 2>&1 | tail -1
    pip install --user -q --no-deps torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -1      # the first install pulls a torchaudio that does not match torch 2.9.1
    python3 src/upper_bound.py --model $s ${xargs//+/ } > runs/upper.log 2>&1 || { echo "UPPER FAILED"; tail -8 runs/upper.log; continue; }
    gcloud storage cp -r runs/upper_* runs/upper.log "$B/runs/$NAME/" 2>&1 | tail -1; echo "done $spec $(date -u +%FT%TZ)"; continue
  fi
  if [ "$m" = prep ]; then      # spec prep:soundscapes: add the forest soundscapes, rebuild the cache, re-score the existing student on the new test sets
    [ -d runs/student_ir_s0_easy ] || gcloud storage cp -r "$B/runs/acoustic-vk2-20261103/student_ir_s0_easy" runs/ 2>&1 | tail -1      # the model trained without the soundscapes, to re-score on the new test sets
    cp runs/student_ir_s0_easy/eval.json runs/student_ir_s0_easy/eval_before.json 2>/dev/null
    python3 scripts/40_soundscapes.py > runs/prep.log 2>&1 && rm -f data/cache8k/eval_all.pkl && python3 scripts/21_cache_8k.py >> runs/prep.log 2>&1 && python3 src/evaluate.py runs/student_ir_s0_easy >> runs/prep.log 2>&1 || { echo "PREP FAILED"; tail -8 runs/prep.log; continue; }
    gcloud storage cp -r runs/prep.log runs/student_ir_s0_easy "$B/runs/$NAME/prep_old_model/" 2>&1 | tail -1; gcloud storage cp data/corpus_v1/corpus.csv manifests/soundscape_listening.csv "$B/runs/$NAME/prep_old_model/" 2>&1 | tail -1; echo "done $spec $(date -u +%FT%TZ)"; continue
  fi
  if [ "$m" = frontend ]; then      # spec frontend:x: check src/frontend_ref.py against the real front end and save test vectors
    python3 src/frontend_check.py > runs/frontend.log 2>&1 || { echo "FRONTEND FAILED"; tail -8 runs/frontend.log; }
    gcloud storage cp -r runs/frontend runs/frontend.log "$B/runs/$NAME/" 2>&1 | tail -1; echo "done $spec $(date -u +%FT%TZ)"; continue
  fi
  if [ "$m" = export ]; then      # spec export:<model>:<tag>:<run dir>+<extra args>: int8 conversion test, see src/export_onnx.py and src/tflite_eval.py
    pip install --user -q onnx onnxscript 2>&1 | tail -1
    python3 src/export_onnx.py --model $s ${xargs//+/ } > runs/export.log 2>&1 || { echo "EXPORT FAILED"; tail -8 runs/export.log; continue; }
    [ -x ~/venv_tf/bin/pip ] || { rm -rf ~/venv_tf; pip install --user -q virtualenv 2>&1 | tail -1; python3 -m virtualenv -q ~/venv_tf && ~/venv_tf/bin/pip install -q tensorflow tf_keras onnx onnx2tf onnx_graphsurgeon onnxsim onnxruntime sng4onnx psutil ai-edge-litert 2>&1 | tail -2; }   # virtualenv does not need the OS python3-venv package
    ~/venv_tf/bin/pip install -q pandas 2>&1 | tail -1      # the scoring script reads the patch labels with pandas
    RUN=$(echo "$xargs" | sed 's/.*--run+\([^+]*\).*/\1/')
    for QT in per-tensor per-channel; do echo "== $QT" >> runs/export.log; PATH=~/venv_tf/bin:$PATH ~/venv_tf/bin/python src/tflite_eval.py $RUN/export $QT >> runs/export.log 2>&1 || echo "TFLITE FAILED $QT"; done; true 2>&1 || { echo "TFLITE FAILED"; tail -12 runs/export.log; }
    TAG=$(basename $RUN); gcloud storage cp -r $RUN/export "$B/runs/$NAME/export_$TAG/" 2>&1 | tail -1; gcloud storage cp runs/export.log "$B/runs/$NAME/export_$TAG.log" 2>&1 | tail -1; echo "done $spec $(date -u +%FT%TZ)"; continue
  fi
  if [ "$m" = slide ]; then      # spec slide:<model>: continuous-audio windows of the fine-tuned model, see src/sliding.py
    python3 src/sliding.py --model $s > runs/slide.log 2>&1 || { echo "SLIDE FAILED"; tail -8 runs/slide.log; continue; }
    gcloud storage cp runs/ft_$s/sliding.npy runs/ft_$s/sliding.csv runs/slide.log "$B/runs/$NAME/ft_$s/" 2>&1 | tail -1; echo "done $spec $(date -u +%FT%TZ)"; continue
  fi
  if [ "$m" = tag ]; then      # spec tag: AudioSet-tag the Belize clips listed in belize_audit_ids.csv (uploaded to the bucket), see src/tag_clips.py
    pip install --user -q panns-inference 2>&1 | tail -1; gcloud storage cp "$B/belize_audit_ids.csv" runs/ 2>&1 | tail -1
    python3 src/tag_clips.py runs/belize_audit_ids.csv > runs/tag.log 2>&1 || { echo "TAG FAILED"; tail -8 runs/tag.log; continue; }
    gcloud storage cp runs/belize_audit.csv runs/tag.log "$B/runs/$NAME/" 2>&1 | tail -1; echo "done $spec $(date -u +%FT%TZ)"; continue
  fi
  if [ "$m" = trig ]; then      # spec trig:<model>: onset-triggered blast test, see src/triggered.py
    python3 src/triggered.py --model $s > runs/trig.log 2>&1 || { echo "TRIG FAILED"; tail -8 runs/trig.log; continue; }
    gcloud storage cp runs/ft_$s/triggered.npy runs/ft_$s/triggered.csv runs/trig.log "$B/runs/$NAME/ft_$s/" 2>&1 | tail -1; echo "done $spec $(date -u +%FT%TZ)"; continue
  fi
  if [ "$m" = ft ]; then      # spec ft:<EfficientAT model name>[::extra+args]: fine-tune a pretrained MobileNetV3, see src/finetune.py
    [ -d ~/EfficientAT ] || git clone -q --depth 1 https://github.com/fschmid56/EfficientAT.git ~/EfficientAT
    pip install --user -q panns-inference hear21passt 2>&1 | tail -1; pip install --user -q --no-deps torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129 2>&1 | tail -1
    python3 src/finetune.py --model $s ${xargs//+/ } > runs/ft_${s}${tag:+_$tag}.log 2>&1 || { echo "FT FAILED $spec"; tail -8 runs/ft_${s}${tag:+_$tag}.log; continue; }
    gcloud storage cp -r runs/ft_${s}* "$B/runs/$NAME/" 2>&1 | tail -1; echo "done $spec $(date -u +%FT%TZ)"; continue
  fi
  python3 src/train.py --model $m --seed $s --epochs 30 ${xargs//+/ } --out $run > $run.log 2>&1 || { echo "TRAIN FAILED $spec"; tail -5 $run.log; continue; }
  python3 src/evaluate.py $run >> runs/eval.log 2>&1 || echo "EVAL FAILED $spec"
  gcloud storage cp -r $run $run.log "$B/runs/$NAME/" 2>&1 | tail -1
  echo "done $spec $(date -u +%FT%TZ)"
done
gcloud storage cp runs/eval.log job.log "$B/runs/$NAME/" 2>&1 | tail -1
echo "finished $(date -u +%FT%TZ); powering off"; sudo shutdown -h +1
