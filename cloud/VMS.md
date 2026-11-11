# GPU VMs for the station sound model (2026-11-02)

Measured today by the provisioning job; the VMs are stopped, not deleted. Account emails are never written here: use the account shown for each project in the local project map (see GCP_PROJECT_MAP in scripts/30_launch.py) section 11 (`<account>`).

| Project | Instance | Zone | Machine | Provisioning | Boot disk | GPU | nvidia-smi | torch.cuda | packages | fp16 matmul 4096 | free disk |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ace-sequencer-478304-h4 | acoustic-ace-20261102 | asia-south1-b | g2-standard-8 | Spot | 150 GB pd-balanced | L4 23 GB, driver 580.178 | pass | pass (torch 2.9.1+cu129) | pass (torchaudio reinstalled to match torch) | 53.6 TFLOPS | 119 GB |
| causal-bongo-508008-e5 | acoustic-bongo-20261102 | asia-south1-b | g2-standard-8 | Spot | 150 GB pd-balanced | L4 23 GB, driver 580.178 | pass | pass | pass | 52.9 TFLOPS | 119 GB |
| jalari | not created | | | | | | | | | | The launch of the job for this project was refused by the tool's permission check (category "Modify Shared Resources"); no gcloud error. Read-only checks worked: no VMs, port 22 open. |

Also in causal-bongo: `bakeoff-gpu-20261102` (asia-south1-c, 120 GB disk, stopped, from the earlier image-detector work). Its start did not succeed, so a new VM was made; the old one was left alone.

Packages on both: soundfile, librosa, torchaudio, timm, scikit-learn, pandas, pyyaml, tqdm, tensorboard, hear21passt (installed with --no-deps). TensorFlow (for int8 conversion) was meant for the jalari VM and is not installed anywhere yet.

VM to VM: ssh key `~/.ssh/id_vm` on each VM, authorised on the other; tested both ways. 200 MB rsync ace to bongo ran at about 255 MB/s inside asia-south1. External IPs change at every start: read them again with the describe command.

## Commands (replace <account>, <project>, <zone>, <name>)
```
gcloud compute instances start <name> --zone=<zone> --account=<account> --project=<project>
gcloud compute instances stop  <name> --zone=<zone> --account=<account> --project=<project>
gcloud compute instances describe <name> --zone=<zone> --account=<account> --project=<project> --format="value(status,networkInterfaces[0].accessConfigs[0].natIP)"
ssh -o IdentitiesOnly=yes -i <bakeoff_key> bakeoff@<external-ip>
```
The key pair `bakeoff_key` was made in the session scratchpad and its public half is in each instance's metadata; if the scratchpad is gone, make a new key and replace it with `gcloud compute instances add-metadata <name> --metadata=ssh-keys="bakeoff:<public key>",block-project-ssh-keys=TRUE`.
Instances were created with `--max-run-duration=4h --instance-termination-action=STOP`; the 4 h clock restarts at each start. Spot VMs can be preempted: jobs must write results as they go.
