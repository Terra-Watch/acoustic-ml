"""Get one GPU VM in a project (start it if it exists, else create it, trying every zone and Spot then on-demand), retrying for hours because L4s are scarce, then hand over to 30_launch.py.
Usage: python3 -I scripts/31_fleet.py PROJECT_ID INSTANCE model:seed[:tag:extra] ...   (the VM powers itself off when its jobs end)"""
import re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
KEYS = "/private/tmp/claude-501/-Users-kireeti-Desktop-Projects-TerraWatch/66a7fa1a-35bf-4929-9bca-fd61c4d502a7/scratchpad/keys.txt"
project, name, *jobs = sys.argv[1:]
rows = [l for l in (HERE.parent / "wildfire-datasets-v3" / "DECISIONS.md").read_text().splitlines() if l.startswith(f"| {project}")]
acct = re.match(r"\|[^|]*\|\s*([^|\s]+)", rows[0]).group(1) if rows else next(a for a in subprocess.run(["gcloud", "auth", "list", "--format=value(account)"], capture_output=True, text=True).stdout.split() if a.startswith("vkireeti16@"))
g = lambda *a: subprocess.run(["gcloud", *a, f"--account={acct}", f"--project={project}"], capture_output=True, text=True)
ZONES = ["asia-south1-c", "asia-south1-a", "asia-south1-b", "asia-southeast1-b", "asia-southeast1-c", "asia-southeast1-a", "asia-east1-b", "asia-east1-a", "asia-east1-c", "asia-northeast1-a", "asia-northeast1-b", "asia-northeast1-c"]   # the home region first, then the nearest ones; the bucket is in asia-south1, other regions only cost a little egress
create = lambda z, spot: g("compute", "instances", "create", name, f"--zone={z}", "--machine-type=g2-standard-8", "--accelerator=type=nvidia-l4,count=1",
    *(["--provisioning-model=SPOT", "--instance-termination-action=STOP"] if spot else ["--maintenance-policy=TERMINATE"]), "--image-family=pytorch-2-9-cu129-ubuntu-2204-nvidia-580",
    "--image-project=deeplearning-platform-release", "--boot-disk-size=100GB", "--scopes=cloud-platform", f"--metadata-from-file=ssh-keys={KEYS}", "--metadata=install-nvidia-driver=True")
for i in range(240):                                                   # about 4 hours
    found = g("compute", "instances", "list", f"--filter=name={name}", "--format=value(zone.basename(),status)").stdout.split()
    zone = None
    if found:
        zone = found[0]
        if found[1] != "RUNNING": g("compute", "instances", "start", name, f"--zone={zone}", "--quiet")
    else:
        for z in ZONES:
            for spot in (True, False):
                if "RUNNING" in create(z, spot).stdout: zone = z; break
            if zone: break
    if zone and g("compute", "instances", "describe", name, f"--zone={zone}", "--format=value(status)").stdout.strip() == "RUNNING":
        r = subprocess.run([sys.executable, "-I", str(HERE / "scripts" / "30_launch.py"), project, name, zone, *jobs], capture_output=True, text=True); print(r.stdout.strip()[-300:] or r.stderr[-300:], flush=True)
        if " running " in r.stdout: sys.exit(0)          # a VM that showed RUNNING and was then taken back (no capacity) never got its jobs: keep trying
    print(f"{project}: no GPU yet (attempt {i + 1})", flush=True); time.sleep(60)
sys.exit(f"{project}: gave up after 4 hours")
