"""Start a stopped GPU VM, copy the code, and start vm_job.sh on it (detached). Accounts come from the local file named by GCP_PROJECT_MAP (default ~/.config/terrawatch/gcp_projects.md, outside git; one line per project: "| PROJECT_ID | ACCOUNT |") and are never printed.
Usage: python3 -I scripts/30_launch.py PROJECT_ID INSTANCE ZONE model:seed [model:seed ...]"""
import os, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
KEY = "/private/tmp/claude-501/-Users-kireeti-Desktop-Projects-TerraWatch/acc03b01-fa54-4585-98f9-f679819ffe2d/scratchpad/bakeoff_key"
project, name, zone, *jobs = sys.argv[1:]
MAP = Path(os.environ.get("GCP_PROJECT_MAP", Path.home() / ".config/terrawatch/gcp_projects.md"))
rows = [l for l in (MAP.read_text().splitlines() if MAP.exists() else []) if l.startswith(f"| {project}")]
acct = re.match(r"\|[^|]*\|\s*([^|\s]+)", rows[0]).group(1) if rows else next(a for a in subprocess.run(["gcloud", "auth", "list", "--format=value(account)"], capture_output=True, text=True).stdout.split() if a.startswith("vkireeti16@"))   # the bucket owner's own project is not in DECISIONS.md
g = lambda *a, **k: subprocess.run(["gcloud", *a, f"--account={acct}", f"--project={project}"], capture_output=True, text=True, **k)
sh = lambda *a: subprocess.run(["ssh", "-o", "IdentitiesOnly=yes", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "ConnectTimeout=10", "-i", KEY, *a], capture_output=True, text=True)

status = g("compute", "instances", "describe", name, f"--zone={zone}", "--format=value(status)").stdout.strip()
for attempt in range(30):                                    # a stopped Spot VM can fail to start when the zone is out of capacity: retry for about 30 minutes
    if status == "RUNNING": break
    r = g("compute", "instances", "start", name, f"--zone={zone}", "--quiet")
    status = g("compute", "instances", "describe", name, f"--zone={zone}", "--format=value(status)").stdout.strip()
    if status != "RUNNING": print(f"{project}: start attempt {attempt + 1} failed ({r.stderr.strip().splitlines()[-1][:90] if r.stderr.strip() else status}); retrying", flush=True); time.sleep(60)
else: sys.exit(f"{project}: could not start {name}")
ip = g("compute", "instances", "describe", name, f"--zone={zone}", "--format=value(networkInterfaces[0].accessConfigs[0].natIP)").stdout.strip()
for _ in range(30):
    if sh(f"bakeoff@{ip}", "echo up").stdout.strip() == "up": break
    time.sleep(10)
else: sys.exit(f"{project}: ssh never came up")
sh(f"bakeoff@{ip}", "mkdir -p ~/acoustic/src ~/acoustic/runs ~/acoustic/scripts")
subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "IdentitiesOnly=yes", "-i", KEY, *map(str, (HERE / "src").glob("*.py")), f"bakeoff@{ip}:~/acoustic/src/"], check=True)
subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "IdentitiesOnly=yes", "-i", KEY, *map(str, (HERE / "scripts").glob("[24]*.py")), f"bakeoff@{ip}:~/acoustic/scripts/"], check=True)
subprocess.run(["scp", "-q", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "IdentitiesOnly=yes", "-i", KEY, str(HERE / "cloud" / "vm_job.sh"), f"bakeoff@{ip}:~/acoustic/vm_job.sh"], check=True)
sh(f"bakeoff@{ip}", f"nohup bash ~/acoustic/vm_job.sh {' '.join(jobs)} > /dev/null 2>&1 &")
print(f"{project}: {name} running {' '.join(jobs)} (zone {zone}); it powers itself off when finished", flush=True)
