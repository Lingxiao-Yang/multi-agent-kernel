"""Four-arm, four-attempt guidance/repair plumbing check on the local GPU."""
import subprocess
import sys
subprocess.run([sys.executable,'scripts/run_campaign.py','--output','results/local_repair_smoke_v1',
    '--search-space','source-v1','--shared-api-guidance','--attempts','4',
    '--cases','softmax_irregular','--seeds','11','--dollar-cap','0.50'],check=True)
