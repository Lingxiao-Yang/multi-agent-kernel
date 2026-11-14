"""Four actual-model smoke searches, serialized after GPU infrastructure controls."""
import subprocess
import sys
import time
from pathlib import Path
while len(list(Path('results/gpu_control_matrix').glob('*/summary.json'))) < 4:
    time.sleep(5)
subprocess.run([sys.executable,'scripts/run_campaign.py','--output','results/llm_local_smoke',
               '--attempts','1','--cases','softmax_irregular','--seeds','11','--dollar-cap','0.50',
               '--validated-pilot','results/local_pilot_v3'],check=True)
