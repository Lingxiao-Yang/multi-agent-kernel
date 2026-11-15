"""Small source-generation pilot on the local GPU; never pooled with A100."""
import subprocess
import sys
subprocess.run([sys.executable,'scripts/run_campaign.py','--output','results/local_source_smoke_v1',
               '--search-space','source-v1','--attempts','1',
               '--cases','softmax_irregular,matmul_rectangular','--seeds','11',
               '--dollar-cap','0.50'],check=True)
