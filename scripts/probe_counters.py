"""Record whether Nsight hardware counters are actually usable."""
import json
import shutil
import subprocess
import sys
from pathlib import Path
p=Path(sys.argv[1]); p.parent.mkdir(parents=True,exist_ok=True)
ncu=shutil.which('ncu')
result={'ncu_path':ncu,'status':'not_installed','counter_access_verified':False}
if ncu:
    cmd=[ncu,'--profile-from-start','off','--launch-count','1','--csv',
         '--metrics','dram__bytes_read.sum,dram__bytes_write.sum,sm__warps_active.avg.pct_of_peak_sustained_active',
         sys.executable,str(Path(__file__).with_name('profile_target.py'))]
    try:
        run=subprocess.run(cmd,capture_output=True,text=True,timeout=120)
        result.update(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,
                      status='available' if run.returncode==0 and 'dram__bytes_read' in run.stdout else 'unavailable')
        result['counter_access_verified']=result['status']=='available'
    except subprocess.TimeoutExpired:
        result['status']='timeout'
p.write_text(json.dumps(result,indent=2))
print(result['status'])
