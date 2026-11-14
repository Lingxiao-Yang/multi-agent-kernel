"""Windows launcher: transient secret, forwarded via environment to local WSL.

The key is not persisted or passed as a command-line argument.
"""
import getpass
import os
import subprocess
from pathlib import Path

key=getpass.getpass('API key for bounded pilot (not saved): ')
env=dict(os.environ)
env['OPENAI_API_KEY']=key
env['WSLENV']=':'.join(filter(None,[env.get('WSLENV',''),'OPENAI_API_KEY/u']))
with Path('logs/llm_local_pilot.log').open('w',encoding='utf-8') as log:
    result=subprocess.run(['wsl','-d','Ubuntu-24.04-D','--exec',
                          str(Path('.venv-linux/bin/python').resolve()).replace('E:', '/mnt/e').replace('\\','/'),
                          'scripts/llm_pilot.py'],env=env,stdout=log,stderr=subprocess.STDOUT)
print('Pilot process exit:',result.returncode)
