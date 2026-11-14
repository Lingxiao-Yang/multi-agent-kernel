import importlib.util
import os
import sys
print('python', sys.executable, sys.version)
for name in ['torch', 'triton', 'numpy', 'pytest']:
    print(name, bool(importlib.util.find_spec(name)))
print('api_key_present', bool(os.environ.get('OPENAI_API_KEY')))
