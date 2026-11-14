"""Explicit allowlist keeps credentials and caches out of the cluster package."""
import base64
import tarfile
from pathlib import Path
root=Path(__file__).resolve().parents[1]
destination=root/'tmp/project.tar.gz'
with tarfile.open(destination,'w:gz') as archive:
    for directory in ['kernel_study','configs','knowledge','tests','scripts','cluster','docs']:
        for path in sorted((root/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                archive.add(path,arcname=path.relative_to(root))
    for name in ['pyproject.toml','.gitignore']:
        archive.add(root/name,arcname=name)
encoded=base64.encodebytes(destination.read_bytes()).decode()
(root/'tmp/project.b64').write_text(encoded)
print(f'Packaged {destination.stat().st_size} bytes; {len(encoded.splitlines())} transfer lines')
