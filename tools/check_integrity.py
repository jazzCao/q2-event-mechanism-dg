"""Verify every distributed file against its release manifest. No dependencies."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]


def main():
    manifest=json.loads((ROOT/'provenance/FILES.json').read_text('utf8'))
    for name,binding in manifest['files'].items():
        path=(ROOT/name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():raise ValueError('missing/unsafe file: '+name)
        raw=path.read_bytes()
        if len(raw)!=binding['bytes'] or hashlib.sha256(raw).hexdigest()!=binding['sha256']:
            raise ValueError('file changed: '+name)
    print(json.dumps({'status':'PASS','files':len(manifest['files'])}))


if __name__=='__main__':main()
