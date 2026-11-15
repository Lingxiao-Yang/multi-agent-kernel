"""Run a local shell script on Lonestar6, or download a tar.gz of remote paths.

Usage:
  python scripts/ls6_remote.py run   <script.sh>
  python scripts/ls6_remote.py fetch <out.tgz> <path relative to $SCRATCH> [...]

Credentials: TACC_USER, TACC_PASSWORD and TACC_TOKEN environment variables,
or interactive prompts. Nothing is stored. The script is sent over the
encrypted channel's stdin (never argv), and the ls6 host key must match
tmp/known_hosts. Set FORWARD_OPENAI_API_KEY to export a key into the remote
session without writing it to disk (prefer ~/.openai_env on the cluster).
"""
import getpass
import os
import shlex
import socket
import sys
from pathlib import Path

import paramiko

HOST = 'ls6.tacc.utexas.edu'
KNOWN_HOSTS = Path(__file__).resolve().parents[1] / 'tmp' / 'known_hosts'


def credential(name, prompt, secret=True):
    value = os.environ.get(name)
    if value:
        return value
    return getpass.getpass(prompt) if secret else input(prompt)


def connect():
    user = credential('TACC_USER', 'TACC user: ', secret=False)
    password = credential('TACC_PASSWORD', 'TACC password: ')
    transport = paramiko.Transport(socket.create_connection((HOST, 22), timeout=20))
    transport.get_security_options().digests = ('hmac-sha2-512', 'hmac-sha2-256')
    transport.start_client(timeout=20)
    key = transport.get_remote_server_key()
    expected = paramiko.HostKeys(str(KNOWN_HOSTS)).lookup(HOST) if KNOWN_HOSTS.exists() else None
    if not expected or expected.get(key.get_name()) != key:
        sys.exit(f'Host key for {HOST} does not match {KNOWN_HOSTS}; refusing to send credentials')
    token = credential('TACC_TOKEN', 'TACC token code: ')  # ask last so the code is fresh

    def handler(title, instructions, prompts):
        return [token if ('token' in p.lower() or 'code' in p.lower()) else password for p, _ in prompts]

    transport.auth_interactive(user, handler)
    return transport


def run_stdin(transport, text, sink):
    chan = transport.open_session(timeout=20)
    chan.exec_command('bash -l -s')
    chan.sendall(text.encode())
    chan.shutdown_write()
    while True:
        data = chan.recv(1 << 20)
        if not data:
            break
        sink(data)
    err = b''
    while chan.recv_stderr_ready():
        err += chan.recv_stderr(65536)
    return chan.recv_exit_status(), err.decode(errors='replace')


def main(argv):
    if len(argv) < 2 or argv[0] not in ('run', 'fetch'):
        sys.exit(__doc__)
    transport = connect()
    try:
        if argv[0] == 'run':
            script = Path(argv[1]).read_text(encoding='utf-8')
            key = os.environ.get('FORWARD_OPENAI_API_KEY')
            prefix = f'export OPENAI_API_KEY={shlex.quote(key)}\n' if key else ''
            status, err = run_stdin(transport, prefix + script, lambda b: sys.stdout.write(b.decode(errors='replace')))
        else:
            out, paths = Path(argv[1]), argv[2:]
            if not paths:
                sys.exit('fetch needs at least one remote path')
            command = 'cd "$SCRATCH" && tar czf - ' + ' '.join(shlex.quote(p) for p in paths) + '\n'
            with out.open('wb') as f:
                status, err = run_stdin(transport, command, f.write)
            print(f'wrote {out} ({out.stat().st_size} bytes)')
        if err.strip():
            print('stderr:', err[-2000:], file=sys.stderr)
        sys.exit(status)
    finally:
        transport.close()


if __name__ == '__main__':
    main(sys.argv[1:])
