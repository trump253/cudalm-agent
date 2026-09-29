#!/usr/bin/env python3
"""CUDALM — v0.9 Phase D: the client-DISCONNECT fault gate.

Starts the real cudalm-server, creates a session, starts a streaming
turn (a large max_new_tokens so the stream is long enough to be
certainly in flight), reads at least one token event, and CLOSES the
socket abruptly — then reconnects and proves:

    * the server is still alive (health OK);
    * stats.live_requests == 0          (no leaked live request);
    * the original session is STILL LIVE;
    * a next turn on that session succeeds (the continuation).

Pinned semantics: a client disconnect is a CANCEL, never a crash and
never a session destroy.

Exit codes: 0 = pass; 1 = failure; 77 = self-skip (no checkpoint).

Provenance: CUDALM-native (v0.9 Phase D).
"""

import json
import os
import socket
import signal
import subprocess
import sys
import time

MODEL = sys.argv[1] if len(sys.argv) > 1 else ""
TOKENIZER = sys.argv[2] if len(sys.argv) > 2 else ""
SERVER = sys.argv[3] if len(sys.argv) > 3 else ""
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 0

HOST = "127.0.0.1"


def skip(msg):
    print("SKIP: %s" % msg, file=sys.stderr)
    sys.exit(77)


def die(msg, extra=""):
    print("FAIL: %s %s" % (msg, extra), file=sys.stderr)
    sys.exit(1)


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((HOST, 0))
    p = s.getsockname()[1]
    s.close()
    return p


def raw_http(method, path, body=None, timeout=300):
    import http.client

    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    conn.request(method, path, body=body if body is not None else b"")
    r = conn.getresponse()
    data = r.read()
    conn.close()
    return r.status, dict(r.getheaders()), data


def stream_then_disconnect(session_id, max_new_tokens):
    """Open a raw socket, start the streaming turn, read at least one
    token event, then close abruptly. Returns the number of token
    events read before the close."""
    body = b"disconnect fault gate"
    request = (
        "POST /v1/sessions/%d/turn/stream?max_new_tokens=%d HTTP/1.1\r\n"
        "Host: %s\r\n"
        "Content-Type: text/plain\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n" % (session_id, max_new_tokens, HOST, len(body))
    ).encode("ascii") + body

    s = socket.create_connection((HOST, PORT), timeout=60)
    s.sendall(request)
    buf = b""
    token_lines = 0
    deadline = time.time() + 120
    saw_token = False
    while time.time() < deadline:
        if saw_token:
            # give the server a couple of events in flight, then GO:
            time.sleep(0.3)
            break
        try:
            chunk = s.recv(65536)
        except socket.timeout:
            die("the stream produced no token event in time")
        if not chunk:
            die("the server closed the stream before a token event")
        buf += chunk
        lines = buf.split(b"\n")
        for line in lines:
            if b'"type":"token"' in line:
                token_lines += 1
                saw_token = True
        if saw_token:
            break
    # the abrupt close (no graceful FIN-and-drain):
    try:
        s.shutdown(socket.SHUT_RDWR)
    except OSError:
        pass
    s.close()
    return token_lines


def main():
    for p, name in ((SERVER, "server binary"), (MODEL, "model"),
                    (TOKENIZER, "tokenizer")):
        if not p or not os.path.exists(p):
            skip("missing %s: %r" % (name, p))

    global PORT
    port = PORT or free_port()
    PORT = port
    proc = subprocess.Popen(
        [SERVER,
         "--model", MODEL,
         "--tokenizer", TOKENIZER,
         "--host", HOST,
         "--port", str(port),
         # the stream runs max_new_tokens=256 (2 tokens/page -> ~135
         # pages) — the pool must cover it:
         "--slots", "4",
         "--pages", "256"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE)

    try:
        # ---- wait for the server --------------------------------------
        deadline = time.time() + 300
        up = False
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stderr.read().decode("utf-8", "replace")
                die("the server exited during startup", out[-2000:])
            try:
                if raw_http("GET", "/healthz", timeout=5)[0] == 200:
                    up = True
                    break
            except OSError:
                pass
            time.sleep(0.5)
        if not up:
            die("the server did not come up in time")

        # ---- create the session ----------------------------------------
        st, _, body = raw_http("POST", "/v1/sessions")
        assert st == 201
        sid = json.loads(body)["session_id"]

        # ---- the abrupt disconnect ---------------------------------------
        n = stream_then_disconnect(sid, max_new_tokens=256)
        print("disconnected mid-stream after %d token event(s)" % n,
              file=sys.stderr)
        assert n >= 1

        # ---- the server must survive; wait for it to come back to the
        #      accept loop (it is single-threaded: it is still draining
        #      the cancelled request's cleanup) -----------------------------
        deadline = time.time() + 120
        ok = False
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stderr.read().decode("utf-8", "replace")
                die("the server CRASHED on the disconnect", out[-2000:])
            try:
                st, _, body = raw_http("GET", "/healthz", timeout=10)
                if st == 200:
                    ok = True
                    break
            except OSError:
                pass
            time.sleep(0.3)
        if not ok:
            die("the server did not return to service after the "
                "disconnect")

        # ---- no leaked live request ----------------------------------------
        st, _, body = raw_http("GET", "/v1/stats")
        stats = json.loads(body)
        assert stats["live_requests"] == 0, stats

        # ---- the session is STILL live ---------------------------------------
        st, _, body = raw_http("GET", "/v1/sessions/%d" % sid)
        assert st == 200, "the session was destroyed by the disconnect (%d)" % st

        # ---- the continuation: a next turn on the SAME session -----------------
        st, _, body = raw_http(
            "POST", "/v1/sessions/%d/turn?max_new_tokens=2" % sid,
            body="continuation".encode("utf-8"))
        assert st == 200, "continuation turn failed: %d %r" % (st, body[:300])
        t = json.loads(body)
        assert len(t["generated_token_ids"]) == 2, t

        print("DISCONNECT GATE PASS: session %d survived, "
              "live_requests=%d, continuation ok (ctx=%d)"
              % (sid, stats["live_requests"], t["context_length"]))
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)

    return 0


if __name__ == "__main__":
    sys.exit(main())
