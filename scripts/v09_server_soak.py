#!/usr/bin/env python3
"""CUDALM — v0.9 Phase D: the bounded serving HTTP SOAK.

Starts the real cudalm-server and runs a BOUNDED mix of operations
(default 50 turns; --iterations scales the groups) — create /
multi-turn / reset / destroy / health / stats, interleaved with the
fault shapes (an invalid session -> 404, a bad request -> 400, a
deadline_ms=0 edge -> 200 disabled) — checking after every group:

    * the server is alive;
    * stats.live_requests is back to 0 (no leaked request);
    * after the destroys, stats.live_sessions falls back.

The final state must be live_sessions == 0 and live_requests == 0.
A 24h-style soak is NOT a hard gate — this is the bounded, repeatable
one (manually scale up with --iterations).

Exit codes: 0 = pass; 1 = failure; 77 = self-skip (no checkpoint).

Provenance: CUDALM-native (v0.9 Phase D).
"""

import json
import os
import random
import signal
import socket
import subprocess
import sys
import time

MODEL = sys.argv[1] if len(sys.argv) > 1 else ""
TOKENIZER = sys.argv[2] if len(sys.argv) > 2 else ""
SERVER = sys.argv[3] if len(sys.argv) > 3 else ""
PORT = int(sys.argv[4]) if len(sys.argv) > 4 else 0

ITERS = 50
for a in sys.argv[4:]:
    if a.startswith("--iterations="):
        ITERS = int(a.split("=", 1)[1])

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
         # the soak keeps up to 3 live sessions concurrently — the
         # pool capacities must cover that (the default 2 slots would
         # reject the 3rd create):
         "--slots", "4",
         "--pages", "64"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE)

    rnd = random.Random(20260929)
    turns = 0
    failures = 0

    def check(cond, msg):
        nonlocal failures
        if not cond:
            failures += 1
            print("  [soak] FAIL: %s" % msg, file=sys.stderr)

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

        live_sessions = []

        for it in range(ITERS):
            op = rnd.randrange(10)
            if op <= 3 and len(live_sessions) < 3:
                # ---- create -------------------------------------------
                st, _, body = raw_http("POST", "/v1/sessions")
                check(st == 201, "create %d" % st)
                if st == 201:
                    live_sessions.append(json.loads(body)["session_id"])
            elif op <= 6 and live_sessions:
                # ---- a turn (sometimes on a random live session) -------
                sid = live_sessions[rnd.randrange(len(live_sessions))]
                q = "?max_new_tokens=%d" % rnd.randrange(1, 5)
                st, _, body = raw_http(
                    "POST", "/v1/sessions/%d/turn%s" % (sid, q),
                    body=("soak turn %d" % it).encode("utf-8"))
                check(st == 200, "turn %d %r" % (st, body[:200]))
                if st == 200:
                    turns += 1
            elif op == 7 and live_sessions:
                # ---- reset ---------------------------------------------
                sid = live_sessions[rnd.randrange(len(live_sessions))]
                st, _, body = raw_http("POST", "/v1/sessions/%d/reset" % sid)
                check(st == 200, "reset %d" % st)
            elif op == 8 and live_sessions:
                # ---- destroy --------------------------------------------
                sid = live_sessions.pop()
                st, _, body = raw_http("DELETE", "/v1/sessions/%d" % sid)
                check(st == 200, "destroy %d" % st)
            else:
                # ---- the fault shapes -------------------------------------
                kind = rnd.randrange(3)
                if kind == 0:
                    # an invalid session -> 404:
                    st, _, body = raw_http(
                        "POST", "/v1/sessions/99999/turn?max_new_tokens=1",
                        body=b"z")
                    check(st == 404, "invalid session should 404, got %d" % st)
                elif kind == 1:
                    # a bad request (unknown parameter) -> 400:
                    if live_sessions:
                        sid = live_sessions[0]
                        st, _, body = raw_http(
                            "POST",
                            "/v1/sessions/%d/turn?bogus=1" % sid,
                            body=b"z")
                        check(st == 400, "bad param should 400, got %d" % st)
                    else:
                        st, _, body = raw_http("POST", "/nope", body=b"z")
                        check(st == 404, "unknown route should 404, got %d" % st)
                else:
                    # deadline_ms=0 = disabled (a valid edge) -> 200:
                    if live_sessions:
                        sid = live_sessions[0]
                        st, _, body = raw_http(
                            "POST",
                            "/v1/sessions/%d/turn?max_new_tokens=1"
                            "&deadline_ms=0" % sid,
                            body=b"deadline zero")
                        check(st == 200, "deadline_ms=0 should 200, got %d"
                              % st)
                        if st == 200:
                            turns += 1

            # ---- the per-group invariants ----------------------------------
            st, _, body = raw_http("GET", "/healthz", timeout=15)
            check(st == 200, "server alive at group %d" % it)
            st, _, body = raw_http("GET", "/v1/stats")
            stats = json.loads(body)
            check(stats["live_requests"] == 0,
                  "live_requests leaked at group %d: %r" % (it, stats))
            check(stats["live_sessions"] == len(live_sessions),
                  "live_sessions mismatch at group %d: %r vs %r"
                  % (it, stats["live_sessions"], len(live_sessions)))

        # ---- teardown: destroy everything ---------------------------------
        for sid in list(live_sessions):
            st, _, body = raw_http("DELETE", "/v1/sessions/%d" % sid)
            check(st == 200, "teardown destroy %d" % st)
        live_sessions = []

        st, _, body = raw_http("GET", "/v1/stats")
        stats = json.loads(body)
        check(stats["live_sessions"] == 0, "final live_sessions %r" % stats)
        check(stats["live_requests"] == 0, "final live_requests %r" % stats)
        st, _, body = raw_http("GET", "/healthz")
        check(st == 200, "the server is alive at the end")

        print("SOAK PASS: iterations=%d turns=%d failures=%d final=%s"
              % (ITERS, turns, failures, json.dumps(stats, sort_keys=True)))
        if failures:
            die("%d soak check(s) failed" % failures)
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
