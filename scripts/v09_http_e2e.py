#!/usr/bin/env python3
"""CUDALM — v0.9 Phase D: the REAL-CHECKPOINT HTTP E2E gate.

Starts the real cudalm-server (real Qwen3.5-0.8B-Base checkpoint) and
drives the full API surface over a real socket / real HTTP:

    GET /healthz
    POST /v1/sessions                    (create A)
    POST /v1/sessions/A/turn             (turn 1, synchronous)
    POST /v1/sessions/A/turn/stream      (turn 2, NDJSON streaming)
    GET /v1/stats
    POST /v1/sessions/A/reset
    POST /v1/sessions/A/turn             (turn after the reset)
    POST /v1/sessions                    (create B)
    POST /v1/sessions/B/turn
    DELETE /v1/sessions/B
    POST /v1/sessions/B/turn             (expect 404)
    DELETE /v1/sessions/A
    GET /v1/stats                        (final: live == 0)
    GET /healthz                         (the server is still alive)

Pinned checks: the 200/201 codes, JSON / NDJSON parse, the stream
token events before the terminal, streamed ids == terminal ids, the
exact context-length growth, the reset to zero, the destroy -> 404,
the final live_sessions / live_requests == 0, and the server alive at
the end.

Exit codes: 0 = pass; 1 = failure; 77 = self-skip (no checkpoint).

Provenance: CUDALM-native (v0.9 Phase D).
"""

import json
import os
import signal
import socket
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
    """One HTTP/1.1 request on a fresh connection (Connection: close).
    Returns (status, headers_dict, body_bytes)."""
    import http.client

    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout)
    payload = body if body is not None else b""
    conn.request(method, path, body=payload)
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
         "--max-sessions", "4",
         "--slots", "4",
         "--pages", "64"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE)

    try:
        # ---- wait for the server (the model load takes a moment) ----
        deadline = time.time() + 300
        up = False
        while time.time() < deadline:
            if proc.poll() is not None:
                out = proc.stderr.read().decode("utf-8", "replace")
                die("the server exited during startup", out[-2000:])
            try:
                st, _, body = raw_http("GET", "/healthz", timeout=5)
                if st == 200:
                    up = True
                    break
            except OSError:
                pass
            time.sleep(0.5)
        if not up:
            die("the server did not come up in time")

        # ---- 1. health ------------------------------------------------
        st, _, body = raw_http("GET", "/healthz")
        assert st == 200, "health status %d" % st
        assert json.loads(body) == {"ok": True}, "health body %r" % body

        # ---- 2. create A ----------------------------------------------
        st, _, body = raw_http("POST", "/v1/sessions")
        assert st == 201, "create status %d" % st
        A = json.loads(body)["session_id"]

        # The REAL tokenizer encodes the raw body (NO trim) into a
        # VARIABLE number of tokens — measure the input token count
        # empirically (a throwaway session: reset -> 0, one turn with
        # max_new_tokens=1 -> context = N_in + 1):
        measured = {}
        def input_tokens(text):
            if text in measured:
                return measured[text]
            st, _, b = raw_http("POST", "/v1/sessions")
            assert st == 201
            m = json.loads(b)["session_id"]
            st, _, b = raw_http(
                "POST", "/v1/sessions/%d/turn?max_new_tokens=1" % m,
                body=text.encode("utf-8"))
            assert st == 200, (st, b[:200])
            n = json.loads(b)["context_length"] - 1
            assert n >= 1, (text, n)
            st, _, b = raw_http("DELETE", "/v1/sessions/%d" % m)
            assert st == 200
            measured[text] = n
            return n

        # ---- 3. turn 1 (synchronous) -----------------------------------
        st, _, body = raw_http(
            "POST", "/v1/sessions/%d/turn?max_new_tokens=8" % A,
            body="Hello".encode("utf-8"))
        assert st == 200, "turn1 status %d body %r" % (st, body[:300])
        t1 = json.loads(body)
        assert len(t1["generated_token_ids"]) == 8, "turn1 ids %r" % t1
        assert t1["context_length"] == input_tokens("Hello") + 8, (
            "turn1 ctx %r" % t1)
        assert t1["finish_reason"] == "MaxNewTokens", t1
        assert isinstance(t1["generated_text"], str) and t1["generated_text"]

        # ---- 4. turn 2 (streaming NDJSON) -------------------------------
        st, hdr, body = raw_http(
            "POST", "/v1/sessions/%d/turn/stream?max_new_tokens=8" % A,
            body="Second".encode("utf-8"))
        assert st == 200, "stream status %d" % st
        assert hdr.get("Content-Type") == "application/x-ndjson", hdr
        lines = [l for l in body.decode("utf-8").split("\n") if l]
        assert len(lines) == 9, "stream lines %d: %r" % (len(lines), lines)
        toks = [json.loads(l) for l in lines[:8]]
        term = json.loads(lines[8])
        assert all(t["type"] == "token" for t in toks), toks
        streamed = [t["token_id"] for t in toks]
        assert term["type"] == "terminal", term
        assert streamed == term["generated_token_ids"], (
            "streamed != terminal ids: %r vs %r"
            % (streamed, term["generated_token_ids"]))
        assert term["context_length"] == t1["context_length"] + \
            input_tokens("Second") + 8, term

        # ---- 5. stats ----------------------------------------------------
        st, _, body = raw_http("GET", "/v1/stats")
        assert st == 200
        stats = json.loads(body)
        assert stats["live_sessions"] >= 1, stats
        assert stats["live_requests"] == 0, stats

        # ---- 6. reset A ---------------------------------------------------
        st, _, body = raw_http("POST", "/v1/sessions/%d/reset" % A)
        assert st == 200, "reset status %d" % st
        assert json.loads(body)["context_length"] == 0, body

        # ---- 7. the turn after the reset ----------------------------------
        st, _, body = raw_http(
            "POST", "/v1/sessions/%d/turn?max_new_tokens=4" % A,
            body="After reset".encode("utf-8"))
        assert st == 200
        t3 = json.loads(body)
        assert t3["context_length"] == input_tokens("After reset") + 4, t3

        # ---- 8. create B + a turn on B -------------------------------------
        st, _, body = raw_http("POST", "/v1/sessions")
        assert st == 201
        B = json.loads(body)["session_id"]
        st, _, body = raw_http(
            "POST", "/v1/sessions/%d/turn?max_new_tokens=2" % B,
            body="B".encode("utf-8"))
        assert st == 200

        # ---- 9. destroy B -> the session is invalid -------------------------
        st, _, body = raw_http("DELETE", "/v1/sessions/%d" % B)
        assert st == 200, "destroy B status %d" % st
        st, _, body = raw_http(
            "POST", "/v1/sessions/%d/turn?max_new_tokens=1" % B,
            body="z".encode("utf-8"))
        assert st == 404, "turn on destroyed B should be 404, got %d" % st

        # ---- 10. destroy A; the final stats ---------------------------------
        st, _, body = raw_http("DELETE", "/v1/sessions/%d" % A)
        assert st == 200
        st, _, body = raw_http("GET", "/v1/stats")
        stats = json.loads(body)
        assert stats["live_sessions"] == 0, stats
        assert stats["live_requests"] == 0, stats

        # ---- 11. the server is still alive ----------------------------------
        st, _, body = raw_http("GET", "/healthz")
        assert st == 200

        print("E2E PASS: turns=4, streamed=%d ids, final stats=%s"
              % (len(streamed), json.dumps(stats, sort_keys=True)))
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
