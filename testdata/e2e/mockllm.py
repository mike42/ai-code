#!/usr/bin/env python3
"""A deliberately slow, OpenAI-compatible mock server.

It exists to prove one thing: that ai-code's main loop stays usable while a
sub-agent runs. So worker requests are slow on purpose, every request is
logged with a timestamp and a role, and the server is threaded -- a serial
server would hide exactly the property under test.
"""
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("MOCK_PORT", "8099"))
MODEL = os.environ.get("MOCK_MODEL", "Mock-Coder-30B")
WORKER_DELAY = float(os.environ.get("MOCK_WORKER_DELAY", "15"))
PARENT_DELAY = float(os.environ.get("MOCK_PARENT_DELAY", "0.3"))
LOG = os.environ.get("MOCK_LOG", "mock.log")
NCTX = 262144
SLOTS = 3

T0 = time.time()
_loglock = threading.Lock()
_inflight = {"n": 0}


def log(role, msg):
    with _loglock:
        line = "[%7.2fs] [inflight=%d] %-6s %s" % (
            time.time() - T0, _inflight["n"], role, msg)
        with open(LOG, "a") as f:
            f.write(line + "\n")
            f.flush()
        print(line, file=sys.stderr, flush=True)


def is_worker(messages):
    for m in messages:
        if m.get("role") == "system" and "worker agent" in (m.get("content") or ""):
            return True
    return False


def last_tool(messages):
    if messages and messages[-1].get("role") == "tool":
        return messages[-1].get("content") or ""
    return None


def last_user(messages):
    for m in reversed(messages):
        if m.get("role") == "user":
            return m.get("content") or ""
    return ""


def sse(obj):
    return ("data: " + json.dumps(obj) + "\n\n").encode()


def chunk(delta=None, finish=None):
    ch = {"index": 0, "delta": delta or {}}
    if finish:
        ch["finish_reason"] = finish
    return {"id": "mock", "object": "chat.completion.chunk", "model": MODEL,
            "choices": [ch]}


def tool_call_chunk(index, call_id, name, args):
    return chunk(delta={"tool_calls": [{
        "index": index, "id": call_id, "type": "function",
        "function": {"name": name, "arguments": args},
    }]})


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _json(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.rstrip("/")
        if p.endswith("/health"):
            return self._json({
                "model_loaded": MODEL,
                "max_models": {"llm": 1},
                "all_models_loaded": [{
                    "model_name": MODEL, "loaded": True,
                    "is_busy": False, "is_streaming": False, "status": "ready"}],
            })
        if p.endswith("/slots"):
            return self._json([
                {"id": i, "n_ctx": NCTX, "is_processing": False,
                 "n_prompt_tokens": 0, "n_prompt_tokens_processed": 0}
                for i in range(SLOTS)
            ])
        if p.endswith("/models"):
            return self._json({"object": "list", "data": [{
                "id": MODEL, "object": "model", "owned_by": "mock",
                "context_length": NCTX,
                "max_context_length": NCTX,
                "capabilities": ["tool-calling"],
            }]})
        self.send_error(404)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            return self.send_error(404)
        n = int(self.headers.get("Content-Length", "0"))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages", [])
        worker = is_worker(msgs)

        with _loglock:
            _inflight["n"] += 1
        try:
            self._respond(msgs, worker)
        finally:
            with _loglock:
                _inflight["n"] -= 1

    def _respond(self, msgs, worker):
        role = "WORKER" if worker else "PARENT"
        text = last_user(msgs)
        log(role, "request in  : %r" % text[:70])

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        if worker:
            self._worker_turn(msgs)
        else:
            self._parent_turn(msgs)

        self.wfile.write(sse(chunk(delta={}, finish="stop")))
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        log(role, "response out")

    # -- worker ----------------------------------------------------------
    def _worker_turn(self, msgs):
        brief = last_user(msgs)
        tag = "worker"
        m = re.search(r"\[(\w+)\]", brief)
        if m:
            tag = m.group(1)

        out = last_tool(msgs)
        if out is not None:
            # SLOWREPORT holds a *model* slot after the tool call has already
            # returned, so one run can exercise both kinds of concurrency:
            # the container pipe during the command, the server slot after it.
            if "SLOWREPORT" in brief:
                for _ in range(max(1, int(WORKER_DELAY / 0.5))):
                    time.sleep(0.5)
            self._emit_text("FINDING(%s): %s" % (tag, " ".join(out.split())[:300]))
            return

        m = re.search(r"RUN: (.+)", brief)
        if m:
            self.wfile.write(sse(tool_call_chunk(
                0, "w_%s" % tag, "bash",
                json.dumps({"command": m.group(1).strip()}))))
            self.wfile.flush()
            self.wfile.write(sse(chunk(delta={}, finish="tool_calls")))
            self.wfile.flush()
            log("WORKER", "%s runs: %s" % (tag, m.group(1).strip()))
            return

        steps = int(WORKER_DELAY / 0.5)
        for _ in range(max(1, steps)):
            time.sleep(0.5)
        self._emit_text("FINDING(%s): slept %.0fs in a background worker, "
                        "then wrote this line." % (tag, WORKER_DELAY))

    # -- parent ----------------------------------------------------------
    def _parent_turn(self, msgs):
        text = last_user(msgs)
        time.sleep(PARENT_DELAY)

        if "<worker-report" in text:
            self._emit_text("I have the worker report now.")
            return

        out = last_tool(msgs)
        if out is not None:
            if "task" not in json.dumps(msgs[-1])[:200] and out.strip():
                self._emit_text("PARENT SAW: %s" % " ".join(out.split())[:200])
                return
            self._emit_text("Started them; carrying on.")
            return

        if "check cwd" in text:
            self.wfile.write(sse(tool_call_chunk(
                0, "p_pwd", "bash", json.dumps({"command": "pwd"}))))
            self.wfile.flush()
            self.wfile.write(sse(chunk(delta={}, finish="tool_calls")))
            self.wfile.flush()
            log("PARENT", "checks its own cwd")
            return

        m = re.search(r"spawn (\d+)", text)
        if m:
            k = int(m.group(1))
            slowreport = "slow container" in text
            dirs = ["/tmp", "/usr", "/etc"]
            for i in range(k):
                cmd = ("cd %s && sleep %d && echo PWD=$(pwd) DAEMON=$PPID"
                       % (dirs[i % len(dirs)], 2 if slowreport else int(WORKER_DELAY)))
                self.wfile.write(sse(tool_call_chunk(
                    i, "call_%d" % (i + 1), "task",
                    json.dumps({
                        "description": "background job %d" % (i + 1),
                        "prompt": ("[JOB%d] think hard about it" % (i + 1))
                                  if ("slow" in text and not slowreport) else
                                  ("[JOB%d]%s RUN: %s"
                                   % (i + 1, " SLOWREPORT" if slowreport else "", cmd)),
                    }))))
                self.wfile.flush()
            self.wfile.write(sse(chunk(delta={}, finish="tool_calls")))
            self.wfile.flush()
            log("PARENT", "asked for %d worker(s)" % k)
            return

        self._emit_text("ACK(%.1fs): %s" % (time.time() - T0, text[:60]))

    def _emit_text(self, body):
        for word in body.split(" "):
            self.wfile.write(sse(chunk(delta={"content": word + " "})))
            self.wfile.flush()
            time.sleep(0.01)


if __name__ == "__main__":
    open(LOG, "w").close()
    srv = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    srv.daemon_threads = True
    print("mock llm on %d, model=%s, worker delay=%.0fs" % (PORT, MODEL, WORKER_DELAY),
          file=sys.stderr, flush=True)
    srv.serve_forever()
