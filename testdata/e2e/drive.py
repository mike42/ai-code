#!/usr/bin/env python3
"""Drive ai-code through a real PTY and photograph the screen.

Keystrokes go in on a wall-clock schedule, output is recorded with
timestamps, and the terminal is rendered with pyte at named moments. The
schedule is fixed in advance, so a session that only *looks* responsive --
because the driver waited for a prompt before typing -- cannot pass: the
driver types when the clock says to, whether or not anything is ready.
"""
import json
import os
import pty
import select
import signal
import sys
import time

import pyte

import snapshot

COLS, ROWS = 100, 34


def run(argv, script, cwd, env, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    screen = pyte.Screen(COLS, ROWS)
    stream = pyte.ByteStream(screen)

    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(cwd)
        os.environ.update(env)
        os.environ["TERM"] = "xterm-256color"
        os.environ["COLUMNS"] = str(COLS)
        os.environ["LINES"] = str(ROWS)
        os.execv(argv[0], argv)

    import fcntl
    import struct
    import termios
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))

    t0 = time.time()
    raw = bytearray()
    cast = []  # (offset, bytes) for an asciicast v2 recording
    events = []
    shots = []
    pending = list(script)

    def snap(label):
        lines = [screen.display[i].rstrip() for i in range(ROWS)]
        while lines and not lines[-1]:
            lines.pop()
        shots.append({"t": round(time.time() - t0, 2), "label": label,
                      "lines": lines,
                      "ansi": snapshot.to_ansi(screen),
                      "html": snapshot.to_html(screen)})

    deadline = max(s[0] for s in script) + 30
    while time.time() - t0 < deadline:
        now = time.time() - t0
        while pending and pending[0][0] <= now:
            _, kind, payload = pending.pop(0)
            if kind == "type":
                os.write(fd, payload.encode())
                events.append((round(now, 2), "TYPED", payload.strip() or repr(payload)))
            elif kind == "snap":
                snap(payload)
                events.append((round(now, 2), "SNAP", payload))
            elif kind == "stop":
                events.append((round(now, 2), "STOP", ""))
                pending = []
                deadline = 0

        r, _, _ = select.select([fd], [], [], 0.05)
        if r:
            try:
                data = os.read(fd, 65536)
            except OSError:
                break
            if not data:
                break
            raw += data
            cast.append((time.time() - t0, data))
            stream.feed(data)

    snap("final")
    try:
        os.write(fd, b"\x04")
        time.sleep(0.4)
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    try:
        os.waitpid(pid, os.WNOHANG)
    except OSError:
        pass

    with open(os.path.join(outdir, name + ".raw"), "wb") as f:
        f.write(bytes(raw))

    # asciicast v2: replay it with `asciinema play`, or read it as JSON.
    with open(os.path.join(outdir, name + ".cast"), "w") as f:
        f.write(json.dumps({"version": 2, "width": COLS, "height": ROWS,
                            "title": name, "env": {"TERM": "xterm-256color"}}) + "\n")
        for off, data in cast:
            f.write(json.dumps([round(off, 3), "o",
                                data.decode("utf-8", "replace")]) + "\n")
    with open(os.path.join(outdir, name + ".json"), "w") as f:
        json.dump({"events": events,
                   "shots": [{k: v for k, v in s.items() if k != "html"}
                             for s in shots]}, f, indent=1)

    # Static snapshots, which is the point: reviewable without a replay.
    with open(os.path.join(outdir, name + ".snapshots.ansi"), "w") as f:
        for s in shots:
            f.write("\n\x1b[1;36m\u2500\u2500 %s  (t=%.2fs)\x1b[0m\n\n"
                    % (s["label"], s["t"]))
            f.write(s["ansi"] + "\n")
    with open(os.path.join(outdir, name + ".snapshots.html"), "w") as f:
        f.write(snapshot.page("ai-code background workers \u2014 " + name,
                              "Terminal state captured at fixed moments "
                              "during a scripted PTY session.", shots))

    out = []
    for s in shots:
        out.append("=" * COLS)
        out.append("  SCREEN @ t=%5.2fs   %s" % (s["t"], s["label"]))
        out.append("=" * COLS)
        out.extend(s["lines"])
        out.append("")
    text = "\n".join(out)
    with open(os.path.join(outdir, name + ".screens.txt"), "w") as f:
        f.write(text)
    return events, shots, text


if __name__ == "__main__":
    spec = json.load(open(sys.argv[1]))
    events, shots, text = run(
        spec["argv"], [tuple(s) for s in spec["script"]],
        spec["cwd"], spec["env"], spec["outdir"], spec["name"])
    print(text)
    print("---- DRIVER TIMELINE ----")
    for t, k, v in events:
        print("[%6.2fs] %-5s %s" % (t, k, v))
