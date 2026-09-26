# End-to-end harness: a mock LLM and a PTY driver

```sh
./run.sh            # both scenarios
./run.sh slots      # host runtime only
./run.sh devcontainer
```

Everything lands in `build/`, which is disposable. Nothing reaches a real
provider: the runs redirect `XDG_CONFIG_HOME` and `XDG_RUNTIME_DIR`, so
neither the real provider config nor the cross-window coordination directory
under `$XDG_RUNTIME_DIR/ai-code/peers` is read. That second one matters — the
first attempt at this picked a model name out of a live session in another
window and ran the whole test against the wrong thing.

`XDG_RUNTIME_DIR` is redirected only to a *subdirectory of the real one*.
Podman keeps its network namespaces there as well, and pointing it at an
ordinary directory fails every container with `pasta failed ... Couldn't open
network namespace` -- which surfaces as a worker unable to find its container,
several layers away from the cause.

## What is here

| File | What it does |
|---|---|
| `mockllm.py` | An OpenAI-compatible server: SSE streaming, `/models`, `/health`, a 3-slot `/slots`. Threaded on purpose. Logs every request with a timestamp, a role and an in-flight count. |
| `drive.py` | Runs the real binary under a PTY, types on a **fixed wall clock**, renders the screen with pyte at named moments. |
| `snapshot.py` | Turns a pyte screen into colour-preserving ANSI and HTML. `screen.display` throws the attributes away, which is most of what the screen looks like. |
| `scenarios/*.json` | A scenario is `argv`, `cwd`, `env`, and a script of `[seconds, "type"\|"snap"\|"stop", payload]`. `@ROOT@` and `@BIN@` are substituted by `run.sh`. |

## Output

- `build/out/<name>.snapshots.html` — every snapshot on one page, in colour.
- `build/out/<name>.snapshots.ansi` — the same, `cat` it in a terminal.
- `build/out/<name>.mock.log` — server side, with in-flight counts.
- `build/out/<name>.log` — plain-text screens plus the driver's timeline.
- `build/out/<name>.cast` — asciicast v2, if you want a replay.

**Clocks.** The mock starts before the binary, so its log runs a few seconds
ahead of the driver timeline. The offset is constant — check it against the
first event before reading anything into a gap. One apparent five-second stall
turned out to be exactly this.

## Why the driver types on a clock

A driver that waits for a prompt before typing proves nothing: a session that
blocks for thirty seconds and then accepts input looks identical to one that
never blocked. Keystrokes go in when the clock says so, whether or not
anything is ready, and the snapshot is whatever was on screen at that instant.

## What it was built to prove

That a sub-agent runs in the background and the session stays usable while it
does. Three claims, each with its own evidence:

1. **The main loop is not blocked.** `slots` starts two workers that each
   sleep 15s *inside the model*, so both hold a server slot for the duration.
   A new user prompt is typed and answered while `inflight=3` — parent plus
   both workers generating at once.
2. **Forks are real shells.** `devcontainer` runs the same thing inside
   `examples/sandbox-demo`. Each worker does
   `cd <dir> && sleep 15 && echo PWD=$(pwd) DAEMON=$PPID` and reports a
   *different* daemon pid in the *same* container — two `--executor-daemon`
   processes attached with `exec -i`. The parent's cwd is unmoved afterwards,
   which is `tool.Executor.Fork` giving each agent its own `tool.State`.
3. **Reports come back.** They land in the scrollback on their own and fold
   into the conversation on the next message.

It found four bugs that unit tests did not: a non-interactive run cancelling
its workers on the way out, a folded report printing its body twice, a
stale worker count on the committed prompt line, and a container that failed
to start being reported as ready -- `start()` checked that the *process*
launched, not that a container was running, so the first sign of trouble was a
worker exec'ing into an id that no longer existed.

## Adapting it

The mock is a fixture, not a model, and it will need editing for anything new.
It dispatches on the last user message and on whether the system prompt
contains `worker agent`:

| Trigger | Behaviour |
|---|---|
| `spawn N …` | emits N `task` calls |
| `… slow …` | the workers sleep in the model rather than running a command |
| `check cwd` | emits a `bash` call running `pwd` |
| last message is a tool result | replies with text, ending the turn |
| anything else | `ACK(<t>): <text>` |

A worker whose brief contains `RUN: <cmd>` emits one `bash` call for that
command and reports its output; otherwise it sleeps `MOCK_WORKER_DELAY`
seconds and reports.

Env: `MOCK_PORT`, `MOCK_MODEL`, `MOCK_WORKER_DELAY`, `MOCK_PARENT_DELAY`.

Adding a scenario is a JSON file in `scenarios/` plus, usually, a new branch
in `_parent_turn`. Keep the trigger a plain substring — the point of this
fixture is that you can read what it will do without running it.
