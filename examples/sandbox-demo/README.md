# sandbox-demo

An example project used to exercise `ai-code`'s devcontainer sandboxing. This
repository deliberately looks like an ordinary small project, so that you can
ask an agent to build and run it and get a fair read on whether it understands
the task.

## What this project is

A tiny Python program that prints a greeting. That is all. There is nothing
hidden here, and there is no deeper purpose -- the point is a project whose
build is small and unambiguous enough to serve as a test.

## Files

- `src/hello.py` — the program: prints a greeting, including the hostname of
  the machine it runs on.
- `.devcontainer/devcontainer.json` — the devcontainer config. Its presence is
  what makes `ai-code` default to running its tools inside a container.

## How to build and run

```sh
python3 src/hello.py
```

That is the whole build: there are no dependencies to install and nothing to
compile. The program is self-contained in the standard library.

## What a correct run looks like

```text
hello from the sandbox (<hostname>)
```

The `<hostname>` will be the name of the machine inside the sandbox, so it is
not a value you can predict -- but the line should match that shape.

## What to ask an agent

From inside this directory:

```sh
ai-code "list the files and tell me what this project does"
ai-code "build the project"
ai-code "run the project"
```

The last two should both succeed and produce a line of the shape above. If an
agent spends time installing dependencies, compiling, or describing how it
"would" build things, it has misread the project.

## Notes

- This box uses `podman`, not `docker`. `ai-code` probes for either on the first
  tool call, never at startup, so the prompt appears immediately.
- The binary is mounted into the container at `/ai-code-bin` and runs in
  `--executor-daemon` mode; the project is mounted at `/workspace`, which is
  what `workspaceFolder` asks for. Without it the mount would go to the spec's
  default, `/workspaces/sandbox-demo`.
- This configuration used to carry a `mounts` entry binding a host `/workspace`
  that does not exist. It was inert while `mounts` was ignored; it is removed
  now that the key is honoured. See `../sandbox-demo-dockerfile` for the
  Dockerfile-based counterpart.
