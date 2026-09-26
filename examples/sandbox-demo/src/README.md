# sandbox-demo

A tiny Python project used to exercise ai-code's devcontainer sandboxing.

## Files

- `src/hello.py` — prints a greeting from the sandbox.
- `.devcontainer/devcontainer.json` — the devcontainer config that makes
  `ai-code` default to running tools in a container.

## Run

```sh
python3 src/hello.py
```

Prints a line of the form `hello from the sandbox (<hostname>)`.
