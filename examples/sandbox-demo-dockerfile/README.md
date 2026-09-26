# sandbox-demo-dockerfile

An example project used to exercise `ai-code`'s devcontainer support where the
container is **built from a Dockerfile** rather than pulled by name. Its sibling,
`../sandbox-demo`, covers the `"image"` case.

## What this project is

Nothing but a devcontainer. There is no program here to build or run -- the
configuration *is* the subject. If you want a project an agent can do work in,
use `../sandbox-demo`.

## Files

- `.devcontainer/Dockerfile` — an Ubuntu 24.04 image with an ordinary build
  toolchain, a passwordless-sudo `ubuntu` user, and git-lfs.
- `.devcontainer/devcontainer.json` — builds that Dockerfile and then exercises
  the properties an image-based configuration never reaches.

## What it exercises

| Key | Why it is here |
|---|---|
| `build.dockerfile`, `build.context` | The image is built, not pulled. `context` is `..`, so it is the example root, not the `.devcontainer` directory |
| `workspaceMount` with `${localWorkspaceFolder}` | Variable substitution; the value is unusable if left literal |
| `workspaceFolder` | `/workspace`, so it differs from the spec default of `/workspaces/<basename>` and a hardcoded guess would be visibly wrong |
| `runArgs: ["--userns=keep-id"]` | Rootless podman. Without it, files the agent writes into the bind mount come back owned by a remapped UID the user cannot edit |
| `remoteUser: "ubuntu"` | Tools run as that user, not as the image's default |
| `overrideCommand` | Read, and currently reported as not applied: it needs the create/start/exec lifecycle, not the single `run` this build uses |

## How to use it

From inside this directory:

```sh
ai-code "run id and pwd, and tell me what OS this is"
```

A correct run reports `/workspace`, the user `ubuntu`, and Ubuntu 24.04 -- that
is, the image the Dockerfile builds, mounted where the configuration asked, as
the user it named.

## Notes

- The first launch builds the image and is slow; the build output is captured
  rather than streamed, so it looks like one slow tool call. Later launches
  reuse the tagged image.
- The image is tagged `ai-code-devcontainer:<hash>`, where the hash covers the
  Dockerfile's contents and the build arguments. Editing the Dockerfile changes
  the tag and rebuilds; leaving it alone reuses.
- `cmd/ai-code/dockerfile_test.go` drives this example end to end, so it cannot
  drift from the code without a test failing.
