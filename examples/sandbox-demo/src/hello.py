#!/usr/bin/env python3
"""Command hello prints a greeting. It exists so the sandbox demo has
something trivial to run inside the devcontainer."""

import os


def main():
    host = os.uname().nodename
    print(f"hello from the sandbox ({host})")


if __name__ == "__main__":
    main()
