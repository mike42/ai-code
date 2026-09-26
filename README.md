# ai-code

This is a custom harness for LLM-assisted software development, built to fit my workflow.

- **Sandboxed by default** - uses project-specific dev-containers for tool calls. Running anything on the host requires an extra CLI flag.
- **Attentive to connectivity** - a `.nocloud` file in the launch directory (or any parent) disables cloud-hosted LLM providers for a session.
- **Accessible** - the harness starts quickly, and does not interfere with terminal scrollback, search, or copy/paste functionality.
- **Gets the most out of smaller on-premises setups** - integrates with [lemonade](https://github.com/lemonade-sdk/lemonade) to auto-detect context size, has an IPC co-ordination mechanism so that instances of the harness do not fight over which model is loaded.

## LLM usage note

This project is developed with the use of LLM's.

