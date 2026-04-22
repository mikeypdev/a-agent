# a-agent

A minimal, single-file AI coding agent optimized for terminal-based workflows. Designed specifically for [a-Shell](https://holzschu.github.io/a-Shell_iOS/) on iOS/iPadOS, but works on any system with Python 3.

It connects to LLMs via the OpenRouter API (or any OpenAI-compatible endpoint) and gives the model tools to read, write, and edit files, run shell commands, search the web, and browse documentation — all from inside your terminal with streaming responses, live thinking display, and syntax highlighting.

> ⚠️ **Warning:** This agent has no guardrails. It can read, write, and delete any file the running user can access. On iOS, a-Shell's sandbox provides some protection. Use at your own risk.
>
> Because a-Shell runs on iOS, it is subject to performance constraints unlike desktop terminals. Large tasks — especially ones that make many shell tool calls — can potentially crash a-Shell.

---

## Quick Start

### 1. Install dependencies

```bash
pip install rich beautifulsoup4 markdownify
```

### 2. Set your API key

```bash
export API_KEY="sk-or-..."
```

Get a key at [openrouter.ai](https://openrouter.ai/).

### 3. Run

```bash
python3 a-agent.py
```

The default model is `anthropic/claude-sonnet-4-5` (via OpenRouter).

---

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `API_KEY` | *(required)* | Your OpenRouter (or compatible) API key |
| `MODEL` | `anthropic/claude-sonnet-4-5` | Model ID to use |
| `API_BASE` | `https://openrouter.ai/api/v1` | API base URL |
| `CONTEXT_WINDOW` | *(auto-detected)* | Override the model's context window size (tokens) |
| `MAX_TOKENS` | *(auto-detected)* | Override the model's max output tokens |
| `THINKING` | `1` | Enable extended thinking (`1`/`true` or `0`/`false`) |
| `SYSTEM_MD` | *(none)* | Path to a markdown file to use as the system prompt |
| `SHELL_TOOL_CONFIRMATION` | `1` | Require confirmation before running shell commands (`1`/`true` or `0`/`false`) |
| `DEBUG_LOG` | *(none)* | Path to a file for debug logging (e.g. `debug.log`) |

---

## Project Instructions

If an `AGENTS.md` file exists in the working directory, its contents are automatically appended to the system prompt. Use this to give the agent project-specific context, coding conventions, or constraints.

---

## File Ignore Patterns

The agent uses `.gitignore` and `.agentignore` to keep noise out of search/listing results and file tools. Files that look like secrets (`.env`, `.pem`, `.key`, etc.) are similarly filtered. **This is not a security boundary** — `run_shell` bypasses all filtering.

---

## Controls

| Key / Command | Action |
|---|---|
| **`c` + Enter** | **Cancel** the agent while it is thinking or responding. (`cancel` and `stop` also work.) It may take up to 10 seconds to process cancellation.|
| **`quit`** / **`exit`** | Close the agent (type at the prompt). **``q``** also works.|

*Note: In a-Shell, `Ctrl+C` kills the app (or at least is supposed to). Use `c` + Enter to cancel a response instead.*

---

## How It Works

The agent follows a research-driven loop:

1. **Prompting:** You give a task; the agent analyzes the context and existing code.
2. **Streaming:** The model's "Thinking" process is displayed live in a dimmed panel, followed by its content.
3. **Tool Use:** If the model needs more information or wants to make changes, it calls its tools.
4. **Verification:** The agent encourages running tests or shell commands to verify all changes.

### Tools

| Tool | Description |
|---|---|
| `read_file` | Read a file (supports line ranges). |
| `write_file` | Create or overwrite a file. |
| `edit_file` | Precise find-and-replace (requires exact match). |
| `run_shell` | Run a shell command. Pipes, `$()`, backticks, and `&` are blocked (they can hang a-Shell). `&&`, `\|\|`, `;` chains are split and run sequentially. |
| `list_dir` | List files in a directory. |
| `grep` | Search file contents with regex. |
| `glob` | Find files matching patterns (e.g. `**/*.py`). |
| `web_search` | Search the web using DuckDuckGo. |
| `read_url` | Fetch any webpage and convert it to clean Markdown (great for docs). |

### Context Management

When the history approaches 80% of the model's limit, the agent automatically summarizes the conversation so far, preserving key facts while freeing up space for new work.

---

## License

See [LICENSE](LICENSE).
