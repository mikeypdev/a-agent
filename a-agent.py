#!/usr/bin/env python3
"""
AI Coding Agent for a-Shell
Uses OpenAI API with context compaction, syntax highlighting, and live thinking display.

Install: pip install rich beautifulsoup4 markdownify
Run:     export API_KEY=sk-or-...
         export API_BASE=https://openrouter.ai/api/v1  (optional, default)
         export MODEL=anthropic/claude-sonnet-4-5       (optional)
         export DEBUG_LOG=a-agent-debug.log             (optional, enables debug logging)
         python3 a-agent.py
"""

from collections.abc import Callable
import difflib
import http.client
import io
import json
import logging
import os
import queue
import re
import shlex
import subprocess
import sys
import threading
import time
import urllib.parse

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_KEY     = os.environ.get("API_KEY", "")
MODEL       = os.environ.get("MODEL", "anthropic/claude-sonnet-4-5")
API_BASE    = os.environ.get("API_BASE", "https://openrouter.ai/api/v1")
CONTEXT_WINDOW = int(os.environ.get("CONTEXT_WINDOW", "0"))
MAX_TOKENS     = int(os.environ.get("MAX_TOKENS", "0"))
THINKING    = os.environ.get("THINKING", "1").lower() not in ("0", "false", "no", "off")
SHELL_TOOL_CONFIRMATION = os.environ.get("SHELL_TOOL_CONFIRMATION", "1").lower() not in ("0", "false", "no", "off")

# ---------------------------------------------------------------------------
# Debug logging  (set DEBUG_LOG to a file path to enable, e.g. a-agent-debug.log)
# ---------------------------------------------------------------------------

_logger = logging.getLogger("a-agent")

def _setup_debug_logging() -> None:
    log_path = os.environ.get("DEBUG_LOG", "")
    if not log_path:
        _logger.setLevel(logging.CRITICAL + 1)
        return
    _logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S"
    ))
    _logger.addHandler(handler)
    _logger.info("Debug logging started → %s", log_path)
    _logger.info("  MODEL=%s  API_BASE=%s  THINKING=%s", MODEL, API_BASE, THINKING)

_setup_debug_logging()

def debug(fmt: str, *args) -> None:
    _logger.debug(fmt, *args)


# ---------------------------------------------------------------------------
# Thinking format detection
# ---------------------------------------------------------------------------

def _detect_thinking_format(base_url: str) -> str:
    if "openrouter.ai" in base_url:
        return "openrouter"
    if "api.z.ai" in base_url or "bigmodel.cn" in base_url:
        return "zai"
    return "openai"

THINKING_FORMAT = _detect_thinking_format(API_BASE)

THINKING_BUDGETS = {
    "minimal": 1024,
    "low":     2048,
    "medium":  8192,
    "high":    16384,
}

REASONING_FIELDS = ("reasoning_content", "reasoning", "reasoning_text")


def thinking_request_params(enabled: bool = True) -> dict:
    if not enabled:
        if THINKING_FORMAT == "openrouter":
            return {"reasoning": {"effort": "none"}}
        return {}
    if THINKING_FORMAT == "openrouter":
        return {"reasoning": {"effort": "high"}}
    if THINKING_FORMAT == "zai":
        return {"thinking": {"type": "enabled"}}
    return {"reasoning_effort": "high"}


def adjust_max_tokens_for_thinking(
    base_max_tokens: int,
    model_max_tokens: int,
    context_window: int,
    used_tokens: int,
    level: str = "high",
) -> int:
    # Ensure input + output never exceeds context_window.
    # available = how many tokens remain for output after input.
    available = context_window - used_tokens
    if available <= 1024:
        return 0

    if not THINKING:
        # No thinking budget needed — cap output at available space minus buffer.
        return max(512, min(base_max_tokens, available - 1024))

    # Thinking models use internal tokens for reasoning (not counted in max_tokens).
    # Budget for internal reasoning steps from the available space.
    budget = min(THINKING_BUDGETS.get(level, 8192), available // 2)
    adjusted = min(base_max_tokens + budget, model_max_tokens, available - 1024)
    return max(adjusted, 512)


_base       = urllib.parse.urlparse(API_BASE)
API_SCHEME  = _base.scheme or "https"
API_HOST    = _base.hostname
API_PREFIX  = _base.path.rstrip("/")
API_PATH    = f"{API_PREFIX}/chat/completions"

if not API_HOST:
    print(f"Error: API_BASE is malformed or missing a hostname: {API_BASE!r}")
    print("  Expected format: https://api.example.com/v1")
    raise SystemExit(1)

USER_AGENT = "a-agent/1.0"
DEFAULT_MODEL_INFO = {"context_window": 128_000, "max_tokens": 8192}
COMPACT_THRESHOLD  = 0.80
MAX_TOOL_OUTPUT    = 50_000

# ---------------------------------------------------------------------------
# Persistent HTTP(S) connection pool
# ---------------------------------------------------------------------------

_conn_type = http.client.HTTPSConnection if API_SCHEME == "https" else http.client.HTTPConnection
_persistent_conn: http.client.HTTPConnection | None = None
_conn_lock = threading.Lock()


def _get_connection() -> http.client.HTTPConnection:
    global _persistent_conn
    with _conn_lock:
        if _persistent_conn is not None:
            return _persistent_conn
        _persistent_conn = _conn_type(API_HOST, timeout=120)
        return _persistent_conn


def _reset_connection() -> None:
    global _persistent_conn
    with _conn_lock:
        if _persistent_conn is not None:
            try:
                _persistent_conn.close()
            except Exception:
                pass
            _persistent_conn = None


def _api_request(method: str, path: str, body: bytes, headers: dict) -> http.client.HTTPResponse:
    debug("HTTP %s %s body=%d bytes", method, path, len(body))
    conn = _get_connection()
    try:
        with _conn_lock:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
        debug("HTTP response: %s %s", resp.status, resp.reason)
        return resp
    except (ConnectionResetError, OSError, http.client.HTTPException) as e:
        debug("Connection error on first attempt, reconnecting: %s", e)
        _reset_connection()
        conn = _get_connection()
        with _conn_lock:
            conn.request(method, path, body=body, headers=headers)
            resp = conn.getresponse()
        debug("HTTP response (retry): %s %s", resp.status, resp.reason)
        return resp


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a coding agent running in a terminal. You help the user write and modify code.

You have these tools:
- read_file: read a file's contents (supports optional offset and limit for line ranges)
- write_file: create or overwrite a file
- edit_file: find-and-replace an exact substring in a file (always read first)
- run_shell: run a shell command
- list_dir: list directory contents
- grep: search file contents with regex (supports recursive directory search and filename glob filtering)
- glob: find files/dirs matching a glob pattern (e.g. '**/*.py', 'src/**/*.js'). Much easier than shell find.
- web_search: search the web using DuckDuckGo. Returns titles, URLs, and snippets.
- read_url: fetch a web page and return its content as clean markdown. Use for reading docs, blog posts, API references, etc.

Rules:
- Always read a file before editing it.
- Use edit_file for targeted changes, write_file only for new files.
- After making changes, run relevant commands to verify they work, if supported by platform.
- Be concise.
- **Do not pipe commands** (e.g. `cmd | head`). Tool output is auto-truncated (50KB, 60 lines displayed). Just run the command — you'll see the important parts. `head`, `tail`, `sort` etc. are fine on their own (e.g. `tail -n 30 file`).

## Shell environment — READ THIS

You are running on **iOS** (Darwin/arm64) inside the **a-Shell** app. All coreutils are **BSD**, NOT GNU.

Critical differences from GNU/Linux that you MUST respect:

| GNU (wrong here)            | BSD (correct here)                  |
|-----------------------------|-------------------------------------|
| `sed -i 's/a/b/' file`     | `sed -i '' 's/a/b/' file`          |
| `sed -i.bak`                | `sed -i '' -e ...` (empty ext)      |
| `grep -P` (Perl regex)     | `grep -E` (extended regex only)     |
| `grep --exclude-dir=dir`   | NOT available; pipe through grep -v |
| `grep --color=auto`        | NOT available                       |
| `date -d '2 days ago'`     | `date -v-2d`                        |
| `date +%s%N` (nanoseconds) | NOT available (only %s epoch)       |
| `find -regex`              | uses POSIX basic regex, not GNU     |
| `find -printf`             | NOT available                       |
| `readlink -f`              | NOT available; use `realpath` or python |
| `xargs -r` (no-run-empty)  | NOT available                       |
| `sort -V` (version sort)   | NOT available                       |
| `wc -l`                     | works fine ✓                        |
| `head -n`, `tail -n`       | works fine ✓                        |
| `awk`                       | works (nawk/mawk style) ✓           |
| `python3`                   | available ✓                         |
| `perl`                     | available ✓                         |

**There is no package manager (no brew, apt, pkg).** Only pip for Python packages.
Do NOT suggest installing GNU coreutils.

**Do NOT use `bash`.** It is NOT installed. Use `dash` instead.
Never generate commands like `bash -c '...'` or `#!/bin/bash` — they will crash the app.
For shell scripts use `#!/bin/sh` or `#!/usr/bin/env dash`.

**a-Shell has significant resource constraints**, so minimize usage of large shell commands.

## a-Shell pipe and concurrency constraints — CRITICAL

**Do NOT use pipes (|) in shell commands.** a-Shell builtins run inside a single process and can only execute one at a time. Piped builtins WILL hang the app permanently.

These constructs will be **rejected** at execution time — do not use them:

| Rejected construct              | Reason                                        | Alternative                              |
|---------------------------------|-----------------------------------------------|------------------------------------------|
| `cmd1 | cmd2` (pipe)           | Requires concurrent builtins → hang           | Use built-in tools or run separately     |
| `$(cmd)` or `` `cmd` ``        | Command substitution → concurrent builtin      | Run inner cmd first, then use result     |
| `cmd &` (background)           | Concurrent execution → hang                    | Run in foreground only                   |
| `<<< "text"` (here-string)     | Bashism, not supported by dash                 | Use temp file or `python3 -c`           |
| `<(cmd)` (process sub)         | Requires concurrent execution → hang           | Use temp file or `python3 -c`           |

These **are** allowed and will be split automatically:

| Allowed            | Behavior                                      |
|--------------------|-----------------------------------------------|
| `cmd1 && cmd2`     | Run sequentially, stop if one fails           |
| `cmd1 || cmd2`     | Run sequentially, stop if one succeeds        |
| `cmd1 ; cmd2`      | Run all sequentially                          |

Prefer the built-in tools (grep, glob, read_file, list_dir) over shell commands — they are faster, safer, and don't risk hanging.

## File safety

Files matching `.gitignore` or `.agentignore` patterns are automatically filtered from search results (grep, glob, list_dir) and blocked from being read, written, or edited directly. Secret-looking filenames (.env, .pem, .key, etc.) are also blocked from writes and edits. If you truly need to access one of these files, use run_shell to do it explicitly."""


def load_system_prompt() -> str:
    system_md = os.environ.get("SYSTEM_MD", "")
    if system_md:
        try:
            with open(system_md) as f:
                prompt = f.read()
        except Exception as e:
            prompt = f"(Failed to load SYSTEM_MD from {system_md}: {e})"
    else:
        prompt = SYSTEM_PROMPT

    try:
        with open("AGENTS.md") as f:
            prompt += "\n\n# Project instructions\n\n" + f.read()
    except FileNotFoundError:
        pass

    return prompt


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file from the filesystem. Supports optional line range via offset (1-based start line) and limit (number of lines).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {
                        "type": "integer",
                        "description": "1-based line number to start reading from. Omit to start at line 1.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Omit to read all lines.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run a shell command and return stdout + stderr",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in a directory",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace an exact substring in a file. old_string must match uniquely unless replace_all is true.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a regex pattern in files. Returns matching lines with filenames and line numbers. Uses extended regex. Searches recursively when given a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Extended regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search. For a directory, searches all files recursively.",
                    },
                    "include": {
                        "type": "string",
                        "description": "Optional glob filter for filenames when searching a directory (e.g. '*.py').",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files and directories matching a glob pattern. Much easier than shell find. Returns sorted list of matching paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py', 'src/**/*.js', 'test_*.c', '*.md'). Supports ** for recursive directory matching.",
                    },
                    "path": {
                        "type": "string",
                        "description": "Base directory to search from. Defaults to '.'.",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web using DuckDuckGo. Returns top results with titles, URLs, and snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_url",
            "description": "Fetch a web page and return its content as clean markdown. Useful for reading documentation, blog posts, API docs, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "URL to fetch.",
                    },
                    "max_length": {
                        "type": "integer",
                        "description": "Max characters of markdown to return (default 10000). Use smaller values for quick checks, larger for deep reads.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Streaming API
# ---------------------------------------------------------------------------

def iter_sse_chunks(messages: list, max_tokens: int):
    req = {
        "model":      MODEL,
        "max_tokens": max_tokens,
        "tools":      TOOLS,
        "stream":     True,
        "messages":   messages,
    }
    req.update(thinking_request_params(THINKING))
    body = json.dumps(req).encode()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
        "Accept":        "text/event-stream",
        "User-Agent":    USER_AGENT,
    }

    resp = None
    _had_error = False
    try:
        resp = _api_request("POST", API_PATH, body=body, headers=headers)

        if resp.status != 200:
            raw = resp.read().decode(errors="replace")
            _reset_connection()
            _had_error = True
            debug("SSE error: HTTP %d — %s", resp.status, raw[:500])
            yield {"type": "error", "text": f"HTTP {resp.status}: {raw}"}
            return

        debug("SSE stream opened, reading chunks…")

        tool_call_accum: dict[int, dict] = {}
        finish_reason = None
        saw_done = False

        while True:
            line = resp.readline()
            if not line:
                break
            line = line.decode(errors="replace").rstrip("\r\n")

            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                saw_done = True
                break

            try:
                chunk = json.loads(payload)
            except json.JSONDecodeError:
                continue

            choice = (chunk.get("choices") or [{}])[0]
            new_fr = choice.get("finish_reason")
            if new_fr:
                finish_reason = new_fr
            delta = choice.get("delta", {})

            thinking_yielded = False
            for field in REASONING_FIELDS:
                val = delta.get(field)
                if val and isinstance(val, str) and len(val) > 0:
                    yield {"type": "thinking", "text": val}
                    thinking_yielded = True
                    break
            if not thinking_yielded:
                for rd in delta.get("reasoning_details", []):
                    if rd.get("type") == "reasoning.text" and rd.get("text"):
                        yield {"type": "thinking", "text": rd["text"]}
                        thinking_yielded = True
                        break
            if not thinking_yielded:
                for tb in delta.get("thinking_blocks", []):
                    if tb.get("type") == "thinking" and tb.get("thinking"):
                        yield {"type": "thinking", "text": tb["thinking"]}

            if delta.get("content"):
                yield {"type": "text", "text": delta["content"]}

            for tc_delta in delta.get("tool_calls", []):
                idx = tc_delta.get("index", 0)
                if idx not in tool_call_accum:
                    tool_call_accum[idx] = {
                        "id":       "",
                        "type":     "function",
                        "function": {"name": "", "arguments": ""},
                    }
                acc = tool_call_accum[idx]
                if tc_delta.get("id"):
                    acc["id"] = tc_delta["id"]
                fn = tc_delta.get("function", {})
                if fn.get("name"):
                    acc["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    acc["function"]["arguments"] += fn["arguments"]

        if tool_call_accum:
            debug("SSE: %d tool_calls accumulated", len(tool_call_accum))
            yield {"type": "tool_calls", "tool_calls": list(tool_call_accum.values())}

        if not saw_done and finish_reason is None:
            debug("SSE: stream interrupted (no [DONE], no finish_reason)")
            yield {"type": "finish", "reason": "interrupted"}
        else:
            debug("SSE: stream done, finish_reason=%s", finish_reason or "stop")
            yield {"type": "finish", "reason": finish_reason or "stop"}

    except Exception as exc:
        _had_error = True
        _reset_connection()
        yield {"type": "error", "text": str(exc)}
    finally:
        if resp is not None:
            try:
                # Use a short timeout so draining doesn't block for minutes
                old_timeout = resp.sock.gettimeout() if resp.sock else None
                if resp.sock:
                    resp.sock.settimeout(5)
                resp.read()
                if resp.sock and old_timeout is not None:
                    resp.sock.settimeout(old_timeout)
            except Exception:
                pass
        if _had_error:
            _reset_connection()


# ---------------------------------------------------------------------------
# Non-streaming API helpers
# ---------------------------------------------------------------------------

_model_info_cache: dict | None = None


def fetch_model_info(model_id: str) -> dict:
    global _model_info_cache
    if _model_info_cache is not None:
        return _model_info_cache

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "User-Agent":    USER_AGENT,
    }

    try:
        model_path = f"{API_PREFIX}/models/{urllib.parse.quote(model_id, safe='')}"
        resp = _api_request("GET", model_path, body=b"", headers=headers)
        raw = resp.read()
        if resp.getheader("Connection", "").lower() == "close":
            _reset_connection()
        if resp.status == 200:
            data = json.loads(raw)
            m = data.get("data", data)
            info = {
                "context_window": m.get("context_length") or DEFAULT_MODEL_INFO["context_window"],
                "max_tokens": (
                    m.get("top_provider", {}).get("max_completion_tokens")
                    or m.get("max_completion_tokens")
                    or DEFAULT_MODEL_INFO["max_tokens"]
                ),
            }
            _model_info_cache = info
            return info
    except Exception:
        _reset_connection()

    try:
        catalog_path = f"{API_PREFIX}/models"
        resp = _api_request("GET", catalog_path, body=b"", headers=headers)
        raw = resp.read()
        if resp.getheader("Connection", "").lower() == "close":
            _reset_connection()
        if resp.status == 200:
            data = json.loads(raw)
            for m in data.get("data", []):
                if m.get("id") == model_id:
                    info = {
                        "context_window": m.get("context_length") or DEFAULT_MODEL_INFO["context_window"],
                        "max_tokens": (
                            m.get("top_provider", {}).get("max_completion_tokens")
                            or DEFAULT_MODEL_INFO["max_tokens"]
                        ),
                    }
                    _model_info_cache = info
                    return info
    except Exception:
        _reset_connection()

    return DEFAULT_MODEL_INFO


def call_api_simple(messages: list, max_tokens: int, retries: int = 3) -> dict:
    body = json.dumps({
        "model":      MODEL,
        "max_tokens": max_tokens,
        "messages":   messages,
    }).encode()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
        "User-Agent":    USER_AGENT,
    }
    last_err = None
    for attempt in range(retries):
        resp = None
        try:
            resp = _api_request("POST", API_PATH, body=body, headers=headers)
            raw = resp.read()
            if resp.getheader("Connection", "").lower() == "close":
                _reset_connection()
            if resp.status != 200:
                raise Exception(f"HTTP {resp.status}: {raw.decode(errors='replace')}")
            return json.loads(raw)
        except Exception as e:
            if isinstance(e, (ConnectionResetError, OSError, http.client.HTTPException)):
                _reset_connection()
            last_err = e
            if attempt < retries - 1:
                time.sleep(1 * (attempt + 1))
    raise last_err


# ---------------------------------------------------------------------------
# Tool output truncation
# ---------------------------------------------------------------------------

def truncate_result(text: str, max_bytes: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= max_bytes:
        return text
    total_lines = text.count("\n") + (0 if text.endswith("\n") else 1)
    half = max_bytes // 2
    head = text[:half]
    tail = text[-half:]
    return (
        head
        + f"\n\n… ✂ truncated ({len(text):,} bytes, {total_lines:,} lines total) …\n\n"
        + tail
    )


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

class ToolRunner:

    _SENSITIVE_PATTERNS = re.compile(
        r'(\.env|\.pem|\.key|\.secret|credentials?|\.htpasswd|id_rsa|id_ed25519)',
        re.I,
    )

    def __init__(self, max_output: int = MAX_TOOL_OUTPUT):
        self._max_output = max_output
        self._gitignore_patterns: list[tuple[str, bool]] = self._load_gitignore()
        self._agentignore_patterns: list[tuple[str, bool]] = self._load_gitignore(".agentignore")
        self._dispatch: dict[str, Callable] = {
            "read_file":  self._read_file,
            "write_file": self._write_file,
            "run_shell":  self._run_shell,
            "edit_file":  self._edit_file,
            "list_dir":   self._list_dir,
            "grep":       self._grep,
            "glob":       self._glob,
            "web_search": self._web_search,
            "read_url":   self._read_url,
        }

    # ------------------------------------------------------------------
    # .gitignore / .agentignore support
    # ------------------------------------------------------------------

    def _load_gitignore(self, filename: str = ".gitignore") -> list[tuple[str, bool]]:
        """Load ignore-file patterns as (pattern, is_negation) tuples."""
        import fnmatch as _fnmatch
        patterns: list[tuple[str, bool]] = []
        try:
            with open(filename) as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    if not line or line.startswith("#"):
                        continue
                    negated = line.startswith("!")
                    if negated:
                        line = line[1:]
                    pat = line.rstrip("/")
                    if pat:
                        patterns.append((pat, negated))
        except FileNotFoundError:
            pass
        return patterns

    def _is_ignored(self, path: str) -> bool:
        """Check if a path matches .gitignore or .agentignore patterns."""
        import fnmatch as _fnmatch
        rel = os.path.relpath(path)
        parts = rel.split(os.sep)
        basename = os.path.basename(rel)

        for pattern_list in (self._gitignore_patterns, self._agentignore_patterns):
            ignored = False
            for pat, negated in pattern_list:
                matched = (
                    _fnmatch.fnmatch(basename, pat)
                    or _fnmatch.fnmatch(rel, pat)
                    or any(_fnmatch.fnmatch(p, pat) for p in parts)
                )
                if matched:
                    ignored = not negated
            if ignored:
                return True
        return False

    def run(self, name: str, inputs: dict) -> str:
        debug("tool: %s(%s)", name, json.dumps(inputs)[:200])
        handler = self._dispatch.get(name)
        if handler is None:
            debug("tool: unknown tool %s", name)
            return f"Unknown tool: {name}"
        try:
            result = handler(inputs)
            debug("tool: %s → %d bytes", name, len(result))
            return result
        except Exception as e:
            debug("tool: %s error: %s", name, e)
            return str(e)

    def _read_file(self, inputs: dict) -> str:
        path = self._expand_home(inputs["path"])
        if self._is_ignored(path):
            return f"⛔ {path} is in .gitignore/.agentignore and will not be read. Use run_shell to override if intentional."
        offset = inputs.get("offset")
        limit = inputs.get("limit")

        if offset is not None or limit is not None:
            start = max((offset or 1), 1)
            selected: list[str] = []
            with open(path) as f:
                for i, line in enumerate(f, start=1):
                    if i < start:
                        continue
                    if limit is not None and i >= start + limit:
                        break
                    selected.append(line)
            with open(path) as f:
                total = sum(1 for _ in f)
            end = start + len(selected) - 1
            header = f"(lines {start}–{end} of {total})\n"
            return truncate_result(header + "".join(selected), self._max_output)

        size = os.path.getsize(path)
        if size > self._max_output:
            total_lines = 0
            head_lines: list[str] = []
            with open(path) as f:
                for i, line in enumerate(f):
                    total_lines += 1
                    if i < self._max_output // 80:
                        head_lines.append(line)
            head = "".join(head_lines)
            return (
                head
                + f"\n\n… ✂ file too large ({size:,} bytes, {total_lines:,} lines total); "
                + f"showing first {len(head_lines)} lines …\n"
            )
        with open(path) as f:
            return f.read()

    def _write_file(self, inputs: dict) -> str:
        path = self._expand_home(inputs["path"])
        if self._is_ignored(path) or self._SENSITIVE_PATTERNS.search(path):
            return f"⛔ Blocked write to {path}: file is gitignored or looks like a secrets file. Use run_shell to override if intentional."
        with open(path, "w") as f:
            f.write(inputs["content"])
        return f"Written: {path}"

    # Regex to detect "python3 -c 'code'" or "python3 -c \"code\"" one-liners
    _PYTHON_C_RE = re.compile(
        r"""^python3?\s+-c\s+(["'])(.*)\1\s*$""", re.DOTALL
    )

    # Regex to detect "cd <dir> && python3 -c '...'" with optional leading cd+&&
    _CD_PYTHON_C_RE = re.compile(
        r"""^cd\s+(\S+)\s*&&\s+python3?\s+-c\s+(["'])(.*)\2\s*$""", re.DOTALL
    )

    @staticmethod
    def _expand_home(path: str) -> str:
        """Expand ~ and ~/... but leave a-Shell bookmarks like ~mydir alone."""
        if path == '~' or path.startswith('~/'):
            return os.path.expanduser(path)
        return path

    def _run_shell(self, inputs: dict) -> str:
        command = inputs["command"].strip()
        command = re.sub(r'\s*2>&1\s*', ' ', command).strip()

        # Python fast paths — avoid shell entirely
        m = self._PYTHON_C_RE.match(command)
        if m:
            return self._run_python_subprocess(m.group(2))

        cd_m = self._CD_PYTHON_C_RE.match(command)
        if cd_m:
            target_dir = self._expand_home(cd_m.group(1))
            code = f"os.chdir({target_dir!r})\n{cd_m.group(3)}"
            return self._run_python_subprocess(code)

        heredoc_m = re.match(
            r"""^python3?\s+<<\s*['"]?(\w+)['"]?\n(.*)\1\s*$""",
            command, re.DOTALL,
        )
        if heredoc_m:
            return self._run_python_subprocess(heredoc_m.group(2))

        # Reject commands targeting / — permission-error spam crashes a-Shell.
        if re.search(r'(?:^|\s)/+(?:\s|$)', command):
            return (
                "⛔ Commands targeting / are blocked — they produce massive "
                "output (permission-error spam) that crashes a-Shell. "
                "Use a specific directory (e.g. ~/Documents/project) or the "
                "glob/grep tools instead."
            )

        # Reject reading from infinite/fast-output devices.
        if re.search(r'/dev/(urandom|zero|random|full)\b', command):
            return (
                "⛔ Reading from /dev/urandom, /dev/zero, etc. produces "
                "unbounded output that crashes a-Shell."
            )

        # Reject dangerous constructs — quote-aware checks on stripped command
        # Only strip single quotes for $()/backtick detection since $(cmd)
        # inside double quotes is still interpreted by the shell.
        sq_stripped = re.sub(r"'[^']*'", '', command)

        if '$(' in sq_stripped or '`' in sq_stripped:
            return (
                "⛔ Command substitution ($() or ``) is unsafe in a-Shell — "
                "it requires running a sub-command concurrently. Run the inner "
                "command separately first, then use its result, or use python3 -c."
            )

        # Strip both quote types for remaining checks
        stripped = re.sub(r"'[^']*'", '', re.sub(r'"[^"]*"', '', command))

        if '<(' in stripped:
            return (
                "⛔ Process substitution (<()) requires concurrent execution "
                "and is unsafe in a-Shell. Use a temp file or python3 -c instead."
            )

        if '<<<' in stripped:
            return (
                "⛔ Here-strings (<<<) are a bashism not supported by dash. "
                "Use echo plus a temp file, or python3 -c instead."
            )

        stripped_r = stripped.rstrip()
        if stripped_r.endswith('&') and not stripped_r.endswith('&&'):
            return (
                "⛔ Background execution (&) is unsafe in a-Shell — builtins "
                "cannot run concurrently. Run commands in the foreground instead."
            )

        # Reject pipes — concurrent builtins hang a-Shell
        # (but allow || which is a chain operator, not a pipe)
        pipe_stripped = re.sub(r'\|\|', '  ', stripped)
        if '|' in pipe_stripped:
            return (
                "⛔ Piped commands (|) are unsafe in a-Shell — they require "
                "concurrent builtins which will hang. Tool output is auto-truncated, "
                "so no need to pipe to head/tail (just run the command). "
                "For other pipes, run each command separately or use built-in tools "
                "(grep, glob, read_file)."
            )

        # Split &&, ||, and ; chains — run each part sequentially
        if '&&' in stripped or '||' in stripped or ';' in stripped:
            return self._run_command_chain(command)

        return self._run_single_command(command)

    def _run_single_command(self, command: str, cwd: str | None = None) -> str:
        """Run a single command with no pipes or chains."""
        needs_shell = bool(re.search(r'[<>]', command))
        if needs_shell:
            return self._run_subprocess(command, shell=True, cwd=cwd)
        try:
            args = shlex.split(command)
        except ValueError:
            return self._run_subprocess(command, shell=True, cwd=cwd)
        args = [self._expand_home(a) for a in args]
        return self._run_subprocess(args, shell=False, cwd=cwd)

    def _run_command_chain(self, command: str) -> str:
        """Split &&, ||, and ; chains and run each command sequentially."""
        tokens = self._tokenize_chain(command)
        results = []
        cwd = None
        last_failed = False
        for op, cmd in tokens:
            cmd = cmd.strip()
            if not cmd:
                continue
            cd_m = re.match(r'^cd\s+(.+)$', cmd)
            if cd_m:
                target = self._expand_home(cd_m.group(1).strip())
                cwd = os.path.abspath(os.path.join(cwd or '.', target))
                try:
                    os.listdir(cwd)
                    results.append(f"(cd {cwd})")
                    last_failed = False
                except OSError as e:
                    results.append(f"(cd failed: {e})")
                    last_failed = True
                    if op == '&&':
                        break
                continue
            result = self._run_single_command(cmd, cwd=cwd)
            last_failed = 'timed out' in result or result.startswith('Error')
            results.append(f"$ {cmd}\n{result}")
            if op == '&&' and last_failed:
                break
            if op == '||' and not last_failed:
                break
        return "\n\n".join(results) if results else "(no commands to run)"

    @staticmethod
    def _tokenize_chain(command: str) -> list[tuple[str, str]]:
        """Split command on &&, ||, and ; (quote-aware). Returns [(op, cmd), ...]."""
        tokens = []
        buf: list[str] = []
        in_sq = False
        in_dq = False
        i = 0
        while i < len(command):
            c = command[i]
            if in_sq:
                buf.append(c)
                if c == "'":
                    in_sq = False
            elif in_dq:
                buf.append(c)
                if c == '"':
                    in_dq = False
            elif c == "'":
                buf.append(c)
                in_sq = True
            elif c == '"':
                buf.append(c)
                in_dq = True
            elif c == '&' and i + 1 < len(command) and command[i + 1] == '&':
                tokens.append(('&&', ''.join(buf)))
                buf = []
                i += 1
            elif c == '|' and i + 1 < len(command) and command[i + 1] == '|':
                tokens.append(('||', ''.join(buf)))
                buf = []
                i += 1
            elif c == ';':
                tokens.append((';', ''.join(buf)))
                buf = []
            else:
                buf.append(c)
            i += 1
        if buf:
            tokens.append(('', ''.join(buf)))
        return tokens

    def _run_python_subprocess(self, code: str) -> str:
        """Execute python3 -c code in a subprocess, capture stdout+stderr."""
        debug("tool: running python in subprocess (%d chars)", len(code))
        try:
            proc = subprocess.Popen(
                ["python3", "-c", code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                return f"(python3 timed out after 10s)\n{stdout}\n{stderr}"
            output = (stdout + stderr).strip() or "(no output)"
            return truncate_result(output, self._max_output)
        except Exception as e:
            return f"Error running python3: {type(e).__name__}: {e}"

    def _run_subprocess(self, args, shell: bool = False, cwd: str | None = None) -> str:
        debug("tool: _run_subprocess shell=%s args=%s", shell, args if isinstance(args, list) else args[:120])
        executable = "dash" if shell else None
        proc = subprocess.Popen(
            args,
            shell=shell,
            executable=executable,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            cwd=cwd,
        )
        try:
            stdout, _ = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
            return f"(command timed out after 10s)\n{stdout}"
        return truncate_result(stdout or "(no output)", self._max_output)

    def _edit_file(self, inputs: dict) -> str:
        path = self._expand_home(inputs["path"])
        if self._is_ignored(path) or self._SENSITIVE_PATTERNS.search(path):
            return f"⛔ Blocked edit to {path}: file is gitignored or looks like a secrets file. Use run_shell to override if intentional."
        old = inputs["old_string"]
        new = inputs["new_string"]
        replace_all = inputs.get("replace_all", False)

        with open(path) as f:
            content = f.read()

        count = content.count(old)
        if count == 0:
            head = "\n".join(content.splitlines()[:20])
            return f"old_string not found in {path}. First 20 lines:\n{head}"
        if count > 1 and not replace_all:
            return f"old_string found {count} times in {path}. Include more surrounding context to make it unique, or set replace_all to true."

        updated = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        with open(path, "w") as f:
            f.write(updated)

        old_lines = old.splitlines(keepends=True)
        new_lines = new.splitlines(keepends=True)
        diff = "".join(difflib.unified_diff(old_lines, new_lines, lineterm="", n=3))
        return f"Edited {path} ({count} replacement{'s' if count > 1 else ''}):\n{diff}"

    def _list_dir(self, inputs: dict) -> str:
        path = self._expand_home(inputs["path"])
        entries = sorted(os.listdir(path))
        visible = [e for e in entries if not self._is_ignored(os.path.join(path, e))]
        return "\n".join(visible)

    def _grep(self, inputs: dict) -> str:
        import fnmatch as _fnmatch

        pattern = inputs["pattern"]
        path    = self._expand_home(inputs["path"])
        include = inputs.get("include", "*")

        regex = re.compile(pattern)

        MAX_GREP_FILES = 500  # don't scan more than this many files

        if os.path.isfile(path):
            files = [path]
        elif os.path.isdir(path):
            files = []
            for root, dirs, filenames in os.walk(path):
                for fn in filenames:
                    if _fnmatch.fnmatch(fn, include):
                        full = os.path.join(root, fn)
                        if not self._is_ignored(full):
                            files.append(full)
                            if len(files) >= MAX_GREP_FILES:
                                break
                if len(files) >= MAX_GREP_FILES:
                    break
        else:
            return f"Path not found: {path}"

        results: list[str] = []
        MAX_GREP_FILE_SIZE = 1_000_000  # skip files larger than 1 MB
        for filepath in files:
            try:
                if os.path.getsize(filepath) > MAX_GREP_FILE_SIZE:
                    continue
                with open(filepath, "rb") as f:
                    chunk = f.read(8192)
                    if b"\x00" in chunk:
                        continue  # skip binary files
                with open(filepath, errors="replace") as f:
                    for lineno, line in enumerate(f, 1):
                        if regex.search(line):
                            results.append(f"{filepath}:{lineno}:{line.rstrip()}")
            except (PermissionError, OSError):
                continue

        if not results:
            return "No matches found."

        text = "\n".join(results)
        return truncate_result(text, self._max_output)

    def _glob(self, inputs: dict) -> str:
        import glob as _glob

        pattern = self._expand_home(inputs["pattern"])
        base    = self._expand_home(inputs.get("path", "."))

        if not os.path.isdir(base):
            return f"Directory not found: {base}"

        full_pattern = os.path.join(base, pattern)
        recursive = "**" in pattern
        MAX_GLOB_RESULTS = 1000
        if recursive:
            # For recursive globs, use an iterator with a cap to avoid
            # materialising a huge list from deep system directories.
            matches = []
            for p in _glob.iglob(full_pattern, recursive=True):
                if not self._is_ignored(p):
                    matches.append(p)
                if len(matches) >= MAX_GLOB_RESULTS:
                    break
            matches.sort()
        else:
            matches = sorted(_glob.glob(full_pattern, recursive=False))
            matches = [m for m in matches if not self._is_ignored(m)]
            if len(matches) > MAX_GLOB_RESULTS:
                matches = matches[:MAX_GLOB_RESULTS]

        if not matches:
            return f"No matches for {pattern} in {base}/"

        if base == ".":
            display = [m.lstrip("./") if m.startswith("./") else m for m in matches]
        else:
            prefix = base.rstrip("/") + "/"
            display = [m[len(prefix):] if m.startswith(prefix) else m for m in matches]

        text = "\n".join(display)
        header = f"{len(matches)} match{'es' if len(matches) != 1 else ''}:\n"
        return header + truncate_result(text, self._max_output)

    def _web_search(self, inputs: dict) -> str:
        import urllib.request
        import urllib.parse

        query = inputs["query"]
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15"
        })
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                html = resp.read().decode(errors="replace")
        except Exception as e:
            return f"Search failed: {e}"

        from bs4 import BeautifulSoup

        results: list[str] = []
        soup = BeautifulSoup(html, "html.parser")

        for result_div in soup.find_all(class_="web-result"):
            title_tag = result_div.find("a", class_="result__a")
            snippet_tag = result_div.find("a", class_="result__snippet")

            if not title_tag:
                continue

            raw_url = title_tag.get("href", "")
            title = title_tag.get_text(strip=True)
            snippet = snippet_tag.get_text(strip=True) if snippet_tag else ""

            if "//duckduckgo.com/l/?uddg=" in raw_url:
                try:
                    uddg = raw_url.split("uddg=")[1].split("&")[0]
                    real_url = urllib.parse.unquote(uddg)
                except Exception:
                    real_url = raw_url
            else:
                real_url = raw_url

            if title:
                results.append(f"### [{title}]({real_url})\n{snippet}")

        if not results:
            return f"No results found for: {query}"

        output = f"## Search results for: {query}\n\n" + "\n\n".join(results[:10])
        return truncate_result(output, self._max_output)

    def _read_url(self, inputs: dict) -> str:
        import urllib.request
        from bs4 import BeautifulSoup
        from markdownify import MarkdownConverter

        url = inputs["url"]
        max_length = inputs.get("max_length", 10000)

        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X) AppleWebKit/605.1.15"
        })
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                content_type = resp.getheader("Content-Type", "")
                MAX_URL_DOWNLOAD = 2_000_000  # 2 MB cap
                raw = resp.read(MAX_URL_DOWNLOAD)

                charset = "utf-8"
                if "charset=" in content_type:
                    charset = content_type.split("charset=")[-1].split(";")[0].strip()
                else:
                    head = raw[:2048].decode("ascii", errors="replace")
                    mc = re.search(r'charset=["\']?([^"\';\s>]+)', head, re.I)
                    if mc:
                        charset = mc.group(1)

                html = raw.decode(charset, errors="replace")
        except Exception as e:
            return f"Failed to fetch {url}: {e}"

        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup.find_all(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
                tag.decompose()
            main = (
                soup.find("main")
                or soup.find("article")
                or soup.find(class_=re.compile(r"content|main|article|body", re.I))
                or soup.find(id=re.compile(r"content|main|article|body", re.I))
                or soup
            )
            text = MarkdownConverter(
                heading_style="ATX",
                bullets="-",
                code_language="",
            ).convert_soup(main)
        except Exception as e:
            return f"Failed to convert HTML: {e}"

        text = re.sub(r'\n{3,}', '\n\n', text).strip()

        if len(text) > max_length:
            text = text[:max_length] + f"\n\n… ✂ truncated at {max_length:,} chars ({len(text):,} total) …"

        return text


_tool_runner = ToolRunner()


# ---------------------------------------------------------------------------
# Context compaction
# ---------------------------------------------------------------------------

def _msg_tokens(msg: dict) -> int:
    return len(json.dumps(msg)) // 4


def estimate_tokens(messages: list) -> int:
    return sum(_msg_tokens(m) for m in messages)


def compact_history(history: list, context_window: int) -> tuple:
    used      = estimate_tokens(history)
    threshold = int(context_window * COMPACT_THRESHOLD)
    debug("compact: used=%d threshold=%d messages=%d", used, threshold, len(history))
    if used < threshold:
        return history, False

    tail_budget = int(context_window * 0.20)
    tail_tokens = 0
    split = len(history)
    for i in range(len(history) - 1, -1, -1):
        tail_tokens += _msg_tokens(history[i])
        if tail_tokens >= tail_budget:
            split = i
            break
    else:
        split = 0

    split = min(split, len(history) - 2)
    split = max(split, 1)

    tail = history[split:]
    head = history[:split]
    if not head:
        return history, False

    max_compact_input_tokens = 4_000
    compact_input = json.dumps(head)
    compact_input_chars = max_compact_input_tokens * 4
    if len(compact_input) > compact_input_chars:
        compact_input = compact_input[:compact_input_chars] + "\n… (truncated)"

    summary_prompt = (
        "Summarize the following conversation and coding work done so far. "
        "Be concise but preserve: key decisions, file names, code written, "
        "errors encountered, and current task state.\n\n"
        + compact_input
    )
    display.spin("Compacting context…")
    try:
        resp    = call_api_simple([{"role": "user", "content": summary_prompt}], max_tokens=1024)
        summary = resp["choices"][0]["message"]["content"]
    except Exception as e:
        summary = f"(summary unavailable: {e})"
    finally:
        display.stop_spin()

    debug("compact: summarized %d head messages → %d chars", len(head), len(summary))

    return [
        {"role": "user",      "content": f"[Session summary]\n{summary}"},
        {"role": "assistant", "content": "Understood. Continuing from that context."},
        *tail,
    ], True


# ---------------------------------------------------------------------------
# Language map (for tool output display)
# ---------------------------------------------------------------------------

LANG_MAP = {
    "py": "python",   "mjs": "javascript", "cjs": "javascript",
    "ts": "typescript", "mts": "typescript", "cts": "typescript",
    "tsx": "typescript", "jsx": "javascript",
    "bash": "sh",     "shell": "sh",     "zsh": "sh",
    "md": "markdown",
    "rs": "rust",     "cs": "csharp",    "gd": "gdscript",
    "rb": "ruby",     "kt": "kotlin",    "pl": "perl",
    "ex": "elixir",   "exs": "elixir",   "erl": "erlang",
    "hs": "haskell",  "clj": "clojure",  "cfg": "ini",
    "conf": "ini",    "gql": "graphql",  "tf": "hcl",
    "": "text",
}


# ---------------------------------------------------------------------------
# Display: single writer thread owning all stdout output
# ---------------------------------------------------------------------------

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

_cancelled = threading.Event()


def _cancel_requested() -> bool:
    """Non-blocking check whether user typed 'c' + Enter on stdin.
    Uses os.read() on the raw fd which returns immediately (or raises
    BlockingIOError) unlike sys.stdin.read() which can block forever
    in a-Shell when select() falsely reports stdin as ready."""
    try:
        import fcntl
        fd = sys.stdin.fileno()
        old = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, old | os.O_NONBLOCK)
        try:
            buf = b""
            while True:
                ch = os.read(fd, 1)
                if not ch:
                    break
                buf += ch
                if ch in (b'\n', b'\r'):
                    break
            line = buf.decode(errors='ignore').strip().lower()
            return line in ('c', 'cancel', 'stop')
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, old)
    except Exception:
        return False


def _drain_stdin() -> None:
    """Discard any pending stdin left over from a failed cancel attempt."""
    try:
        import fcntl
        fd = sys.stdin.fileno()
        old = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, old | os.O_NONBLOCK)
        try:
            while os.read(fd, 1024):
                pass
        except (BlockingIOError, OSError):
            pass
        finally:
            fcntl.fcntl(fd, fcntl.F_SETFL, old)
    except Exception:
        pass


class _Display:
    """Single background thread that owns all writes to stdout.
    Serialises spinner animation, streaming text, and Rich rendering
    so they never stomp each other."""

    _SPIN     = "spin"
    _STOP     = "stop"
    _STREAM   = "stream"
    _RENDER   = "render"
    _PRINT    = "print"
    _RULE     = "rule"

    def __init__(self):
        self._queue: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def spin(self, message: str) -> None:
        self._queue.put((self._SPIN, message))

    def stop_spin(self) -> None:
        self._queue.put((self._STOP, None))

    def stream(self, text: str, *, style: str = "") -> None:
        self._queue.put((self._STREAM, (text, style)))

    def render(self, renderable) -> None:
        done = threading.Event()
        self._queue.put((self._RENDER, (renderable, done)))
        done.wait(timeout=60)

    def print(self, *args, **kwargs) -> None:
        done = threading.Event()
        self._queue.put((self._PRINT, (args, kwargs, done)))
        done.wait(timeout=10)

    def rule(self) -> None:
        done = threading.Event()
        self._queue.put((self._RULE, done))
        done.wait(timeout=5)

    def _clear_spinner(self) -> None:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()

    def _write_spinner_frame(self, msg: str, start: float, idx: int) -> None:
        elapsed = time.monotonic() - start
        frame = _SPINNER_FRAMES[idx % len(_SPINNER_FRAMES)]
        sys.stdout.write(f"\r\033[2m {frame} {msg} ({elapsed:.1f}s)\033[0m")
        sys.stdout.flush()

    def _run(self) -> None:
        spinning = False
        spin_msg = ""
        spin_start = 0.0
        spin_idx = 0
        buf_console = Console(
            file=io.StringIO(), force_terminal=True, legacy_windows=False
        )

        while True:
            try:
                timeout = 0.15 if spinning else 1.0
                try:
                    cmd, data = self._queue.get(timeout=timeout)
                except queue.Empty:
                    if spinning:
                        self._write_spinner_frame(spin_msg, spin_start, spin_idx)
                        spin_idx += 1
                    continue

                if cmd == self._SPIN:
                    spin_msg = data
                    spin_start = time.monotonic()
                    spin_idx = 0
                    spinning = True

                elif cmd == self._STOP:
                    if spinning:
                        self._clear_spinner()
                        spinning = False

                elif cmd == self._STREAM:
                    text, style = data
                    if spinning:
                        self._clear_spinner()
                        spinning = False
                    if style == "dim":
                        sys.stdout.write(f"\033[2m{text}\033[0m")
                    else:
                        sys.stdout.write(text)
                    sys.stdout.flush()

                elif cmd == self._RENDER:
                    renderable, done_event = data
                    if spinning:
                        self._clear_spinner()
                        spinning = False
                    try:
                        buf_console.file = io.StringIO()
                        buf_console.print(renderable)
                        sys.stdout.write(buf_console.file.getvalue())
                        sys.stdout.flush()
                    except Exception:
                        pass
                    finally:
                        done_event.set()

                elif cmd == self._PRINT:
                    args, kwargs, done_event = data
                    if spinning:
                        self._clear_spinner()
                        spinning = False
                    try:
                        buf_console.file = io.StringIO()
                        buf_console.print(*args, **kwargs)
                        sys.stdout.write(buf_console.file.getvalue())
                        sys.stdout.flush()
                    except Exception:
                        pass
                    finally:
                        done_event.set()

                elif cmd == self._RULE:
                    done_event = data
                    if spinning:
                        self._clear_spinner()
                        spinning = False
                    try:
                        buf_console.file = io.StringIO()
                        buf_console.rule()
                        sys.stdout.write(buf_console.file.getvalue())
                        sys.stdout.flush()
                    except Exception:
                        pass
                    finally:
                        done_event.set()

            except Exception:
                pass


_TOOL_LABEL = {
    "run_shell": lambda i: f"Running: {i.get('command', '')[:60]}",
    "web_search": lambda i: f"Searching: {i.get('query', '')[:60]}",
    "read_url": lambda i: f"Fetching: {i.get('url', '')[:60]}",
    "read_file": lambda i: f"Reading: {i.get('path', '')}",
    "write_file": lambda i: f"Writing: {i.get('path', '')}",
    "edit_file": lambda i: f"Editing: {i.get('path', '')}",
    "grep": lambda i: f"Searching: {i.get('pattern', '')[:40]}",
    "glob": lambda i: f"Globbing: {i.get('pattern', '')}",
    "list_dir": lambda i: f"Listing: {i.get('path', '')}",
}

display = _Display()


# ---------------------------------------------------------------------------
# Agent turn
# ---------------------------------------------------------------------------

def _context_str(history: list, context_window: int) -> str:
    used = estimate_tokens(history)
    pct = int(used / context_window * 100) if context_window else 0
    return f"{used // 1000}k/{context_window // 1000}k ({pct}%)"


def _run_agent(prompt: str, history: list, model_info: dict, system_prompt: str) -> None:
    _cancelled.clear()
    if not history:
        history.append({"role": "system", "content": system_prompt})

    history.append({"role": "user", "content": prompt})

    _run_agent_loop(prompt, history, model_info)


def _run_agent_loop(prompt: str, history: list, model_info: dict) -> None:
    accumulated: list[str] = []
    stream_retries = 0
    max_retries = 3
    max_iterations = 50
    iteration = 0

    while True:
        iteration += 1
        if iteration > max_iterations:
            display.print("[dim]⚠ Max iterations reached — stopping to prevent runaway loop.[/dim]\n")
            break
        history, compacted = compact_history(
            history, model_info["context_window"]
        )
        if compacted:
            ctx = _context_str(history, model_info["context_window"])
            display.print(f"[dim]⚡ Context compacted – older messages summarized.  Context: {ctx}[/dim]\n")

        used_tokens = estimate_tokens(history)
        turn_max = adjust_max_tokens_for_thinking(
            model_info["max_tokens"], model_info["max_tokens"],
            model_info["context_window"], used_tokens
        )

        content_buf: list[str] = []
        thinking_buf: list[str] = []
        tool_calls: list = []
        finish_reason = None
        q: queue.Queue = queue.Queue()
        need_retry = False

        _reset_connection()

        def _produce():
            for ev in iter_sse_chunks(history, turn_max):
                q.put(ev)
            q.put(None)

        t = threading.Thread(target=_produce, daemon=True)
        t.start()

        thinking_printed_header = False
        thinking_transitioned = False
        got_first_chunk = False
        last_cancel_check = time.monotonic()

        # Show a spinner while waiting for the first stream event
        display.spin("Waiting for response… (c + Enter to cancel)")

        try:
            while True:
                try:
                    event = q.get(timeout=0.25)
                except queue.Empty:
                    if _cancelled.is_set() or _cancel_requested():
                        _cancelled.set()
                        break
                    continue

                # Check cancel every 0.5s even during active streaming
                now = time.monotonic()
                if now - last_cancel_check >= 0.5:
                    if _cancel_requested():
                        _cancelled.set()
                        break
                    last_cancel_check = now
                if event is None:
                    break

                etype = event["type"]

                if etype == "thinking":
                    if not got_first_chunk:
                        display.stop_spin()
                        got_first_chunk = True
                    if not thinking_printed_header:
                        display.stream("◦ Thinking… ", style="dim")
                        thinking_printed_header = True
                    thinking_buf.append(event["text"])
                    display.stream(event["text"], style="dim")

                elif etype == "text":
                    if not got_first_chunk:
                        display.stop_spin()
                        got_first_chunk = True
                    if thinking_printed_header:
                        display.stream("\n\n", style="dim")
                        thinking_printed_header = False
                    if not thinking_transitioned:
                        display.spin("Generating response…")
                        thinking_transitioned = True
                    content_buf.append(event["text"])

                elif etype == "tool_calls":
                    tool_calls = event["tool_calls"]

                elif etype == "finish":
                    finish_reason = event["reason"]

                elif etype == "error":
                    display.stop_spin()
                    stream_retries += 1
                    if stream_retries <= max_retries:
                        err_msg = event["text"][:200]
                        display.stream(f"\n⚠ Error ({stream_retries}/{max_retries}): {err_msg}\n")
                        while True:
                            leftover = q.get()
                            if leftover is None:
                                break
                        display.spin(f"Retrying in {stream_retries}s…")
                        t.join(timeout=10)
                        time.sleep(stream_retries)
                        display.stop_spin()
                        need_retry = True
                        break
                    raise RuntimeError(event["text"])
        finally:
            display.stop_spin()
            if _cancelled.is_set():
                _reset_connection()

        display.spin("Finishing stream…")
        t.join(timeout=0.2 if _cancelled.is_set() else 5)
        display.stop_spin()

        if _cancelled.is_set():
            display.print("\n[dim]Cancelled.[/dim]\n")
            return

        if need_retry:
            continue

        stream_retries = 0
        full_content = "".join(content_buf)

        # Ensure we're on a new line after streaming
        display.stream("\n")

        if finish_reason in ("length", "interrupted"):
            accumulated.append(full_content)
            if full_content.strip():
                display.render(Markdown(full_content))
            history.append({"role": "assistant", "content": full_content})
            history.append({"role": "user", "content": "Your response was cut off mid-stream. Continue exactly where you left off."})
            label = "interrupted" if finish_reason == "interrupted" else "cut off"
            display.print(f"\n[dim]⚠ Response {label} – continuing…[/dim]\n")
            continue

        if tool_calls:
            accumulated.append(full_content)
            if full_content.strip():
                display.render(Markdown(full_content))

            assistant_msg = {
                "role":       "assistant",
                "content":    full_content or None,
                "tool_calls": tool_calls,
            }
            history.append(assistant_msg)
            accumulated.clear()

            for tc in tool_calls:
                if _cancel_requested():
                    _cancelled.set()
                if _cancelled.is_set():
                    display.print("\n[dim]Cancelled.[/dim]\n")
                    return
                name = tc["function"]["name"]
                try:
                    inputs = json.loads(tc["function"]["arguments"])
                except json.JSONDecodeError:
                    inputs = {"_raw": tc["function"]["arguments"]}

                _tool_label = _TOOL_LABEL.get(name, lambda i: name)
                if SHELL_TOOL_CONFIRMATION and name == "run_shell":
                    display.stop_spin()
                    display.print(f"[yellow]▸ Shell command: {inputs.get('command', '')}[/yellow]")
                    try:
                        resp = input("  Run? [Y/n] ").strip().lower()
                    except EOFError:
                        resp = ""
                    if resp in ("n", "no"):
                        _cancelled.set()
                        result = "⛔ Command cancelled by user."
                    else:
                        display.spin(_tool_label(inputs))
                        try:
                            result = _tool_runner.run(name, inputs)
                        finally:
                            display.stop_spin()
                else:
                    display.spin(_tool_label(inputs))
                    try:
                        result = _tool_runner.run(name, inputs)
                    finally:
                        display.stop_spin()

                if name in ("web_search", "read_url"):
                    display_lang = "markdown"
                else:
                    path = inputs.get("path", "")
                    ext = path.rsplit(".", 1)[-1].lower() if "." in path else ""
                    display_lang = LANG_MAP.get(ext, ext if ext else "text")

                lines = result.splitlines()
                if len(lines) > 60:
                    half = 30
                    display_text = (
                        f"```{display_lang}\n"
                        + "\n".join(lines[:half])
                        + f"\n```\n\n… ✂ {len(lines) - 60} lines omitted ({len(lines)} total) …\n\n```{display_lang}\n"
                        + "\n".join(lines[-half:])
                        + "\n```"
                    )
                else:
                    display_text = f"```{display_lang}\n{result}\n```"

                tool_md = f"**{name}** → `{json.dumps(inputs)}`\n\n{display_text}"
                display.render(Panel(Markdown(tool_md), title=name, border_style="yellow"))

                history.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      truncate_result(result),
                })

            display.print()
            continue

        accumulated.append(full_content)
        final = "".join(accumulated)
        accumulated.clear()
        if final.strip():
            display.render(Markdown(final))
            history.append({"role": "assistant", "content": final})
        display.rule()
        break


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not API_KEY:
        print("Error: API_KEY not set.")
        print("  export API_KEY=your-key-here")
        print("  export API_BASE=https://openrouter.ai/api/v1  # optional")
        raise SystemExit(1)

    input_console = Console()
    history: list = []
    system_prompt = load_system_prompt()
    model_info = DEFAULT_MODEL_INFO

    display.spin("Fetching model info…")
    try:
        model_info = fetch_model_info(MODEL)
    except Exception:
        pass
    finally:
        display.stop_spin()
    if CONTEXT_WINDOW:
        model_info["context_window"] = CONTEXT_WINDOW
    if MAX_TOKENS:
        model_info["max_tokens"] = MAX_TOKENS

    ctx_k = model_info["context_window"] // 1000
    tok_k = model_info["max_tokens"] // 1000
    display.print(f"[bold]a-agent[/bold]")
    display.print(f"  Model:    {MODEL}")
    display.print(f"  Provider: {API_HOST}")
    display.print(f"  Context:  {ctx_k}k  │  Max output: {tok_k}k")
    display.print("[dim]Type your prompt, or 'quit' to exit. c + Enter to cancel.[/dim]\n")

    while True:
        try:
            ctx = _context_str(history, model_info["context_window"])
            prompt = input_console.input(f"[bold cyan]▸[/] [dim]{ctx}[/dim] ").strip()
        except KeyboardInterrupt:
            display.print("\n[dim]Ctrl+C — type 'quit' to exit[/dim]")
            continue
        except EOFError:
            display.print("\nGoodbye!")
            break

        # Filter out terminal escape sequences (e.g. Kitty keyboard protocol
        # sends sequences like [99;5u for Ctrl+C when in that mode).
        prompt = re.sub(r'\x1b\[[^a-zA-Z]*[a-zA-Z]', '', prompt)

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            display.print("Goodbye!")
            break

        _cancelled.clear()
        try:
            _run_agent(prompt, history, model_info, system_prompt)
        except Exception as e:
            display.print(f"\n[red]Error: {e}[/red]\n")
            debug("agent error: %s", e)
        _drain_stdin()


if __name__ == "__main__":
    main()
