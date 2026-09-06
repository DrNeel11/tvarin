"""A minimal Cline-style coding agent driven entirely by a local Ollama
model, built to stress-test Ramer under a realistic agentic workload
(file reads/writes, shell commands, multi-step iteration) instead of a
synthetic one-shot prompt -- the shape of traffic a real coding agent
sends, not a hand-picked benchmark prompt.

Uses Ollama's native tool-calling (/api/chat + tools), the same
mechanism Cline/Continue use under the hood -- not a hand-rolled
text-parsing scheme, since Qwen2.5-instruct models are actually trained
for this ("tools" in their reported capabilities).

If ramer_local.py has already probed the model on this machine, its
cached optimal num_gpu is reused automatically. If not, this deliberately
does NOT trigger a probe itself -- run `ramer_local.py probe <model>`
first if you want Ramer's placement rather than Ollama's auto default.

All file/command tools are confined to one working directory -- by
default the throwaway agent_sandbox/ next to this repo, or --workdir if
given (e.g. a real project open in VS Code). This drives an untested
local model with real file-write and shell-exec tools: point --workdir
at something you're genuinely fine with a model editing, ideally a
directory already under version control so a bad edit is a `git diff`
away from undoing.

Usage:
    python scripts/mini_agent.py <model> "<task description>"
    python scripts/mini_agent.py qwen2.5:14b-instruct "Create hello.py that prints Hello World, then run it."
    python scripts/mini_agent.py --workdir . qwen2.5:14b-instruct "Add a .gitignore entry for *.log files"
"""

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # Windows' default console codepage can't print a lot of what models write (emoji, smart quotes, box-drawing)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ramer_local import OLLAMA_HOST, cache_key, model_digest, load_cache, gpu_vram_mib, gpu_name  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
SANDBOX = REPO_ROOT / "agent_sandbox"  # default; overridden by --workdir in main() before any tool runs

MAX_STEPS = 15
COMMAND_TIMEOUT_S = 30

SYSTEM_PROMPT = """You are a careful coding agent working inside a single project directory.
Use the available tools to read and write files and run commands to accomplish the user's task.
Work step by step: inspect before you assume, write code, run it, read any errors, and fix them.
When the task is fully done, reply with plain text (no tool call) summarizing what you did.
Keep file paths relative and simple -- everything you touch lives in one project directory.
Always respond in English, regardless of what language any file content you read happens to contain."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative file path"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a file in the project directory with the given content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "Full file content to write"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files in a directory in the project directory.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Relative directory path, '.' for root"}},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command inside the project directory and return its output.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string", "description": "Shell command to run"}},
                "required": ["command"],
            },
        },
    },
]


def _resolve(path: str) -> Path:
    """Keep every tool confined to SANDBOX -- refuse anything that escapes it."""
    p = (SANDBOX / path).resolve()
    if SANDBOX not in p.parents and p != SANDBOX:
        raise ValueError(f"path '{path}' escapes the sandbox")
    return p


def tool_read_file(path: str) -> str:
    p = _resolve(path)
    if not p.exists():
        return f"error: {path} does not exist"
    return p.read_text(errors="replace")


def tool_write_file(path: str, content: str) -> str:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


def tool_list_dir(path: str = ".") -> str:
    p = _resolve(path)
    if not p.exists():
        return f"error: {path} does not exist"
    entries = sorted(f"{e.name}/" if e.is_dir() else e.name for e in p.iterdir())
    return "\n".join(entries) if entries else "(empty)"


def tool_run_command(command: str) -> str:
    try:
        proc = subprocess.run(
            command, shell=True, cwd=SANDBOX, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S,
        )
        out = f"exit code: {proc.returncode}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        return out[:4000]
    except subprocess.TimeoutExpired:
        return f"error: command timed out after {COMMAND_TIMEOUT_S}s"


TOOL_FUNCS = {
    "read_file": tool_read_file,
    "write_file": tool_write_file,
    "list_dir": tool_list_dir,
    "run_command": tool_run_command,
}


def ramers_num_gpu(model: str) -> object:
    """Reuse ramer_local.py's cache if this model has already been probed
    on this machine -- don't trigger a probe here, that's a separate,
    explicit step."""
    digest = model_digest(model)
    key = cache_key(model, digest, 40)  # 40 matches ramer_local.py's own default --num-predict
    cache = load_cache()
    if key in cache:
        return cache[key]["recommended"]["num_gpu"]
    return "auto"


def chat(model: str, messages: list, num_gpu) -> dict:
    options = {} if num_gpu == "auto" else {"num_gpu": num_gpu}
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps({"model": model, "messages": messages, "tools": TOOLS, "stream": False, "options": options}).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read())
    body["_wall_s"] = round(time.time() - t0, 2)
    return body


SESSION_LOG_PATH = REPO_ROOT / "mini_agent_sessions.jsonl"


def main():
    global SANDBOX
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model")
    parser.add_argument("task", nargs="+")
    parser.add_argument("--num-gpu", type=str, default=None,
                         help="override the placement instead of using ramer_local.py's cache -- "
                              "an int, or 'auto' for Ollama's own default. For A/B comparing placements on the identical task.")
    parser.add_argument("--workdir", type=str, default=None,
                         help="real project directory to operate in, e.g. your VS Code workspace folder "
                              "(pass '.' to use the current directory). Defaults to the throwaway agent_sandbox/ "
                              "next to this repo. All file/command tools stay confined to this directory.")
    cli_args = parser.parse_args()
    model = cli_args.model
    task = " ".join(cli_args.task)

    if cli_args.workdir is not None:
        SANDBOX = Path(cli_args.workdir).resolve()
        if not SANDBOX.exists():
            print(f"[agent] error: --workdir '{cli_args.workdir}' does not exist")
            sys.exit(1)
    else:
        SANDBOX.mkdir(exist_ok=True)

    if cli_args.num_gpu is not None:
        num_gpu = "auto" if cli_args.num_gpu == "auto" else int(cli_args.num_gpu)
        placement_label = f"forced --num-gpu={num_gpu}"
    else:
        num_gpu = ramers_num_gpu(model)
        placement_label = "Ramer-optimized" if num_gpu != "auto" else "Ollama auto -- not probed yet, run ramer_local.py probe first for Ramer placement"

    print(f"[agent] model={model}  num_gpu={num_gpu} ({placement_label})")
    print(f"[agent] sandbox: {SANDBOX}")
    print(f"[agent] task: {task}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    turns = []  # per-model-call stats, to see how throughput/context trend over a long session

    def record_turn(resp: dict):
        eval_count = resp.get("eval_count") or 0
        eval_duration_ns = resp.get("eval_duration") or 1
        turns.append(dict(
            turn=len(turns) + 1,
            prompt_eval_count=resp.get("prompt_eval_count", 0),
            eval_count=eval_count,
            tok_s=round(eval_count / (eval_duration_ns / 1e9), 2) if eval_count else 0.0,
            wall_s=resp.get("_wall_s"),
        ))

    def finish(final_content: str, steps_taken: int):
        session_wall_s = round(sum(t["wall_s"] for t in turns), 2)
        print(f"\n[agent] done after {steps_taken} step(s), {len(turns)} model call(s), {session_wall_s}s total model time:\n")
        print(final_content.strip())
        print(f"\n[agent] per-call trend (prompt size grows as the session goes on):")
        print(f"  {'call':>4} {'prompt tok':>10} {'gen tok':>7} {'tok/s':>7} {'wall s':>7}")
        for t in turns:
            print(f"  {t['turn']:>4} {t['prompt_eval_count']:>10} {t['eval_count']:>7} {t['tok_s']:>7} {t['wall_s']:>7}")
        with open(SESSION_LOG_PATH, "a") as f:
            f.write(json.dumps(dict(
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=model, num_gpu=num_gpu, task=task,
                steps=steps_taken, turns=turns, session_wall_s=session_wall_s,
            )) + "\n")

    for step in range(1, MAX_STEPS + 1):
        resp = chat(model, messages, num_gpu)
        record_turn(resp)
        msg = resp.get("message", {})
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or []
        if not tool_calls:
            finish(msg.get("content", ""), step)
            return

        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}
            print(f"[step {step}] {name}({args})")
            func = TOOL_FUNCS.get(name)
            if func is None:
                result = f"error: unknown tool '{name}'"
            else:
                try:
                    result = func(**args)
                except Exception as e:
                    result = f"error: {e}"
            preview = result if len(result) <= 300 else result[:300] + "... (truncated)"
            print(f"          -> {preview}\n")
            messages.append({"role": "tool", "content": result})

    finish("(stopped: no final answer after the step limit -- likely stuck in a loop)", MAX_STEPS)


if __name__ == "__main__":
    main()
