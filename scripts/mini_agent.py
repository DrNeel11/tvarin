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


WRITE_SHRINK_GUARD_RATIO = 0.5  # refuse a same-path overwrite below this fraction of the existing size
WRITE_SHRINK_GUARD_MIN_BYTES = 200  # below this, a small file legitimately being replaced isn't worth flagging


def tool_write_file(path: str, content: str) -> str:
    """write_file always fully overwrites -- there's no patch/diff tool. Seen for
    real on an actual project: asked to *modify* three existing files (add a
    parameter threaded through several functions), a 14B local model instead
    wrote small illustrative stubs over all three, silently discarding real,
    working logic (a keyword-matched rule engine, a live API integration, most
    of a REST API's routes) -- tests went from 29/29 to 5 failing, and nothing
    about the tool result said anything was wrong. A large same-path shrink is
    exactly what that failure looks like from here, so it's refused rather than
    silently applied -- cheap to check, and a legitimate large deletion can
    still happen, just not as an unremarked side effect of an ambiguous
    'modify this file' instruction."""
    p = _resolve(path)
    if p.exists():
        old_size = p.stat().st_size
        new_size = len(content.encode())
        if old_size >= WRITE_SHRINK_GUARD_MIN_BYTES and new_size < old_size * WRITE_SHRINK_GUARD_RATIO:
            return (f"error: refusing to write {path} -- this would shrink it from {old_size} to {new_size} "
                     f"bytes ({100 * new_size / old_size:.0f}% of the original). That looks like an accidental "
                     f"full overwrite of an existing file rather than the intended edit. Read the full current "
                     f"file first and write back the complete content with only the intended change, or write "
                     f"to a different path if a smaller file is genuinely intended.")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"wrote {len(content)} bytes to {path}"


def git_changed_files(cwd: Path) -> list[str] | None:
    """None means "not a git repo" (or git unavailable) -- verification is
    skipped rather than faked, since there's no ground truth to compare
    against without one. A list (possibly empty) means git ran cleanly and
    is the actual state of the working tree, independent of anything the
    model claims to have done."""
    try:
        # "-- ." scopes to cwd's own subtree -- without it, a sandbox that's a
        # subdirectory of a larger repo (the default agent_sandbox/, which
        # lives inside this repo) would report changes anywhere in the whole
        # repo, not just within the sandbox this session actually touched.
        result = subprocess.run(
            ["git", "-C", str(cwd), "status", "--porcelain", "--", "."], capture_output=True, text=True, timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return [line[3:].strip() for line in result.stdout.splitlines() if line.strip()]


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


def ramers_num_gpu(model: str, num_ctx: int) -> object:
    """Reuse ramer_local.py's cache if this model has been probed on this
    machine AT THIS CONTEXT WINDOW -- don't trigger a probe here, that's a
    separate, explicit step. The context match is not optional: a split
    probed at 4096 and reused at 8192 went from ~15 tok/s to 1.47 tok/s
    here, because the doubled KV cache pushed it over the VRAM cliff.
    No entry for this num_ctx means Ollama's auto placement, which does
    account for KV cache size on its own."""
    digest = model_digest(model)
    key = cache_key(model, digest, 40, num_ctx)  # 40 matches ramer_local.py's own default --num-predict
    cache = load_cache()
    if key in cache:
        return cache[key]["recommended"]["num_gpu"]
    return "auto"


def chat(model: str, messages: list, num_gpu, num_ctx: int, tools: list | None = None) -> dict:
    """Streams the response. Found the hard way on a real project: a step
    that reads two source files then writes a full rewritten one can run
    for minutes at ~10 tok/s, and with stream:false that was minutes of
    total silence -- indistinguishable from a hang -- followed by a
    300s urllib timeout killing a perfectly healthy generation. Streaming
    fixes both: text shows up as it's produced, a token counter ticks
    while the model works on a tool call (whose JSON isn't worth
    streaming to a terminal), and the socket timeout now bounds the gap
    *between* tokens, so a real stall is still caught but a long answer
    isn't.

    tools defaults to this script's own TOOLS so the CLI usage below is
    unchanged; the app/ package passes its own tool schemas through here
    instead of duplicating this streaming implementation."""
    options = {"num_ctx": num_ctx}
    if num_gpu != "auto":
        options["num_gpu"] = num_gpu
    req = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=json.dumps({"model": model, "messages": messages, "tools": tools if tools is not None else TOOLS, "stream": True, "options": options}).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    content_parts = []
    tool_calls = []
    final = {}
    n_chunks = 0
    last_progress = 0.0
    showing_progress = False
    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            line = line.strip()
            if not line:
                continue
            chunk = json.loads(line)
            n_chunks += 1
            msg = chunk.get("message", {})
            delta = msg.get("content", "")
            if msg.get("tool_calls"):
                tool_calls.extend(msg["tool_calls"])
            if delta:
                if showing_progress:
                    print("\r" + " " * 60 + "\r", end="", flush=True)
                    showing_progress = False
                print(delta, end="", flush=True)
                content_parts.append(delta)
            elif time.time() - last_progress > 1.0:
                # nothing printable yet (model is mid tool-call) -- keep proving it's alive
                print(f"\r  ... working ({n_chunks} tokens, {time.time() - t0:.0f}s)", end="", flush=True)
                showing_progress = True
                last_progress = time.time()
            if chunk.get("done"):
                final = chunk
                break
    if showing_progress:
        print("\r" + " " * 60 + "\r", end="", flush=True)
    if content_parts:
        print()
    message = {"role": "assistant", "content": "".join(content_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    final["message"] = message
    final["_wall_s"] = round(time.time() - t0, 2)
    return final


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
    parser.add_argument("--num-ctx", type=int, default=8192,
                         help="context window in tokens. Ollama's default of 4096 is too small once real source "
                              "files get read into the conversation -- past it, Ollama silently drops the OLDEST "
                              "content, i.e. the system prompt and your task. Bigger costs VRAM for KV cache.")
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

    num_ctx = cli_args.num_ctx
    if cli_args.num_gpu is not None:
        num_gpu = "auto" if cli_args.num_gpu == "auto" else int(cli_args.num_gpu)
        placement_label = f"forced --num-gpu={num_gpu}"
    else:
        num_gpu = ramers_num_gpu(model, num_ctx)
        if num_gpu != "auto":
            placement_label = f"Ramer-optimized, probed at num_ctx={num_ctx}"
        else:
            placement_label = (f"Ollama auto -- no Ramer placement probed at num_ctx={num_ctx}; run "
                               f"`ramer_local.py probe {model} --num-ctx {num_ctx}` to get one")

    print(f"[agent] model={model}  num_gpu={num_gpu} ({placement_label})  num_ctx={num_ctx}")
    print(f"[agent] sandbox: {SANDBOX}")
    print(f"[agent] task: {task}\n")

    baseline_changed = git_changed_files(SANDBOX)
    if baseline_changed:
        print(f"[agent] warning: {SANDBOX} has {len(baseline_changed)} uncommitted change(s) before this session "
              f"even starts -- verification below can't tell those apart from ones this session makes. "
              f"Commit or stash first for a clean baseline.\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]
    turns = []  # per-model-call stats, to see how throughput/context trend over a long session
    files_written = []  # every path a successful write_file call actually touched this session -- ground truth
                         # to check the model's own claims against, not a substitute for reading the diff yourself

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

    def finish(final_content: str, steps_taken: int, already_streamed: bool = True):
        session_wall_s = round(sum(t["wall_s"] for t in turns), 2)
        print(f"\n[agent] done after {steps_taken} step(s), {len(turns)} model call(s), {session_wall_s}s total model time.")
        if not already_streamed:
            print(f"\n{final_content.strip()}")
        print(f"\n[agent] per-call trend (prompt size grows as the session goes on):")
        print(f"  {'call':>4} {'prompt tok':>10} {'gen tok':>7} {'tok/s':>7} {'wall s':>7}")
        for t in turns:
            print(f"  {t['turn']:>4} {t['prompt_eval_count']:>10} {t['eval_count']:>7} {t['tok_s']:>7} {t['wall_s']:>7}")

        # Ground-truth check against git, not against what the model's own
        # final answer claims to have done. Motivated by a real failure: a
        # session that never called write_file even once still reported
        # "modified 3 files" in prose, and a `pytest` run that trivially
        # passed (nothing had changed) read, out of context, like
        # corroborating evidence. Neither the model's summary nor a passing
        # test run is proof any edit happened -- git status is.
        verified = None
        final_changed = git_changed_files(SANDBOX)
        if final_changed is not None:
            new_changes = [f for f in final_changed if f not in (baseline_changed or [])]
            if not files_written:
                verified = False
                print(f"\n[agent] verification: 0 write_file calls succeeded this session, and git shows "
                      f"{'no changes' if not new_changes else f'{len(new_changes)} changed file(s) anyway (' + ', '.join(new_changes) + ')'} "
                      f"-- if the summary above claims files were modified, that claim is unverified. Treat this "
                      f"as a no-op unless you check `git status` yourself.")
            else:
                unconfirmed = [p for p in files_written if p not in new_changes]
                verified = not unconfirmed
                print(f"\n[agent] verification: write_file succeeded for {len(files_written)} file(s) this session "
                      f"({', '.join(files_written)}); git confirms {len(new_changes)} file(s) actually changed "
                      f"({', '.join(new_changes) if new_changes else 'none'}).")
                if unconfirmed:
                    print(f"[agent] warning: {', '.join(unconfirmed)} reported a successful write but git shows "
                          f"no resulting change -- investigate before trusting that edit happened.")

        with open(SESSION_LOG_PATH, "a") as f:
            f.write(json.dumps(dict(
                ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=model, num_gpu=num_gpu, task=task,
                steps=steps_taken, turns=turns, session_wall_s=session_wall_s,
                files_written=files_written, git_verified=verified,
            )) + "\n")

    for step in range(1, MAX_STEPS + 1):
        try:
            resp = chat(model, messages, num_gpu, num_ctx)
        except (TimeoutError, OSError) as e:
            # Seen for real: a generation can go fully silent mid-stream (no
            # chunks at all, not just slow ones) for longer than the 300s
            # per-read socket timeout in chat() -- a genuine Ollama/driver
            # stall, not the "healthy long generation" that streaming was
            # built to distinguish from a hang. That case still deserves a
            # clear stop and a record of what happened, not a raw traceback
            # a real usage session -- same philosophy as ramer_local.py's
            # safe_measure treating a failed candidate as data, not a crash.
            print(f"\n[agent] stalled: no response from Ollama for over 300s on step {step} ({e}). "
                  f"Stopping -- this is a real stall, not just a slow generation (streaming already "
                  f"tolerates those). Check `ollama ps` / GPU state before retrying.")
            with open(SESSION_LOG_PATH, "a") as f:
                f.write(json.dumps(dict(
                    ts=time.strftime("%Y-%m-%dT%H:%M:%S"), model=model, num_gpu=num_gpu, task=task,
                    steps=step, turns=turns, stalled=True, stall_error=str(e),
                )) + "\n")
            sys.exit(1)
        record_turn(resp)
        msg = resp.get("message", {})
        messages.append(msg)

        used = resp.get("prompt_eval_count") or 0
        if used > 0.85 * num_ctx:
            print(f"[agent] warning: conversation is at {used}/{num_ctx} context tokens -- past the limit Ollama "
                  f"silently drops the oldest messages (the task itself). Re-run with a larger --num-ctx if this keeps going.")

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
            if name == "write_file" and not result.startswith("error") and "path" in args:
                files_written.append(args["path"])
            preview = result if len(result) <= 300 else result[:300] + "... (truncated)"
            print(f"          -> {preview}\n")
            messages.append({"role": "tool", "content": result})

    finish("(stopped: no final answer after the step limit -- likely stuck in a loop)", MAX_STEPS, already_streamed=False)


if __name__ == "__main__":
    main()
