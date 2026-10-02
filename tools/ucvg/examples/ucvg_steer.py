#!/usr/bin/env python3
"""
ucvg_steer.py — test the modified llama-server by chatting to an LLM with vectors applied to it.
Point SERVER_EXE at your llama-server binary to auto-launch it. Leave it None if you will start the server yourself.

Loads one or more control vectors into a (started or already-running) llama-server
and steers each response by sending an additive offset per request.
"""

import json
import subprocess
import sys
import threading
import time
from typing import Dict, List, Optional, TypedDict

import requests

# Configuration

# Point SERVER_EXE at your llama-server binary to auto-launch it; leave it None to start one yourself.
SERVER_EXE: Optional[str] = None
# SERVER_EXE = r"C:\path\to\llama-server.exe"

MODEL_PATH  = "<path/to/model.gguf>"
HOST        = "127.0.0.1"
PORT        = 8000
BASE_URL    = f"http://{HOST}:{PORT}"

MODEL_NAME    = "llama"   # arbitrary name sent to the server
MAX_TOKENS    = 512
TEMPERATURE   = 1.0
SYSTEM_PROMPT = "Respond without markdown."

# Extra request fields for model-specific chat templates (e.g. thinking toggles).
EXTRA_BODY: Dict[str, object] = {"chat_template_kwargs": {"enable_thinking": False}}

# Server launch options (used only when SERVER_EXE is set); a --dynamic-cv-add flag is appended per VECTORS entry.
SERVER_ARGS: List[str] = [
    "-m", MODEL_PATH,
    "--host", HOST, "--port", str(PORT),
    "-ngl", "99",
    "--ctx-size", "8192",
    "-ctk", "q8_0", "-ctv", "q8_0",   # no reason not to quantize the context to >= 8bit
    "--repeat-penalty", "1.15",        # raise this when using larger CV magnitudes so the model loops less
]

# Control vectors to load and steer; these are the defaults, editable at runtime (see /help).
class Vector(TypedDict):
    name: str          # unique id used by the / commands
    additive: float    # constant offset along the vector (0 = none)
    scale_pos: float   # gain on deviations ABOVE the default state (1 = no gain)
    scale_neg: float   # gain on deviations BELOW the default state (1 = no gain)
    path: str          # control-vector .gguf to load

VECTORS: List[Vector] = [
    {
        "name":      "example",
        "additive":  0.0,
        "scale_pos": 1.0,
        "scale_neg": 1.0,
        "path":      "<path/to/your/control_vector.gguf>",
    },
]

_server: Optional[subprocess.Popen] = None


# Server management

def is_running() -> bool:
    try:
        return requests.get(f"{BASE_URL}/health", timeout=2).status_code == 200
    except requests.RequestException:
        return False


def _is_info_line(line: str) -> bool:
    """True for an 'I' (info) server log line, e.g. '0.00.123.456 I srv ...'."""
    parts = line.split()
    return len(parts) >= 2 and parts[0][0].isdigit() and parts[1] == "I"


def _mirror_server_output(pipe) -> None:
    """Mirror the launched server's warnings/errors (skipping 'I' info lines) to this console."""
    for raw in iter(pipe.readline, b""):
        line = raw.decode("utf-8", errors="replace")
        if not _is_info_line(line):
            sys.stdout.write(line)   # drop info ('I'); keep W/E + anything else (e.g. a crash)
            sys.stdout.flush()


def start_server() -> None:
    global _server
    if SERVER_EXE is None:          # ensure_server already guards this; also narrows it to str for the checker
        return
    cmd = [SERVER_EXE] + SERVER_ARGS
    for v in VECTORS:
        cmd += ["--dynamic-cv-add", str(v["path"])]
    print(f"Starting llama-server on {BASE_URL} ...")
    # Capture the server's stdout/stderr through a pipe (not the console) and mirror only
    # warnings/errors to this terminal, dropping the 'I' info lines. No window, no log files.
    _server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    threading.Thread(target=_mirror_server_output, args=(_server.stdout,), daemon=True).start()


def stop_server() -> None:
    global _server
    if _server is not None:
        print("\nStopping llama-server ...")
        _server.terminate()
        _server.wait()
        _server = None


def ensure_server() -> None:
    if is_running():
        return
    if SERVER_EXE is None:
        raise SystemExit(f"No server at {BASE_URL}. Start one, or set SERVER_EXE to auto-launch.")
    start_server()
    if _server is None:                          # only set when SERVER_EXE was actually configured
        raise SystemExit(f"No server at {BASE_URL}. Start one, or set SERVER_EXE to auto-launch.")
    deadline = time.time() + 180
    while time.time() < deadline:
        if _server.poll() is not None:          # it crashed during startup (output is above)
            raise SystemExit("llama-server exited before becoming healthy — see output above.")
        if is_running():
            return
        time.sleep(1)
    raise SystemExit("llama-server did not become healthy in time.")


# Chat

def coefficients() -> Dict[str, object]:
    """This request's per-vector coefficients: additive offsets + asymmetric gains."""
    return {
        "additive_vector_coefficients": [float(v["additive"]) for v in VECTORS],
        "multiplicative_vector_coefficients": [
            {"scale_positive": float(v["scale_pos"]), "scale_negative": float(v["scale_neg"])}
            for v in VECTORS
        ],
    }


def chat(messages: List[Dict[str, str]]) -> str:
    body = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "stream": True,
        **coefficients(),
        **EXTRA_BODY,
    }
    out: List[str] = []
    with requests.post(f"{BASE_URL}/chat/completions", json=body, stream=True) as r:
        if r.status_code != 200:
            raise RuntimeError(f"server rejected the request (HTTP {r.status_code}): {r.text}")
        for line in r.iter_lines():
            if not line:
                continue
            s = line.decode() if isinstance(line, bytes) else str(line)
            s = s.strip()
            if s.startswith("data:"):                    # SSE event prefix (text/event-stream)
                s = s[5:].strip()
            if not s or s == "[DONE]" or s.startswith(":"):   # blank, terminator, or comment ping
                continue
            choices = json.loads(s).get("choices")
            if not choices:
                continue
            text = choices[0].get("delta", {}).get("content")
            if text is None:
                continue
            sys.stdout.write(text)
            sys.stdout.flush()
            out.append(text)
    print()
    return "".join(out)


# Commands

def find(name: str) -> Optional[Vector]:
    for v in VECTORS:
        if str(v["name"]).lower() == name.lower():
            return v
    return None


def show() -> None:
    for v in VECTORS:
        print(f"  {str(v['name']):<14} additive={float(v['additive']):+.3f}   "
              f"scale_pos={float(v['scale_pos']):.2f}   scale_neg={float(v['scale_neg']):.2f}")


def set_coeff(line: str, word: str, key: str) -> None:
    parts = line.split()
    if len(parts) < 3:
        print(f"Usage: /{word} <name> <value>")
        return
    v = find(parts[1])
    if v is None:
        print(f"No vector named '{parts[1]}'")
        return
    try:
        value = float(parts[2])
    except ValueError:
        print(f"'{parts[2]}' is not a number")
        return
    # TypedDict fields are assigned by literal key, so map the command's field here.
    if key == "additive":
        v["additive"] = value
    elif key == "scale_pos":
        v["scale_pos"] = value
    elif key == "scale_neg":
        v["scale_neg"] = value
    else:
        print(f"Unknown field '{key}'")
        return
    show()


def command(line: str) -> bool:
    """Handle a /command; returns True if the user asked to quit."""
    name = line.split()[0].lstrip("/")
    if name in ("", "show"):
        show()
    elif name == "add":
        set_coeff(line, "add", "additive")
    elif name == "pos":
        set_coeff(line, "pos", "scale_pos")
    elif name == "neg":
        set_coeff(line, "neg", "scale_neg")
    elif name in ("help", "?"):
        print("  /add <name> <offset>   set the additive offset (0 = none)")
        print("  /pos <name> <gain>     gain on deviations above default (1 = off)")
        print("  /neg <name> <gain>     gain on deviations below default (1 = off)")
        print("  /                      show the current coefficients")
        print("  /quit                  stop the server and exit")
        print("  (anything else is sent to the model; empty line skips; Ctrl-C quits)")
    elif name == "quit":
        return True
    else:
        print(f"Unknown '/{name}' — type /help")
    return False


# Main loop

def main() -> None:
    ensure_server()
    messages: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []
    print(f"\nUCVG steering · {BASE_URL}")
    print("Type /help for commands; anything else is sent to the model. Ctrl-C to quit.\n")
    show()
    while True:
        try:
            user = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not user.strip():
            continue
        if user.startswith("/"):
            if command(user):
                break
            continue
        messages.append({"role": "user", "content": user})
        try:
            reply = chat(messages)
        except Exception as e:
            print(f"\nError: {e}")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    try:
        main()
    finally:
        stop_server()
