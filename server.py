#!/usr/bin/env python3
"""
Studio Generation Center
========================
A SIMPLIFIED REMOTE CONTROL for a fleet of ComfyUI instances ("lanes"). It is NOT
a replacement for ComfyUI -- you keep ComfyUI for the thought-out work. This is
the quick-idea front door, with plain-language controls instead of a node graph.

  PICTURE tab  -> Qwen-Image-2.1   (text-to-image, image edit)
  VIDEO tab    -> MiniMax-H3       (text-to-video, first/last frame, reference)
  WORKFLOW tab -> a finished still goes straight into a video as its first frame

WHAT THIS PROCESS IS ALLOWED TO DO TO A COMFYUI LANE
----------------------------------------------------
  GET  /system_stats        read status + VRAM
  GET  /queue               read queue depth
  GET  /history/<prompt_id> read job result
  GET  /view?...            read an output file
  POST /upload/image        push a reference file
  POST /prompt              submit a job
  POST /free                drop MODEL WEIGHTS from VRAM (process stays alive)
  POST /interrupt           cancel a job (only from the Stop button)
  WS   /ws?clientId=...     read progress events

That is the whole list. It NEVER restarts, kills, updates or reconfigures a lane,
a container or a model server, and it never deletes a file it did not create.
Your lanes may be running someone else's production work; this app is a polite
guest on them.

Python 3 standard library only -- no pip, no venv, no build step.
Every host, port, model filename and the listen port come from config.json.
"""

import base64
import json
import os
import random
import re
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ---------------------------------------------------------------------------
# Config
#
# NOTHING about your machines is hard-coded in this file. Copy
# config.example.json to config.json and edit that. See the README for the
# full schema; the short version is:
#
#   port      the port this app listens on
#   title     what the page calls itself
#   lanes[]   one entry per ComfyUI instance: id, name, host, port, gpu,
#             gpu_label, caps (["image"] / ["video"] / both)
#   models    the exact .safetensors filenames as your ComfyUI sees them
#   status_only  optional read-only tile (e.g. an LLM server), never dispatched to
# ---------------------------------------------------------------------------

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "data")
CHAIN_DIR = os.path.join(DATA_DIR, "chain")
JOBS_FILE = os.path.join(DATA_DIR, "jobs.json")
CONFIG_FILE = os.environ.get("GENCENTER_CONFIG") or os.path.join(APP_DIR, "config.json")
EXAMPLE_FILE = os.path.join(APP_DIR, "config.example.json")

MODEL_KEYS = (
    "qwen_unet", "qwen_clip", "qwen_vae",
    "h3_unet_fl2va", "h3_unet_ref2va", "h3_clip_nvfp4", "h3_clip_int8",
    "h3_vae_video", "h3_vae_audio", "h3_turbo_lora",
)
LANE_KEYS = ("id", "name", "host", "port", "gpu", "gpu_label", "caps")


def die(msg):
    sys.stderr.write("\n" + msg.rstrip() + "\n\n")
    raise SystemExit(2)


def load_config():
    """Read config.json, or explain exactly what to do instead of crashing."""
    if not os.path.exists(CONFIG_FILE):
        hint = ""
        if os.path.exists(EXAMPLE_FILE):
            hint = ("    cp %s %s\n"
                    "    # then edit it: put in your own ComfyUI hosts, ports and model filenames\n"
                    % (EXAMPLE_FILE, CONFIG_FILE))
        die("No config file at %s\n\n"
            "This app ships no machine addresses of its own. Create one from the example:\n\n"
            "%s\n"
            "Set GENCENTER_CONFIG=/path/to/config.json to keep it somewhere else."
            % (CONFIG_FILE, hint))
    try:
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
    except ValueError as e:
        die("%s is not valid JSON: %s\n"
            "Tip: JSON has no comments and no trailing commas." % (CONFIG_FILE, e))
    if not isinstance(cfg, dict):
        die("%s must contain a JSON object." % CONFIG_FILE)

    lanes = cfg.get("lanes")
    if not isinstance(lanes, list) or not lanes:
        die("%s needs a non-empty \"lanes\" list -- one entry per ComfyUI instance." % CONFIG_FILE)
    seen = set()
    for i, lane in enumerate(lanes):
        if not isinstance(lane, dict):
            die("lanes[%d] in %s must be an object." % (i, CONFIG_FILE))
        missing = [k for k in LANE_KEYS if k not in lane]
        if missing:
            die("lanes[%d] (%s) in %s is missing: %s"
                % (i, lane.get("id", "no id"), CONFIG_FILE, ", ".join(missing)))
        if lane["id"] in seen:
            die("Two lanes share the id %r in %s. Lane ids must be unique."
                % (lane["id"], CONFIG_FILE))
        seen.add(lane["id"])
        caps = lane["caps"]
        if not isinstance(caps, list) or not caps or set(caps) - {"image", "video"}:
            die("lanes[%d] (%s): \"caps\" must be [\"image\"], [\"video\"] or both."
                % (i, lane["id"]))
        lane.setdefault("box", "")
        lane.setdefault("note", "offline")
        try:
            lane["port"] = int(lane["port"])
        except (TypeError, ValueError):
            die("lanes[%d] (%s): \"port\" must be a number." % (i, lane["id"]))

    models = cfg.get("models")
    if not isinstance(models, dict):
        die("%s needs a \"models\" object with the .safetensors filenames." % CONFIG_FILE)
    missing = [k for k in MODEL_KEYS if not models.get(k)]
    if missing:
        die("\"models\" in %s is missing: %s\n"
            "These are the filenames ComfyUI itself shows in its loader dropdowns.\n"
            "If you only run one of the two model families, leave the others as the\n"
            "example's placeholder strings -- they are only read when that tab is used."
            % (CONFIG_FILE, ", ".join(missing)))
    return cfg


CONFIG = load_config()

PORT = int(CONFIG.get("port", 3998))
BIND = CONFIG.get("bind", "0.0.0.0")
TITLE = CONFIG.get("title", "Studio Generation Center")

os.makedirs(CHAIN_DIR, exist_ok=True)

# Every lane is ALWAYS listed in the UI. Down lanes glow red, they are never
# removed. `gpu` is the collision key: two lanes with the same gpu key are on
# the same physical card and cannot both hold weights (H3 ~20GB + Qwen ~17GB
# against a 24GB card). See free_colliding_lanes().
# `gpu_label` is what a human reads on screen: card names, not driver indices.
LANES = CONFIG["lanes"]
LANE_BY_ID = {l["id"]: l for l in LANES}

# Optional read-only strip tile (not a ComfyUI lane, never dispatched to).
# Omit "status_only" from config.json and the tile simply does not appear.
FLEET_LLM = CONFIG.get("status_only") or None

# Model filenames exactly as ComfyUI lists them in its loader dropdowns.
_M = CONFIG["models"]
QWEN_UNET = _M["qwen_unet"]
QWEN_CLIP = _M["qwen_clip"]
QWEN_VAE = _M["qwen_vae"]

H3_UNET_FL2VA = _M["h3_unet_fl2va"]
H3_UNET_REF2VA = _M["h3_unet_ref2va"]
H3_CLIP_NVFP4 = _M["h3_clip_nvfp4"]
H3_CLIP_INT8 = _M["h3_clip_int8"]
H3_VAE_VIDEO = _M["h3_vae_video"]
H3_VAE_AUDIO = _M["h3_vae_audio"]

# The only LoRA wired in is a 4-step speed distillation. There is no quality LoRA.
# Attached automatically on the fast rows; never on the 16/20-step rows, where it
# would fight the schedule.
H3_TURBO_LORA = _M["h3_turbo_lora"]

_T = CONFIG.get("timing") or {}
POLL_SECONDS = float(_T.get("poll_seconds", 4.0))          # lane status poll
JOB_POLL_SECONDS = float(_T.get("job_poll_seconds", 3.0))  # history poll for active jobs
HTTP_TIMEOUT = float(_T.get("http_timeout", 8.0))
FREE_SETTLE_SECONDS = float(_T.get("free_settle_seconds", 2.0))  # let the driver release after /free

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

STATE_LOCK = threading.Lock()
LANE_STATE = {l["id"]: {"up": False, "checked": 0, "err": "never polled"} for l in LANES}
FLEET_STATE = {"up": False, "detail": ""}

JOBS_LOCK = threading.Lock()
JOBS = {}            # job_id -> job dict
JOB_ORDER = deque()  # newest last
PROMPT_INDEX = {}    # (lane_id, prompt_id) -> job_id

LOG = deque(maxlen=250)
LOG_LOCK = threading.Lock()

# One stable clientId per lane so ComfyUI routes that lane's progress events to
# our websocket. Prompts are submitted with the same id. These are PERSISTED:
# after a restart we reconnect with the same id, so ComfyUI keeps routing the
# progress of a job we queued before the restart straight back to us.
CLIENTS_FILE = os.path.join(DATA_DIR, "clients.json")
try:
    with open(CLIENTS_FILE) as _f:
        LANE_CLIENT_ID = json.load(_f)
except Exception:
    LANE_CLIENT_ID = {}
for _l in LANES:
    LANE_CLIENT_ID.setdefault(_l["id"], str(uuid.uuid4()))
try:
    with open(CLIENTS_FILE, "w") as _f:
        json.dump(LANE_CLIENT_ID, _f)
except Exception:
    pass


def log(text, level="info"):
    with LOG_LOCK:
        LOG.appendleft({"ts": time.time(), "text": text, "level": level})
    print("[%s] %s" % (level, text), flush=True)


def suggest_lanes(cap):
    """Name the lanes that can actually do `cap`, for a plain-words error."""
    names = [l["name"] for l in LANES if cap in l["caps"]]
    if not names:
        return "No lane in your config can do that yet."
    if len(names) == 1:
        return "Use %s." % names[0]
    return "Pick %s or %s." % (", ".join(names[:-1]), names[-1])


def human_time(seconds):
    """Plain words, never a raw float on screen."""
    s = int(round(seconds or 0))
    if s < 60:
        return "%d seconds" % s
    m, r = divmod(s, 60)
    if m < 60:
        return "%d min %d s" % (m, r) if r else "%d min" % m
    h, m = divmod(m, 60)
    return "%d h %d min" % (h, m)


# ---------------------------------------------------------------------------
# HTTP helpers (plain stdlib, short timeouts, never raise into a thread loop)
# ---------------------------------------------------------------------------

def lane_url(lane, path):
    return "http://%s:%d%s" % (lane["host"], lane["port"], path)


def http_get_json(url, timeout=HTTP_TIMEOUT):
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def http_get_bytes(url, timeout=60.0):
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), r.headers.get("Content-Type", "application/octet-stream")


def http_post_json(url, payload, timeout=30.0):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        try:
            return {"_http_error": e.code, "_body": json.loads(raw)}
        except Exception:
            return {"_http_error": e.code, "_body": raw[:2000]}


def http_post_multipart(url, fields, files, timeout=180.0):
    """files: list of (fieldname, filename, content_type, data)."""
    boundary = "----genctr%s" % uuid.uuid4().hex
    out = []
    for k, v in fields.items():
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"\r\n\r\n%s\r\n"
                    % (boundary, k, v)).encode("utf-8"))
    for fieldname, filename, ctype, data in files:
        out.append(("--%s\r\nContent-Disposition: form-data; name=\"%s\"; filename=\"%s\"\r\n"
                    "Content-Type: %s\r\n\r\n" % (boundary, fieldname, filename, ctype)).encode("utf-8"))
        out.append(data)
        out.append(b"\r\n")
    out.append(("--%s--\r\n" % boundary).encode("utf-8"))
    body = b"".join(out)
    req = urllib.request.Request(url, data=body, method="POST", headers={
        "Content-Type": "multipart/form-data; boundary=%s" % boundary,
        "Content-Length": str(len(body)),
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8", "replace")
        return {"_http_error": e.code, "_body": raw[:2000]}


# ---------------------------------------------------------------------------
# Lane status poller
# ---------------------------------------------------------------------------

def poll_lane_once(lane):
    st = {"up": False, "checked": time.time()}
    try:
        stats = http_get_json(lane_url(lane, "/system_stats"), timeout=4.0)
        st["up"] = True
        st["comfy"] = stats.get("system", {}).get("comfyui_version", "?")
        devs = stats.get("devices") or []
        if devs:
            d = devs[0]
            st["device"] = d.get("name", "?")
            st["vram_total"] = d.get("vram_total", 0)
            st["vram_free"] = d.get("vram_free", 0)
            st["torch_vram_alloc"] = d.get("torch_vram_total", 0)
        st["ram_free"] = stats.get("system", {}).get("ram_free", 0)
    except Exception as e:
        st["err"] = "%s" % (e,)
        with STATE_LOCK:
            LANE_STATE[lane["id"]] = st
        return
    try:
        q = http_get_json(lane_url(lane, "/queue"), timeout=4.0)
        st["running"] = len(q.get("queue_running", []))
        st["pending"] = len(q.get("queue_pending", []))
        # A ComfyUI queue item is [number, prompt_id, prompt, extra_data, outputs].
        # Knowing which ids are live lets a job we inherited across a restart show
        # as running instead of sitting on "queued" until it finishes.
        ids = []
        for bucket in ("queue_running", "queue_pending"):
            for item in q.get(bucket, []):
                if isinstance(item, list) and len(item) > 1 and isinstance(item[1], str):
                    ids.append(item[1])
        st["live_ids"] = ids
    except Exception:
        st["running"] = 0
        st["pending"] = 0
        st["live_ids"] = []
    with STATE_LOCK:
        LANE_STATE[lane["id"]] = st


def lane_poller(lane):
    # Stagger so seven lanes do not all fire in the same instant.
    time.sleep(random.random() * 2.0)
    while True:
        try:
            poll_lane_once(lane)
        except Exception:
            traceback.print_exc()
        time.sleep(POLL_SECONDS)


def fleet_poller():
    """Optional extra tile: any OpenAI-compatible /v1/models endpoint. Status only."""
    path = FLEET_LLM.get("path", "/v1/models")
    while True:
        try:
            d = http_get_json("http://%s:%d%s" % (FLEET_LLM["host"], FLEET_LLM["port"], path), timeout=4.0)
            ids = [m.get("id") for m in d.get("data", [])]
            FLEET_STATE.update({"up": True, "detail": ", ".join([i for i in ids if i][:2]) or "up"})
        except Exception:
            FLEET_STATE.update({"up": False, "detail": "offline"})
        time.sleep(15.0)


# ---------------------------------------------------------------------------
# Minimal websocket client (stdlib only) for live step progress
# ---------------------------------------------------------------------------

class WSClient(object):
    def __init__(self, host, port, path):
        self.host, self.port, self.path = host, port, path
        self.sock = None
        self.buf = b""

    def connect(self):
        s = socket.create_connection((self.host, self.port), timeout=8.0)
        s.settimeout(40.0)
        key = base64.b64encode(os.urandom(16)).decode()
        req = ("GET %s HTTP/1.1\r\nHost: %s:%d\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n"
               "Sec-WebSocket-Key: %s\r\nSec-WebSocket-Version: 13\r\n\r\n"
               % (self.path, self.host, self.port, key))
        s.sendall(req.encode())
        head = b""
        while b"\r\n\r\n" not in head:
            chunk = s.recv(4096)
            if not chunk:
                raise IOError("closed during handshake")
            head += chunk
        head, rest = head.split(b"\r\n\r\n", 1)
        if b" 101 " not in head.split(b"\r\n")[0] + b" ":
            raise IOError("handshake rejected: %s" % head.split(b"\r\n")[0][:80])
        self.sock, self.buf = s, rest
        return self

    def _need(self, n):
        while len(self.buf) < n:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise IOError("closed")
            self.buf += chunk

    def _send(self, opcode, payload=b""):
        mask = os.urandom(4)
        n = len(payload)
        hdr = bytes([0x80 | opcode])
        if n < 126:
            hdr += bytes([0x80 | n])
        elif n < 65536:
            hdr += bytes([0x80 | 126]) + struct.pack(">H", n)
        else:
            hdr += bytes([0x80 | 127]) + struct.pack(">Q", n)
        hdr += mask
        masked = bytes(b ^ mask[i % 4] for i, b in enumerate(payload))
        self.sock.sendall(hdr + masked)

    def read_message(self):
        """Returns (opcode, data) for a complete message, handling fragments."""
        frames = []
        first_op = None
        while True:
            self._need(2)
            b0, b1 = self.buf[0], self.buf[1]
            fin = b0 & 0x80
            op = b0 & 0x0F
            masked = b1 & 0x80
            ln = b1 & 0x7F
            off = 2
            if ln == 126:
                self._need(4)
                ln = struct.unpack(">H", self.buf[2:4])[0]
                off = 4
            elif ln == 127:
                self._need(10)
                ln = struct.unpack(">Q", self.buf[2:10])[0]
                off = 10
            mask = b""
            if masked:
                self._need(off + 4)
                mask = self.buf[off:off + 4]
                off += 4
            self._need(off + ln)
            data = self.buf[off:off + ln]
            self.buf = self.buf[off + ln:]
            if masked:
                data = bytes(b ^ mask[i % 4] for i, b in enumerate(data))
            if op == 0x9:      # ping -> pong
                self._send(0xA, data)
                continue
            if op == 0xA:      # pong
                continue
            if op == 0x8:      # close
                raise IOError("server closed websocket")
            if first_op is None:
                first_op = op
            frames.append(data)
            if fin:
                return first_op, b"".join(frames)

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass


def ws_listener(lane):
    """Per-lane progress listener. Reconnects forever, never crashes the app."""
    path = "/ws?clientId=" + LANE_CLIENT_ID[lane["id"]]
    backoff = 3.0
    while True:
        with STATE_LOCK:
            up = LANE_STATE[lane["id"]].get("up")
        if not up:
            time.sleep(5.0)
            continue
        ws = WSClient(lane["host"], lane["port"], path)
        try:
            ws.connect()
            backoff = 3.0
            while True:
                op, data = ws.read_message()
                if op != 0x1:      # binary frames are preview images; ignore
                    continue
                try:
                    msg = json.loads(data.decode("utf-8"))
                except Exception:
                    continue
                handle_ws_message(lane, msg)
        except Exception:
            pass
        finally:
            ws.close()
        time.sleep(backoff)
        backoff = min(backoff * 1.6, 30.0)


def handle_ws_message(lane, msg):
    mtype = msg.get("type")
    data = msg.get("data") or {}
    pid = data.get("prompt_id")
    if not pid:
        return
    with JOBS_LOCK:
        jid = PROMPT_INDEX.get((lane["id"], pid))
        if not jid:
            return
        job = JOBS.get(jid)
        if not job or job["status"] in ("done", "error"):
            return
        if mtype == "progress":
            job["status"] = "running"
            job["step"] = int(data.get("value") or 0)
            job["total"] = int(data.get("max") or 0) or job.get("total") or 0
            job["updated"] = time.time()
        elif mtype == "executing":
            job["status"] = "running"
            job["node"] = data.get("node")
            job["updated"] = time.time()
        elif mtype == "execution_error":
            job["status"] = "error"
            job["error"] = str(data.get("exception_message") or data.get("exception_type") or "execution error")
            job["finished"] = time.time()
            log("%s on %s stopped with an error" % (job["kind"].title(), lane["name"]), "error")


# ---------------------------------------------------------------------------
# Job store
# ---------------------------------------------------------------------------

def save_jobs():
    try:
        with JOBS_LOCK:
            keep = list(JOB_ORDER)[-200:]
            out = [JOBS[j] for j in keep if j in JOBS]
        tmp = JOBS_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(out, f)
        os.replace(tmp, JOBS_FILE)
    except Exception:
        traceback.print_exc()


def load_jobs():
    if not os.path.exists(JOBS_FILE):
        return
    try:
        with open(JOBS_FILE) as f:
            arr = json.load(f)
        with JOBS_LOCK:
            for j in arr:
                # A render belongs to the LANE, not to us: restarting this app does
                # not stop it. So keep mid-flight jobs mid-flight and let the history
                # poller resolve them. Only something ancient is truly lost.
                if j.get("status") in ("queued", "running"):
                    if time.time() - j.get("started", 0) > 6 * 3600:
                        j["status"] = "interrupted"
                JOBS[j["id"]] = j
                JOB_ORDER.append(j["id"])
                if j.get("prompt_id"):
                    PROMPT_INDEX[(j["lane"], j["prompt_id"])] = j["id"]
        log("Picked up %d earlier result(s)" % len(arr))
    except Exception:
        traceback.print_exc()


def collect_outputs(hist_entry):
    """ComfyUI puts SaveImage under 'images' and SaveVideo ALSO under 'images'
    (with animated: true). Scan every list of file dicts."""
    outs = []
    for node_id, node_out in (hist_entry.get("outputs") or {}).items():
        for key, val in node_out.items():
            if not isinstance(val, list):
                continue
            for item in val:
                if isinstance(item, dict) and item.get("filename"):
                    fn = item["filename"]
                    ext = os.path.splitext(fn)[1].lower()
                    outs.append({
                        "filename": fn,
                        "subfolder": item.get("subfolder", ""),
                        "type": item.get("type", "output"),
                        "media": "video" if ext in (".mp4", ".webm", ".mov", ".mkv") else "image",
                    })
    return outs


def job_poller():
    """Authoritative completion check. The websocket drives the step counter;
    /history decides done-ness, so a dropped socket never loses a job."""
    while True:
        try:
            with JOBS_LOCK:
                active = [dict(j) for j in JOBS.values()
                          if j.get("status") in ("queued", "running") and j.get("prompt_id")]
            for job in active:
                lane = LANE_BY_ID.get(job["lane"])
                if not lane:
                    continue
                # Nothing on this fleet legitimately runs for six hours. If we are
                # still waiting after that, the lane was restarted under us.
                if time.time() - job.get("started", 0) > 6 * 3600:
                    with JOBS_LOCK:
                        if job["id"] in JOBS:
                            JOBS[job["id"]]["status"] = "interrupted"
                    save_jobs()
                    continue
                if job["status"] == "queued":
                    with STATE_LOCK:
                        live = LANE_STATE.get(lane["id"], {}).get("live_ids") or []
                    if job["prompt_id"] in live:
                        with JOBS_LOCK:
                            if job["id"] in JOBS and JOBS[job["id"]]["status"] == "queued":
                                JOBS[job["id"]]["status"] = "running"
                try:
                    hist = http_get_json(lane_url(lane, "/history/%s" % job["prompt_id"]), timeout=6.0)
                except Exception:
                    continue
                entry = hist.get(job["prompt_id"])
                if not entry:
                    continue
                status = (entry.get("status") or {})
                if not status.get("completed") and status.get("status_str") != "error":
                    continue
                outs = collect_outputs(entry)
                with JOBS_LOCK:
                    j = JOBS.get(job["id"])
                    if not j or j["status"] in ("done", "error"):
                        continue
                    if status.get("status_str") == "error" or (not outs and status.get("completed") is False):
                        j["status"] = "error"
                        msg = ""
                        for m in status.get("messages", []):
                            if m and m[0] == "execution_error":
                                msg = str(m[1].get("exception_message", ""))[:300]
                        j["error"] = msg or "the lane reported an error"
                        log("%s on %s stopped with an error" % (j["kind"].title(), lane["name"]), "error")
                    else:
                        j["status"] = "done"
                        j["outputs"] = outs
                        j["finished"] = time.time()
                        j["elapsed"] = round(j["finished"] - j.get("started", j["finished"]), 1)
                        log("%s finished on %s in %s" % (j["kind"].title(), lane["name"], human_time(j["elapsed"])), "ok")
                save_jobs()
        except Exception:
            traceback.print_exc()
        time.sleep(JOB_POLL_SECONDS)


# ---------------------------------------------------------------------------
# The VRAM rule
# ---------------------------------------------------------------------------

def free_colliding_lanes(target_lane):
    """H3 needs ~20GB, Qwen ~17GB, a 24GB card cannot hold both. Two lanes on the SAME card
    cannot both hold weights. Before dispatching, drop the other lane's weights.

    Returns (ok, notes). ok=False means a sibling is actively rendering and we
    must NOT yank its weights -- we refuse the dispatch instead of OOMing it.
    """
    notes = []
    sibs = [l for l in LANES if l["gpu"] == target_lane["gpu"] and l["id"] != target_lane["id"]]
    for sib in sibs:
        with STATE_LOCK:
            st = dict(LANE_STATE.get(sib["id"], {}))
        if not st.get("up"):
            continue
        try:
            q = http_get_json(lane_url(sib, "/queue"), timeout=5.0)
        except Exception:
            notes.append("Could not check whether %s is busy, so I left its card alone" % sib["name"])
            continue
        busy = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
        if busy:
            # Only suggest a lane that can actually do the thing being asked for.
            with STATE_LOCK:
                other = [l["name"] for l in LANES
                         if set(l["caps"]) & set(target_lane["caps"])
                         and l["gpu"] != target_lane["gpu"]
                         and LANE_STATE.get(l["id"], {}).get("up")]
            tip = ("Try %s instead, or wait about 10 minutes." % other[0]) if other \
                else "Give it about 10 minutes and try again."
            return False, ["%s is busy on %s right now, and both jobs will not fit on one card. %s"
                           % (sib["name"], sib["gpu_label"], tip)]
        before = st.get("vram_free", 0)
        try:
            http_post_json(lane_url(sib, "/free"), {"unload_models": True, "free_memory": True}, timeout=30.0)
        except Exception as e:
            notes.append("Could not make room on %s (%s). Going ahead anyway." % (sib["gpu_label"], e))
            continue
        time.sleep(FREE_SETTLE_SECONDS)
        try:
            after = http_get_json(lane_url(sib, "/system_stats"), timeout=5.0)["devices"][0]["vram_free"]
        except Exception:
            after = before
        gained = after - before
        if gained > 512 * 1024 * 1024:
            notes.append("Made room on %s: %s let go of %.1f GB (%.1f GB free now)"
                         % (sib["gpu_label"], sib["name"], gained / 1e9, after / 1e9))
        else:
            notes.append("%s was already clear, nothing to move" % sib["gpu_label"])
    for n in notes:
        log(n, "vram")
    return True, notes


# ---------------------------------------------------------------------------
# Graph builders
# ---------------------------------------------------------------------------

def snap_frames(length):
    """H3's frame grid is 17n+5. 362 = 15.1s is the trained max; 124 = ~5s the min."""
    length = int(length)
    n = max(0, int(round((length - 5) / 17.0)))
    n = max(7, min(21, n))          # 124 .. 362
    return 17 * n + 5


def qwen_t2i_graph(p):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p["prompt"]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": p.get("negative", "")}},
        "6": {"class_type": "EmptySD3LatentImage",
              "inputs": {"width": p["width"], "height": p["height"], "batch_size": 1}},
        "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.1}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["7", 0], "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0],
            "seed": p["seed"], "steps": p["steps"], "cfg": p["cfg"],
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "gencenter/IMG"}},
    }


def qwen_edit_graph(p):
    """Qwen-Image-2.1 edit. TextEncodeQwenImage21 is the 2.1-native encoder: it
    takes the reference images, emits positive + negative conditioning AND the
    matched latent (its own tooltip: any other latent size shifts the edit)."""
    refs = p.get("ref_images") or []
    if not refs:
        raise ValueError("Add at least one picture to work from.")
    g = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": QWEN_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": QWEN_CLIP, "type": "qwen_image", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": QWEN_VAE}},
        "7": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["1", 0], "shift": 3.1}},
        "4": {"class_type": "TextEncodeQwenImage21", "inputs": {
            "clip": ["2", 0], "prompt": p["prompt"], "negative_prompt": p.get("negative", ""),
            "resolution": int(p.get("resolution", 1024)), "vae": ["3", 0]}},
        "8": {"class_type": "KSampler", "inputs": {
            "model": ["7", 0], "positive": ["4", 0], "negative": ["4", 1], "latent_image": ["4", 2],
            "seed": p["seed"], "steps": p["steps"], "cfg": p["cfg"],
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "gencenter/EDIT"}},
    }
    for i, name in enumerate(refs[:10], start=1):
        nid = str(100 + i)
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["4"]["inputs"]["images.image_%d" % i] = [nid, 0]
    return g


def h3_fl2va_graph(p):
    """The 'cheers' recipe: fl2va unet + NVFP4 AWQ encoder + MiniMaxH3ImageToVideo,
    res_multistep/simple, 20 steps, no LoRA. first_frame/last_frame optional, and
    with neither wired it is pure text-to-video (which is what cheers was)."""
    clip = p.get("encoder") or H3_CLIP_NVFP4
    g = {
        "6": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET_FL2VA, "weight_dtype": "default"}},
        "13": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "minimax", "device": "default"}},
        "11": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE_VIDEO}},
        "24": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE_AUDIO}},
        "104": {"class_type": "MiniMaxH3ImageToVideo", "inputs": {
            "clip": ["13", 0], "vae": ["11", 0], "prompt": p["prompt"],
            "width": p["width"], "height": p["height"], "length": p["length"]}},
        "16": {"class_type": "BasicGuider", "inputs": {"model": ["6", 0], "conditioning": ["104", 0]}},
        "17": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "9": {"class_type": "BasicScheduler", "inputs": {
            "model": ["6", 0], "scheduler": "simple", "steps": p["steps"], "denoise": 1.0}},
        "15": {"class_type": "RandomNoise", "inputs": {"noise_seed": p["seed"]}},
        "14": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["15", 0], "guider": ["16", 0], "sampler": ["17", 0],
            "sigmas": ["9", 0], "latent_image": ["104", 1]}},
        "10": {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["11", 0]}},
        "23": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["14", 0], "vae": ["24", 0]}},
        "91": {"class_type": "CreateVideo", "inputs": {"images": ["10", 0], "audio": ["23", 0], "fps": 24.0}},
        "92": {"class_type": "SaveVideo", "inputs": {
            "video": ["91", 0], "filename_prefix": "gencenter/VID", "format": "auto", "codec": "auto"}},
    }
    if p.get("turbo_lora"):
        # Model-only LoRA: the H3 CLIP is separate, so only the UNET gets patched.
        g["7"] = {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["6", 0], "lora_name": H3_TURBO_LORA, "strength_model": 1.0}}
        g["16"]["inputs"]["model"] = ["7", 0]
        g["9"]["inputs"]["model"] = ["7", 0]
    if p.get("first_frame"):
        g["200"] = {"class_type": "LoadImage", "inputs": {"image": p["first_frame"]}}
        g["104"]["inputs"]["first_frame"] = ["200", 0]
    if p.get("last_frame"):
        g["201"] = {"class_type": "LoadImage", "inputs": {"image": p["last_frame"]}}
        g["104"]["inputs"]["last_frame"] = ["201", 0]
    return g


def h3_ref2va_graph(p):
    """ref2va: up to 9 reference images + up to 3 reference videos (frames at 24fps,
    2-15s each) with the first video's audio carried through. INT8 encoder by default.
    Audio is NEVER fed in as TTS -- H3 speaks the dialogue in the prompt itself."""
    clip = p.get("encoder") or H3_CLIP_INT8
    g = {
        "159": {"class_type": "UNETLoader", "inputs": {"unet_name": H3_UNET_REF2VA, "weight_dtype": "default"}},
        "160": {"class_type": "CLIPLoader", "inputs": {"clip_name": clip, "type": "minimax", "device": "default"}},
        "162": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE_VIDEO}},
        "163": {"class_type": "VAELoader", "inputs": {"vae_name": H3_VAE_AUDIO}},
        "164": {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "clip": ["160", 0], "vae": ["162", 0], "audio_vae": ["163", 0], "prompt": p["prompt"],
            "width": p["width"], "height": p["height"], "length": p["length"],
            "ref_image_size": p.get("ref_image_size", "match")}},
        "166": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "res_multistep"}},
        "167": {"class_type": "BasicScheduler", "inputs": {
            "model": ["159", 0], "scheduler": "simple", "steps": p["steps"], "denoise": 1.0}},
        "168": {"class_type": "BasicGuider", "inputs": {"model": ["159", 0], "conditioning": ["164", 0]}},
        "169": {"class_type": "RandomNoise", "inputs": {"noise_seed": p["seed"]}},
        "177": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["169", 0], "guider": ["168", 0], "sampler": ["166", 0],
            "sigmas": ["167", 0], "latent_image": ["164", 1]}},
        "180": {"class_type": "VAEDecodeAudio", "inputs": {"samples": ["177", 0], "vae": ["163", 0]}},
        "181": {"class_type": "VAEDecode", "inputs": {"samples": ["177", 0], "vae": ["162", 0]}},
        "182": {"class_type": "CreateVideo", "inputs": {"fps": 24, "bit_depth": 8,
                                                        "images": ["181", 0], "audio": ["180", 0]}},
        "184": {"class_type": "SaveVideo", "inputs": {
            "filename_prefix": "gencenter/REF", "format": "auto", "codec": "auto", "video": ["182", 0]}},
    }
    for i, name in enumerate((p.get("ref_images") or [])[:9]):
        nid = str(400 + i)
        g[nid] = {"class_type": "LoadImage", "inputs": {"image": name}}
        g["164"]["inputs"]["ref_images.ref_image_%d" % i] = [nid, 0]
    for i, name in enumerate((p.get("ref_videos") or [])[:3]):
        vid, cid = str(300 + i * 2), str(301 + i * 2)
        g[vid] = {"class_type": "LoadVideo", "inputs": {"file": name}}
        g[cid] = {"class_type": "GetVideoComponents", "inputs": {"video": [vid, 0]}}
        g["164"]["inputs"]["ref_videos.ref_video_%d" % i] = [cid, 0]
        if i == 0 and p.get("keep_audio", True):
            g["164"]["inputs"]["ref_video_audios.ref_video_audio_0"] = [cid, 1]
    return g


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def dispatch(lane, graph, kind, mode, meta):
    ok, notes = free_colliding_lanes(lane)
    if not ok:
        return {"ok": False, "error": notes[0], "notes": notes}

    body = {"prompt": graph, "client_id": LANE_CLIENT_ID[lane["id"]]}
    res = http_post_json(lane_url(lane, "/prompt"), body, timeout=60.0)
    if "_http_error" in res or not res.get("prompt_id"):
        # Keep the machine detail for the expander; show the user one plain sentence.
        detail = res.get("_body", res)
        detail_txt = json.dumps(detail)[:1500] if isinstance(detail, dict) else str(detail)[:1500]
        log("%s turned the job down: %s" % (lane["name"], detail_txt[:300]), "error")
        return {"ok": False,
                "error": "%s would not take that job. Nothing was lost - change a setting and try again, "
                         "or send it to another lane." % lane["name"],
                "detail": detail_txt, "notes": notes}

    jid = uuid.uuid4().hex[:12]
    job = {
        "id": jid, "lane": lane["id"], "lane_name": lane["name"], "prompt_id": res["prompt_id"],
        "kind": kind, "mode": mode, "status": "queued", "step": 0, "total": meta.get("steps", 0),
        "created": time.time(), "started": time.time(), "updated": time.time(),
        "outputs": [], "notes": notes,
    }
    job.update(meta)
    with JOBS_LOCK:
        JOBS[jid] = job
        JOB_ORDER.append(jid)
        PROMPT_INDEX[(lane["id"], res["prompt_id"])] = jid
    log("%s started on %s" % (kind.title(), lane["name"]), "ok")
    save_jobs()
    return {"ok": True, "job": job, "notes": notes}


# ---------------------------------------------------------------------------
# Image fitting for the workflow chain (sips = macOS builtin, no pip)
# ---------------------------------------------------------------------------

def image_size(path):
    out = subprocess.run(["/usr/bin/sips", "-g", "pixelWidth", "-g", "pixelHeight", path],
                         capture_output=True, text=True, timeout=30).stdout
    w = re.search(r"pixelWidth:\s*(\d+)", out)
    h = re.search(r"pixelHeight:\s*(\d+)", out)
    if not (w and h):
        raise ValueError("could not read image size: %s" % out[:200])
    return int(w.group(1)), int(h.group(1))


def fit_to_aspect(src, dst, target_w, target_h):
    """Center-crop to the video aspect, then scale to the exact render size.
    Returns a human sentence describing what we did, for the UI."""
    sw, sh = image_size(src)
    src_ar = sw / float(sh)
    tgt_ar = target_w / float(target_h)
    shutil.copyfile(src, dst)
    if abs(src_ar - tgt_ar) < 0.01:
        subprocess.run(["/usr/bin/sips", "-z", str(target_h), str(target_w), dst], capture_output=True, timeout=60)
        return "aspect already matched, scaled %dx%d -> %dx%d" % (sw, sh, target_w, target_h)
    if src_ar > tgt_ar:
        cw, ch = int(round(sh * tgt_ar)), sh
    else:
        cw, ch = sw, int(round(sw / tgt_ar))
    subprocess.run(["/usr/bin/sips", "-c", str(ch), str(cw), dst], capture_output=True, timeout=60)
    subprocess.run(["/usr/bin/sips", "-z", str(target_h), str(target_w), dst], capture_output=True, timeout=60)
    return "center-cropped %dx%d to %dx%d, then scaled to %dx%d" % (sw, sh, cw, ch, target_w, target_h)


# ---------------------------------------------------------------------------
# Multipart parsing (cgi is deprecated/removed; this is ~40 lines and ours)
# ---------------------------------------------------------------------------

def parse_multipart(body, boundary):
    parts = []
    delim = b"--" + boundary
    for chunk in body.split(delim):
        if not chunk or chunk[:2] == b"--":
            continue
        chunk = chunk.lstrip(b"\r\n")
        if b"\r\n\r\n" not in chunk:
            continue
        raw_head, data = chunk.split(b"\r\n\r\n", 1)
        if data.endswith(b"\r\n"):
            data = data[:-2]
        head = {}
        for line in raw_head.decode("utf-8", "replace").split("\r\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                head[k.strip().lower()] = v.strip()
        disp = head.get("content-disposition", "")
        name = re.search(r'name="([^"]*)"', disp)
        fname = re.search(r'filename="([^"]*)"', disp)
        parts.append({
            "name": name.group(1) if name else "",
            "filename": fname.group(1) if fname else None,
            "content_type": head.get("content-type", "application/octet-stream"),
            "data": data,
        })
    return parts


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "GenerationCenter/1.0"

    def log_message(self, fmt, *args):
        pass

    # -- helpers ------------------------------------------------------------
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_blob(self, data, ctype, filename=None, download=False, ranges=False):
        """`ranges=True` serves a real 206 when asked. Safari will not play a
        <video> from a source that cannot do byte ranges, so the gallery needs it."""
        start, end = 0, len(data) - 1
        partial = False
        if ranges and not download:
            m = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", "") or "")
            if m and len(data):
                g1, g2 = m.group(1), m.group(2)
                if g1:
                    start = min(int(g1), len(data) - 1)
                    end = min(int(g2), len(data) - 1) if g2 else len(data) - 1
                elif g2:                       # suffix form: bytes=-500
                    start = max(0, len(data) - int(g2))
                if start <= end:
                    partial = True
        body = data[start:end + 1] if partial else data
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Accept-Ranges", "bytes" if ranges else "none")
        if partial:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, len(data)))
        if filename:
            disp = "attachment" if download else "inline"
            self.send_header("Content-Disposition", '%s; filename="%s"' % (disp, filename))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        buf = b""
        while len(buf) < n:
            chunk = self.rfile.read(min(65536, n - len(buf)))
            if not chunk:
                break
            buf += chunk
        return buf

    def read_json(self):
        return json.loads(self.read_body().decode("utf-8") or "{}")

    # -- GET ----------------------------------------------------------------
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        try:
            if u.path in ("/", "/index.html"):
                with open(os.path.join(APP_DIR, "index.html"), "rb") as f:
                    return self.send_blob(f.read(), "text/html; charset=utf-8")
            if u.path == "/api/lanes":
                return self.send_json(self.lanes_payload())
            if u.path == "/api/jobs":
                return self.send_json(self.jobs_payload(q))
            if u.path == "/api/view":
                return self.proxy_view(q)
            if u.path == "/api/health":
                return self.send_json({"ok": True, "port": PORT, "lanes": len(LANES)})
            self.send_json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            self.send_json({"error": str(e)}, 500)

    def lanes_payload(self):
        out = []
        with STATE_LOCK:
            snap = {k: dict(v) for k, v in LANE_STATE.items()}
        for l in LANES:
            st = snap.get(l["id"], {})
            out.append({
                "id": l["id"], "name": l["name"], "box": l["box"], "note": l["note"],
                "shared": l.get("shared", ""),
                "endpoint": "%s:%d" % (l["host"], l["port"]), "gpu": l["gpu"], "gpu_label": l["gpu_label"],
                "caps": l["caps"],
                "up": bool(st.get("up")), "err": st.get("err", ""),
                "device": st.get("device", ""),
                "vram_free": st.get("vram_free", 0), "vram_total": st.get("vram_total", 0),
                "running": st.get("running", 0), "pending": st.get("pending", 0),
                "comfy": st.get("comfy", ""), "checked": st.get("checked", 0),
            })
        payload = {"lanes": out, "title": TITLE, "fleet_llm": None}
        if FLEET_LLM:
            payload["fleet_llm"] = dict(FLEET_STATE, name=FLEET_LLM.get("name", "Other server"),
                                        gpu_label=FLEET_LLM.get("gpu_label", ""),
                                        endpoint="%s:%d" % (FLEET_LLM["host"], FLEET_LLM["port"]))
        return payload

    def jobs_payload(self, q):
        limit = int((q.get("limit") or ["60"])[0])
        with JOBS_LOCK:
            ids = list(JOB_ORDER)[-limit:][::-1]
            jobs = [dict(JOBS[i]) for i in ids if i in JOBS]
        with LOG_LOCK:
            logs = list(LOG)[:60]
        return {"jobs": jobs, "log": logs, "now": time.time()}

    def proxy_view(self, q):
        """Pull a finished file from a lane's /view and hand it to the browser."""
        lane = LANE_BY_ID.get((q.get("lane") or [""])[0])
        if not lane:
            return self.send_json({"error": "unknown lane"}, 400)
        params = {
            "filename": (q.get("filename") or [""])[0],
            "subfolder": (q.get("subfolder") or [""])[0],
            "type": (q.get("type") or ["output"])[0],
        }
        url = lane_url(lane, "/view?" + urllib.parse.urlencode(params))
        data, ctype = http_get_bytes(url, timeout=120.0)
        download = (q.get("dl") or ["0"])[0] == "1"
        return self.send_blob(data, ctype, os.path.basename(params["filename"]), download, ranges=True)

    # -- POST ---------------------------------------------------------------
    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        try:
            if u.path == "/api/upload":
                return self.api_upload()
            if u.path == "/api/generate":
                return self.api_generate()
            if u.path == "/api/chain":
                return self.api_chain()
            if u.path == "/api/cancel":
                return self.api_cancel()
            if u.path == "/api/forget":
                return self.api_forget()
            self.send_json({"error": "not found"}, 404)
        except BrokenPipeError:
            pass
        except Exception as e:
            traceback.print_exc()
            self.send_json({"ok": False, "error": str(e)}, 500)

    def api_upload(self):
        """Browser -> us -> the target lane's POST /upload/image."""
        ctype = self.headers.get("Content-Type", "")
        m = re.search(r"boundary=([^;]+)", ctype)
        if not m:
            return self.send_json({"ok": False, "error": "expected multipart"}, 400)
        boundary = m.group(1).strip().strip('"').encode()
        parts = parse_multipart(self.read_body(), boundary)
        lane_id = ""
        files = []
        for p in parts:
            if p["name"] == "lane" and not p["filename"]:
                lane_id = p["data"].decode("utf-8", "replace").strip()
            elif p["filename"]:
                files.append(p)
        lane = LANE_BY_ID.get(lane_id)
        if not lane:
            return self.send_json({"ok": False, "error": "unknown lane %r" % lane_id}, 400)
        uploaded = []
        for p in files:
            res = http_post_multipart(
                lane_url(lane, "/upload/image"),
                {"type": "input", "overwrite": "false"},
                [("image", p["filename"], p["content_type"], p["data"])])
            if "_http_error" in res or not res.get("name"):
                return self.send_json({"ok": False, "error": "%s would not take %s. Try a smaller file or the other lane."
                                       % (lane["name"], p["filename"]), "detail": str(res)[:800]}, 502)
            name = res["name"]
            if res.get("subfolder"):
                name = res["subfolder"] + "/" + name
            uploaded.append({"name": name, "original": p["filename"], "bytes": len(p["data"])})
        log("Sent %d picture/clip(s) over to %s" % (len(uploaded), lane["name"]))
        return self.send_json({"ok": True, "files": uploaded})

    def api_generate(self):
        p = self.read_json()
        lane = LANE_BY_ID.get(p.get("lane"))
        if not lane:
            return self.send_json({"ok": False, "error": "Pick a lane first."}, 400)
        with STATE_LOCK:
            if not LANE_STATE.get(lane["id"], {}).get("up"):
                return self.send_json({"ok": False, "error": "%s is offline right now. Pick one of the lanes glowing green." % lane["name"]}, 409)
        prompt = (p.get("prompt") or "").strip()
        if not prompt:
            return self.send_json({"ok": False, "error": "Tell it what you want first."}, 400)
        seed = int(p.get("seed") or 0) or random.randint(1, 2 ** 48)
        steps = max(1, min(80, int(p.get("steps") or 20)))
        kind = p.get("kind")
        mode = p.get("mode")

        if kind == "image":
            if "image" not in lane["caps"]:
                return self.send_json({"ok": False, "error": "%s does not make pictures. %s"
                                       % (lane["name"], suggest_lanes("image"))}, 400)
            cfg = float(p.get("cfg") or 2.5)
            w = int(p.get("width") or 1328)
            h = int(p.get("height") or 1328)
            w, h = (w // 16) * 16, (h // 16) * 16
            args = {"prompt": prompt, "negative": p.get("negative", ""), "width": w, "height": h,
                    "steps": steps, "cfg": cfg, "seed": seed,
                    "ref_images": p.get("ref_images") or [], "resolution": p.get("resolution", 1024)}
            try:
                graph = qwen_edit_graph(args) if mode == "edit" else qwen_t2i_graph(args)
            except ValueError as e:
                return self.send_json({"ok": False, "error": str(e)}, 400)
            meta = {"prompt": prompt, "negative": p.get("negative", ""), "seed": seed, "steps": steps,
                    "cfg": cfg, "width": w, "height": h, "model": "Qwen-Image-2.1 INT8",
                    "refs": len(args["ref_images"])}
            return self.send_json(dispatch(lane, graph, "image", mode, meta))

        if kind == "video":
            if "video" not in lane["caps"]:
                return self.send_json({"ok": False, "error": "%s does not make videos. %s"
                                       % (lane["name"], suggest_lanes("video"))}, 400)
            w = int(p.get("width") or 960)
            h = int(p.get("height") or 544)
            w, h = (w // 32) * 32, (h // 32) * 32
            length = snap_frames(p.get("length") or 362)
            # The turbo LoRA is a 4-step distillation: attach it for the fast rows (<= 8 steps),
            # never above that, where it fights the schedule and smooths detail away.
            turbo = bool(p.get("turbo_lora")) if p.get("turbo_lora") is not None else (steps <= 8)
            args = {"prompt": prompt, "width": w, "height": h, "length": length, "steps": steps,
                    "turbo_lora": turbo,
                    "seed": seed, "encoder": p.get("encoder") or None,
                    "first_frame": p.get("first_frame"), "last_frame": p.get("last_frame"),
                    "ref_images": p.get("ref_images") or [], "ref_videos": p.get("ref_videos") or [],
                    "keep_audio": bool(p.get("keep_audio", True)),
                    "ref_image_size": p.get("ref_image_size", "match")}
            if mode == "ref2v":
                if not args["ref_images"] and not args["ref_videos"]:
                    return self.send_json({"ok": False, "error":
                                           "Add at least one picture or clip for it to work from."}, 400)
                graph = h3_ref2va_graph(args)
                model = "H3 ref2va INT8"
            elif mode == "fl2va":
                if not args["first_frame"] and not args["last_frame"]:
                    return self.send_json({"ok": False, "error":
                                           "Add a starting picture (an ending picture is optional)."}, 400)
                graph = h3_fl2va_graph(args)
                model = "H3 fl2va"
            else:
                args["first_frame"] = args["last_frame"] = None
                graph = h3_fl2va_graph(args)
                model = "H3 fl2va (text-to-video)"
            meta = {"prompt": prompt, "seed": seed, "steps": steps, "turbo_lora": turbo, "width": w, "height": h,
                    "length": length, "seconds": round(length / 24.0, 2), "model": model,
                    "refs": len(args["ref_images"]), "ref_videos": len(args["ref_videos"]),
                    "chained_from": p.get("chained_from")}
            return self.send_json(dispatch(lane, graph, "video", mode, meta))

        return self.send_json({"ok": False, "error": "unknown kind %r" % kind}, 400)

    def api_chain(self):
        """WORKFLOW: carry a finished Qwen still from its lane straight onto an H3
        lane's input dir. Separate ComfyUI instances do not share an input folder,
        so this is: GET /view on the source -> fit to the video aspect locally
        -> POST /upload/image on the target. The user never touches a file."""
        p = self.read_json()
        with JOBS_LOCK:
            job = dict(JOBS.get(p.get("job_id") or "", {}))
        if not job:
            return self.send_json({"ok": False, "error": "I cannot find that result any more."}, 404)
        src_lane = LANE_BY_ID.get(job["lane"])
        tgt_lane = LANE_BY_ID.get(p.get("target_lane"))
        if not src_lane or not tgt_lane:
            return self.send_json({"ok": False, "error": "unknown lane"}, 400)
        with STATE_LOCK:
            if not LANE_STATE.get(tgt_lane["id"], {}).get("up"):
                return self.send_json({"ok": False, "error": "%s is offline right now. Pick a lane glowing green." % tgt_lane["name"]}, 409)
        idx = int(p.get("output_index") or 0)
        outs = job.get("outputs") or []
        if idx >= len(outs):
            return self.send_json({"ok": False, "error": "That result has no picture to carry over."}, 400)
        out = outs[idx]
        vw = int(p.get("video_width") or 960)
        vh = int(p.get("video_height") or 544)

        url = lane_url(src_lane, "/view?" + urllib.parse.urlencode(
            {"filename": out["filename"], "subfolder": out.get("subfolder", ""), "type": out.get("type", "output")}))
        data, _ = http_get_bytes(url, timeout=120.0)
        base = "chain_%s_%d" % (job["id"], idx)
        raw_path = os.path.join(CHAIN_DIR, base + "_raw.png")
        fit_path = os.path.join(CHAIN_DIR, base + "_%dx%d.png" % (vw, vh))
        with open(raw_path, "wb") as f:
            f.write(data)
        try:
            note = fit_to_aspect(raw_path, fit_path, vw, vh)
        except Exception as e:
            fit_path = raw_path
            note = "could not resize locally (%s), sent the image at its original size" % e

        with open(fit_path, "rb") as f:
            payload = f.read()
        res = http_post_multipart(
            lane_url(tgt_lane, "/upload/image"),
            {"type": "input", "overwrite": "false"},
            [("image", os.path.basename(fit_path), "image/png", payload)])
        if "_http_error" in res or not res.get("name"):
            return self.send_json({"ok": False, "error": "%s would not take the picture. Try the other lane."
                                   % tgt_lane["name"], "detail": str(res)[:800]}, 502)
        name = res["name"]
        if res.get("subfolder"):
            name = res["subfolder"] + "/" + name
        msg = "Carried the picture over to %s (%s)" % (tgt_lane["name"], note)
        log(msg, "chain")
        return self.send_json({"ok": True, "name": name, "note": note, "message": msg,
                               "source_prompt": job.get("prompt", ""), "seed": job.get("seed")})

    def api_cancel(self):
        """Ask a lane to drop its own pending/running item. This is ComfyUI's own
        /interrupt: it cancels a JOB, it does not touch the process."""
        p = self.read_json()
        lane = LANE_BY_ID.get(p.get("lane"))
        if not lane:
            return self.send_json({"ok": False, "error": "unknown lane"}, 400)
        try:
            http_post_json(lane_url(lane, "/interrupt"), {}, timeout=10.0)
        except Exception as e:
            return self.send_json({"ok": False, "error": str(e)}, 502)
        jid = p.get("job_id")
        with JOBS_LOCK:
            if jid and jid in JOBS and JOBS[jid]["status"] in ("queued", "running"):
                JOBS[jid]["status"] = "error"
                JOBS[jid]["error"] = "you stopped this one"
        log("Stopped the job running on %s" % lane["name"], "warn")
        return self.send_json({"ok": True})


    def api_forget(self):
        """Remove a finished result from the gallery. This drops OUR record of the job only:
        the picture or clip stays on the lane's own disk. This app never deletes your files."""
        p = self.read_json()
        jid = p.get("job_id")
        with JOBS_LOCK:
            j = JOBS.get(jid)
            if not j:
                return self.send_json({"ok": False, "error": "no such result"}, 404)
            if j.get("status") in ("queued", "running"):
                return self.send_json({"ok": False, "error": "that one is still going, stop it first"}, 409)
            JOBS.pop(jid, None)
        save_jobs()
        return self.send_json({"ok": True})


def main():
    # Bind first, before starting any threads: if the port is taken there is no
    # point polling seven lanes, and a stack trace is a poor way to say
    # "something else is already using this port".
    try:
        srv = ThreadingHTTPServer((BIND, PORT), Handler)
    except OSError as e:
        if getattr(e, "errno", None) in (48, 98):   # EADDRINUSE on BSD / Linux
            die("Port %d is already in use.\n\n"
                "Either this app is already running, or something else has the port.\n"
                "Find it with:  lsof -nP -iTCP:%d -sTCP:LISTEN\n"
                "Or pick another port by changing \"port\" in %s."
                % (PORT, PORT, os.path.basename(CONFIG_FILE)))
        if getattr(e, "errno", None) == 49:         # EADDRNOTAVAIL
            die("Cannot bind to %r. Check \"bind\" in %s: use \"0.0.0.0\" for every\n"
                "interface, or \"127.0.0.1\" for this machine only."
                % (BIND, os.path.basename(CONFIG_FILE)))
        raise
    srv.daemon_threads = True

    load_jobs()
    for l in LANES:
        threading.Thread(target=lane_poller, args=(l,), daemon=True).start()
        threading.Thread(target=ws_listener, args=(l,), daemon=True).start()
    threading.Thread(target=job_poller, daemon=True).start()
    if FLEET_LLM:
        threading.Thread(target=fleet_poller, daemon=True).start()
    log("%s is up on http://%s:%d" % (TITLE, BIND, PORT), "ok")
    # Basename only: this log is shown in the browser, and a full path on screen
    # is how a home directory ends up in someone's screenshot.
    log("Loaded %d lane(s) from %s" % (len(LANES), os.path.basename(CONFIG_FILE)))
    print("Config file: %s" % CONFIG_FILE, flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
