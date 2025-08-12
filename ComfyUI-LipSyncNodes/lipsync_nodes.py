import os
import json
import math
import shutil
import subprocess
import tempfile
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import hashlib

DEFAULT_OUT_DIR = os.environ.get("LIPSYNC_OUTPUT_DIR", "/Users/brianwankum/pinokio/drive/drives/peers/d1736483982708/output")
CACHE_DIR = os.path.join(DEFAULT_OUT_DIR, ".lipsync_cache")
VERSION = "0.8"
print(f"[LipSyncNodes] module v{VERSION} loaded")

def _which(exe: str) -> Optional[str]:
    return shutil.which(exe)

def _ffmpeg() -> Optional[str]:
    p = _which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None

def _ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p

def _load_image_rgba(path: str) -> Image.Image:
    img = Image.open(path).convert("RGBA")
    return img

def _seconds_to_frame_idx(t: float, fps: int) -> int:
    return int(math.floor(t * fps + 1e-6))

def _read_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def _get_audio_duration_s(path: str) -> float:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".wav":
            import wave
            with wave.open(path, "rb") as w:
                fr = float(w.getframerate() or 1)
                return w.getnframes() / fr
        if ext in (".aif", ".aiff"):
            import aifc
            with aifc.open(path, "rb") as a:
                fr = float(a.getframerate() or 1)
                return a.getnframes() / fr
    except Exception:
        pass
    try:
        from pydub import AudioSegment  # type: ignore
        seg = AudioSegment.from_file(path)
        return float(len(seg)) / 1000.0
    except Exception:
        pass
    try:
        from mutagen import File as MutagenFile  # type: ignore
        mf = MutagenFile(path)
        if mf is not None and getattr(mf, "info", None) and getattr(mf.info, "length", None):
            return float(mf.info.length)
    except Exception:
        pass
    try:
        ffprobe = shutil.which("ffprobe")
        if ffprobe:
            proc = subprocess.run(
                [ffprobe, "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )
            s = (proc.stdout or "").strip()
            if s:
                return float(s)
    except Exception:
        pass
    try:
        ff = _ffmpeg()
        if ff:
            proc = subprocess.run([ff, "-i", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out = (proc.stderr or "") + (proc.stdout or "")
            import re
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", out)
            if m:
                hh, mm, ss = int(m.group(1)), int(m.group(2)), float(m.group(3))
                return float(hh * 3600 + mm * 60) + ss
    except Exception:
        pass
    return 0.0

def _to_bool(x, default=False):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y", "on"):
            return True
        if s in ("false", "0", "no", "n", "off"):
            return False
        if s == "":
            return default
    return bool(x) if x is not None else default

def _to_int(x, default=0):
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return int(x)
        if isinstance(x, int):
            return x
        if isinstance(x, str):
            s = x.strip()
            if not s or s.lower() in ("nan", "none"):
                return default
            return int(float(s))
        return int(float(x))
    except Exception:
        return default

def _expand_time_macros(p: str) -> str:
    import re, time
    if not isinstance(p, str) or "[time(" not in p:
        return p
    def repl(m):
        fmt = m.group(1)
        try:
            return time.strftime(fmt)
        except Exception:
            return fmt
    return re.sub(r"\[time\((.*?)\)\]", repl, p)

import hashlib
def _cache_audio_to_wav(src_path: str, cache_root: str):
    _ensure_dir(cache_root)
    key = hashlib.sha1(str(src_path).encode("utf-8")).hexdigest()[:16]
    ext = os.path.splitext(src_path)[1].lower()
    cached_src = os.path.join(cache_root, f"{key}{ext if ext else '.bin'}")
    cached_wav = os.path.join(cache_root, f"{key}.wav")
    meta_path = os.path.join(cache_root, f"{key}.json")

    def _write_meta(duration_s: float) -> None:
        try:
            st = os.stat(src_path) if os.path.exists(src_path) else None
            meta = {
                "src_path": src_path,
                "size": getattr(st, "st_size", None),
                "mtime": int(getattr(st, "st_mtime", 0) or 0),
                "cached_src": cached_src,
                "cached_wav": cached_wav,
                "duration_s": float(duration_s),
            }
            with open(meta_path, "w") as f:
                json.dump(meta, f)
        except Exception:
            pass

    if os.path.exists(src_path):
        try:
            need_copy = True
            if os.path.exists(cached_src):
                try:
                    if os.path.getmtime(cached_src) >= os.path.getmtime(src_path):
                        need_copy = False
                except Exception:
                    pass
            if need_copy:
                shutil.copy2(src_path, cached_src)
        except Exception as e:
            print(f"[LipSyncNodes] Cache copy failed for {src_path}: {e}")

        try:
            need_wav = True
            if os.path.exists(cached_wav):
                try:
                    if os.path.getmtime(cached_wav) >= os.path.getmtime(cached_src):
                        need_wav = False
                except Exception:
                    pass
            if need_wav:
                ff = _ffmpeg()
                if ff:
                    subprocess.run(
                        [ff, "-y", "-i", cached_src, "-ac", "1", "-ar", "22050", "-vn", "-acodec", "pcm_s16le", cached_wav],
                        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                    )
                else:
                    from pydub import AudioSegment  # type: ignore
                    seg = AudioSegment.from_file(cached_src).set_channels(1).set_frame_rate(22050)
                    seg.export(cached_wav, format="wav")
        except Exception as e:
            print(f"[LipSyncNodes] Cache WAV build failed: {e}")
            return None, 0.0

        dur = _get_audio_duration_s(cached_wav) if os.path.exists(cached_wav) else 0.0
        _write_meta(dur)
        return (cached_wav if os.path.exists(cached_wav) else None, float(dur))

    if os.path.exists(cached_wav):
        try:
            meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        except Exception:
            meta = {}
        dur = meta.get("duration_s") or _get_audio_duration_s(cached_wav)
        print(f"[LipSyncNodes] Source missing; using cached audio: {cached_wav}")
        return (cached_wav, float(dur or 0.0))

    return (None, 0.0)

def _unique_basename(out_dir: str, base: str) -> str:
    base = _expand_time_macros(str(base or "lipsync_out")).strip() or "lipsync_out"
    candidate = base
    n = 1
    while (os.path.exists(os.path.join(out_dir, candidate + ".mp4"))
           or os.path.exists(os.path.join(out_dir, candidate + ".mov"))
           or os.path.isdir(os.path.join(out_dir, candidate + "_png"))):
        candidate = f"{base}_{n:03d}"
        n += 1
    return candidate

def _encode_mp4_from_pngs(ffmpeg_path: str, fps: int, pattern: str, out_path: str) -> bool:
    import subprocess, os
    base_cmd = [ffmpeg_path, "-y", "-framerate", str(fps), "-start_number", "0", "-i", pattern]
    attempts = [
        base_cmd + ["-pix_fmt", "yuv420p", "-c:v", "libx264", "-crf", "18", "-movflags", "+faststart", "-r", str(fps), out_path],
        base_cmd + ["-pix_fmt", "yuv420p", "-c:v", "h264_videotoolbox", "-r", str(fps), out_path],
        base_cmd + ["-pix_fmt", "yuv420p", "-c:v", "mpeg4", "-q:v", "3", "-r", str(fps), out_path],
    ]
    for cmd in attempts:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            return True
    return False

def _encode_prores_from_pngs(ffmpeg_path: str, fps: int, pattern: str, out_path: str) -> bool:
    import subprocess, os
    cmd = [ffmpeg_path, "-y", "-framerate", str(fps), "-start_number", "0", "-i", pattern,
           "-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le",
           "-r", str(fps),
           out_path]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 0

def _safe_replace(src: str, dst: str) -> None:
    try:
        os.replace(src, dst)
    except Exception:
        try:
            if os.path.exists(dst):
                os.remove(dst)
        except Exception:
            pass
        shutil.move(src, dst)

def _mux_audio_attach(ffmpeg_path: str, video_in: str, audio_in: str, out_is_mp4: bool, fps: int):
    if not ffmpeg_path or not os.path.exists(video_in) or not os.path.exists(audio_in):
        return (False, "ffmpeg/video/audio missing")
    root, ext = os.path.splitext(video_in)
    container = "mp4" if ext.lower() == ".mp4" else "mov"
    tmp_out = f"{root}.__tmp__{ext}"
    base = [ffmpeg_path, "-y","-i", video_in,"-i", audio_in,"-map", "0:v:0", "-map", "1:a:0","-shortest"]
    attempts = []
    if out_is_mp4:
        attempts.append(base + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-f", container, tmp_out])
        attempts.append(base + ["-c:v", "copy", "-c:a", "aac_at", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-f", container, tmp_out])
        attempts.append(base + ["-c:v", "copy", "-c:a", "libmp3lame", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart", "-f", container, tmp_out])
        attempts.append([ffmpeg_path, "-y","-i", video_in, "-i", audio_in,"-map", "0:v:0", "-map", "1:a:0","-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "18","-c:a", "aac", "-b:a", "192k", "-ar", "48000","-movflags", "+faststart", "-r", str(fps),"-shortest", "-f", container, tmp_out])
        attempts.append([ffmpeg_path, "-y","-i", video_in, "-i", audio_in,"-map", "0:v:0", "-map", "1:a:0","-c:v", "h264_videotoolbox","-c:a", "aac", "-b:a", "192k", "-ar", "48000","-movflags", "+faststart", "-r", str(fps),"-shortest", "-f", container, tmp_out])
        attempts.append([ffmpeg_path, "-y","-i", video_in, "-i", audio_in,"-map", "0:v:0", "-map", "1:a:0","-c:v", "mpeg4", "-q:v", "3","-c:a", "aac", "-b:a", "192k", "-ar", "48000","-movflags", "+faststart", "-r", str(fps),"-shortest", "-f", container, tmp_out])
    else:
        attempts.append(base + ["-c:v", "copy", "-c:a", "pcm_s16le", "-ar", "48000", "-f", container, tmp_out])
        attempts.append(base + ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-f", container, tmp_out])
        attempts.append([ffmpeg_path, "-y","-i", video_in, "-i", audio_in,"-map", "0:v:0", "-map", "1:a:0","-c:v", "prores_ks", "-profile:v", "4","-c:a", "pcm_s16le", "-ar", "48000","-r", str(fps),"-shortest", "-f", container, tmp_out])
    last_err = ""
    for cmd in attempts:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode == 0 and os.path.exists(tmp_out) and os.path.getsize(tmp_out) > 0:
            _safe_replace(tmp_out, video_in)
            return (True, "")
        else:
            last_err = proc.stderr or f"returncode={proc.returncode}"
            try:
                if os.path.exists(tmp_out):
                    os.remove(tmp_out)
            except Exception:
                pass
    return (False, last_err or "unknown ffmpeg error")

class AudioToPhonemesNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_path": ("STRING", {"multiline": False}),
                "fps": ("INT", {"default": 24, "min": 1, "max": 240}),
                "backend": (["rhubarb"],),
                "language_hint": ("STRING", {"default": "auto"}),
                "rhubarb_path": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("DICT", "INT", "STRING")
    RETURN_NAMES = ("timeline", "fps", "summary")
    FUNCTION = "run"
    CATEGORY = "LipSyncNodes"

    def run(self, audio_path: str, fps: int, backend: str, language_hint: str, rhubarb_path: str):
        duration_guess = 0.0
        try:
            duration_guess = _get_audio_duration_s(audio_path)
        except Exception:
            pass
        if backend != "rhubarb":
            summary = "Unsupported backend"
            timeline = {"fps": int(fps), "origin": "seconds", "events": [],"meta": {"error": summary, "audio_path": audio_path,"audio_duration_s": float(duration_guess)}}
            return (timeline, int(fps), summary)
        exe = rhubarb_path.strip() or _which("rhubarb")
        if not exe:
            summary = "Rhubarb not found"
            print(f"[LipSyncNodes] {summary}")
            timeline = {"fps": int(fps), "origin": "seconds", "events": [],"meta": {"error": summary, "audio_path": audio_path,"audio_duration_s": float(duration_guess)}}
            return (timeline, int(fps), summary)
        audio_for_rhubarb = ""
        cached_wav, d_cache = _cache_audio_to_wav(audio_path, CACHE_DIR)
        if cached_wav:
            audio_for_rhubarb = cached_wav
            if d_cache and d_cache > 0:
                duration_guess = max(duration_guess, d_cache)
        elif os.path.exists(audio_path):
            audio_for_rhubarb = audio_path
            try:
                d2 = _get_audio_duration_s(audio_for_rhubarb)
                if d2 > 0:
                    duration_guess = max(duration_guess, d2)
            except Exception:
                pass
        else:
            summary = f"Audio not found and no cache available: {audio_path}"
            print(f"[LipSyncNodes] {summary}")
            timeline = {"fps": int(fps), "origin": "seconds", "events": [],"meta": {"error": summary, "audio_path": audio_path,"audio_duration_s": float(duration_guess)}}
            return (timeline, int(fps), summary)

        with tempfile.TemporaryDirectory() as td:
            ext = os.path.splitext(audio_for_rhubarb)[1].lower()
            if ext not in (".wav", ".aif", ".aiff"):
                tmp_wav = os.path.join(td, "in.wav")
                ff = _ffmpeg()
                try:
                    if ff:
                        subprocess.run([ff, "-y", "-i", audio_for_rhubarb, "-ac", "1", "-ar", "22050","-vn", "-acodec", "pcm_s16le", tmp_wav],check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        from pydub import AudioSegment  # type: ignore
                        seg = AudioSegment.from_file(audio_for_rhubarb).set_channels(1).set_frame_rate(22050)
                        seg.export(tmp_wav, format="wav")
                    audio_for_rhubarb = tmp_wav
                except Exception as e:
                    print(f"[LipSyncNodes] Audio convert failed, using original: {e}")

            try:
                d2 = _get_audio_duration_s(audio_for_rhubarb)
                if d2 > 0:
                    duration_guess = max(duration_guess, d2)
            except Exception:
                pass

            out_json = os.path.join(td, "rhubarb.json")
            cmd = [exe, "-f", "json", "-o", out_json, audio_for_rhubarb]
            if language_hint and language_hint != "auto":
                cmd.extend(["-r", language_hint])

            try:
                subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                rh = _read_json(out_json)
                cues = rh.get("mouthCues", [])
                events = [{"t0": float(c.get("start", 0)), "t1": float(c.get("end", 0)), "code": str(c.get("value", "X")).strip().upper() or "X"} for c in cues]
                print(f"[LipSyncNodes] Rhubarb cues: {len(events)}  detected audio duration: {duration_guess:.3f}s")
                timeline = {"fps": int(fps), "origin": "seconds", "events": events,"meta": {"source": "rhubarb","audio_path": audio_path,"audio_cached_wav": cached_wav or "","audio_duration_s": float(duration_guess),"used_file": audio_for_rhubarb}}
                return (timeline, int(fps), f"{len(events)} events, duration={duration_guess:.3f}s")
            except subprocess.CalledProcessError as e:
                summary = f"Rhubarb failed: {e}"
                print(f"[LipSyncNodes] {summary}")
                timeline = {"fps": int(fps), "origin": "seconds", "events": [],"meta": {"error": summary,"source": "rhubarb","audio_path": audio_path,"audio_cached_wav": cached_wav or "","audio_duration_s": float(duration_guess),"used_file": audio_for_rhubarb}}
                return (timeline, int(fps), summary)

class LipSyncComposerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "timeline": ("DICT",),
                "fps": ("INT", {"default": 24, "min": 1, "max": 240}),
                "width": ("INT", {"default": 1024, "min": 4, "max": 4096}),
                "height": ("INT", {"default": 1024, "min": 4, "max": 4096}),
                "output_basename": ("STRING", {"default": "lipsync_out"}),
                "background": ("STRING", {"default": "transparent"}),
                "assets_folder": ("STRING", {"default": ""}),
                "use_spritesheet": ("BOOLEAN", {"default": False}),
                "spritesheet_path": ("STRING", {"default": ""}),
                "spritesheet_map_json": ("STRING", {"default": ""}),
                "grid_cols": ("INT", {"default": 0, "min": 0, "max": 32}),
                "grid_rows": ("INT", {"default": 0, "min": 0, "max": 32}),
                "map_csv": ("STRING", {"default": ""}),
                "rest_code": ("STRING", {"default": "X"}),
                "position_x": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                "position_y": ("INT", {"default": 0, "min": -8192, "max": 8192}),
                "scale_percent": ("INT", {"default": 100, "min": 1, "max": 400}),
                "write_png": ("BOOLEAN", {"default": True}),
                "write_mp4": ("BOOLEAN", {"default": True}),
                "write_prores": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
                "attach_audio": ("BOOLEAN", {"default": True}),
                "audio_source": ("STRING", {"default": ""})
            }
        }
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "png_folder")
    FUNCTION = "compose"
    CATEGORY = "LipSyncNodes"

    def _read_map_csv(self, csv_path: str):
        import csv
        mapping = {}
        if not csv_path or not os.path.exists(csv_path):
            codes = list("ABCDEFGH") + ["X"]
            for c in codes:
                fname = f"pb_{c}.png" if c != "X" else "pb_rest.png"
                mapping[c] = {"file": fname, "priority": 1, "min_hold_frames": 2, "cf_in": 1, "cf_out": 1}
            return mapping
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                code = row.get("code", "").strip().upper()
                if not code: continue
                mapping[code] = {"file": row.get("file", "").strip(),"priority": int(row.get("priority", "1")),"min_hold_frames": int(row.get("min_hold_frames", "2")),"cf_in": int(row.get("cf_in", "1")),"cf_out": int(row.get("cf_out", "1")),}
        return mapping

    def _build_asset_lookup(self, use_spritesheet, assets_folder, spritesheet_path, spritesheet_map_json, grid_cols, grid_rows, map_csv):
        mapping = self._read_map_csv(map_csv)
        out = {}
        if use_spritesheet and spritesheet_path:
            sheet = Image.open(spritesheet_path).convert("RGBA")
            rects = {}
            if spritesheet_map_json and os.path.exists(spritesheet_map_json):
                raw = _read_json(spritesheet_map_json)
                for k, v in raw.items():
                    rects[k.upper()] = tuple(map(int, v))
            elif grid_cols > 0 and grid_rows > 0:
                cell_w = sheet.width // grid_cols
                cell_h = sheet.height // grid_rows
                order = list(mapping.keys())
                idx = 0
                for r in range(grid_rows):
                    for c in range(grid_cols):
                        if idx >= len(order): break
                        x = c * cell_w; y = r * cell_h
                        rects[order[idx]] = (x, y, cell_w, cell_h); idx += 1
            for code in mapping.keys():
                rect = rects.get(code)
                if not rect: continue
                x, y, w, h = rect
                out[code] = sheet.crop((x, y, x+w, y+h)).convert("RGBA")
        else:
            base = assets_folder or ""
            for code, conf in mapping.items():
                fp = os.path.join(base, conf["file"]) if base else conf["file"]
                if os.path.exists(fp):
                    out[code] = Image.open(fp).convert("RGBA")
        if not out:
            out["X"] = Image.new("RGBA", (1,1), (0,0,0,0))
        return out

    def _timeline_to_frame_codes(self, timeline, fps, mapping, rest_code):
        events = timeline.get("events", [])
        t_end = max([float(ev.get("t1", 0.0)) for ev in events], default=0.0)
        try:
            duration_hint = float(timeline.get("meta", {}).get("audio_duration_s", 0.0))
        except Exception:
            duration_hint = 0.0
        if duration_hint <= 0:
            meta = timeline.get("meta", {}) or {}
            used_file = meta.get("used_file") or meta.get("audio_path")
            if isinstance(used_file, str) and os.path.exists(used_file):
                try:
                    dh2 = _get_audio_duration_s(used_file)
                    if dh2 > 0:
                        print(f"[LipSyncNodes] Fallback measured audio duration: {dh2:.3f}s from {used_file}")
                        duration_hint = dh2
                except Exception:
                    pass
        total_seconds = max(t_end, duration_hint)
        total_frames = max(1, int(math.ceil(total_seconds * fps)))
        frames = [rest_code] * total_frames
        for i in range(total_frames):
            t = i / fps
            best_code = rest_code
            best_pri = -9999
            for ev in events:
                t0 = float(ev.get("t0", 0.0))
                t1 = float(ev.get("t1", 0.0))
                if t0 <= t < t1:
                    code = str(ev.get("code", rest_code)).upper()
                    pri = mapping.get(code, {}).get("priority", 1)
                    if pri >= best_pri:
                        best_pri = pri
                        best_code = code
            frames[i] = best_code
        i = 0
        while i < len(frames):
            c = frames[i]
            hold = mapping.get(c, {}).get("min_hold_frames", 1)
            run = 1
            j = i + 1
            while j < len(frames) and frames[j] == c:
                run += 1
                j += 1
            if run < hold:
                end = min(len(frames), i + hold)
                for k in range(i, end):
                    frames[k] = c
                j = end
            i = j
        print(f"[LipSyncNodes] frames={len(frames)} fps={fps} t_end={t_end:.3f}s audio={duration_hint:.3f}s")
        return frames

    def compose(self, timeline, fps, width, height, output_basename, background,
                assets_folder, use_spritesheet, spritesheet_path, spritesheet_map_json,
                grid_cols, grid_rows, map_csv, rest_code,
                position_x, position_y, scale_percent, write_png, write_mp4, write_prores,
                output_dir="", attach_audio=True, audio_source=""):
        output_dir = str(output_dir or "").strip()
        if not output_dir:
            output_dir = os.environ.get("LIPSYNC_OUTPUT_DIR", DEFAULT_OUT_DIR)
        from os.path import expanduser, expandvars
        output_dir = _expand_time_macros(expandvars(expanduser(output_dir)))
        _ensure_dir(output_dir)
        print(f"[LipSyncNodes] Output directory: {output_dir}")
        assets_folder = str(assets_folder or "").strip()
        spritesheet_path = str(spritesheet_path or "").strip()
        spritesheet_map_json = str(spritesheet_map_json or "").strip()
        map_csv = str(map_csv or "").strip()
        rest_code = (str(rest_code or "X").strip().upper() or "X")
        background = str(background or "transparent").strip()
        use_spritesheet = _to_bool(use_spritesheet)
        write_png = _to_bool(write_png, True)
        write_mp4 = _to_bool(write_mp4, True)
        write_prores = _to_bool(write_prores, False)
        grid_cols = _to_int(grid_cols, 0)
        grid_rows = _to_int(grid_rows, 0)
        position_x = _to_int(position_x, 0)
        position_y = _to_int(position_y, 0)
        scale_percent = max(1, _to_int(scale_percent, 100))
        width = _to_int(width, 1024)
        height = _to_int(height, 1024)
        fps = _to_int(fps or timeline.get("fps", 24), 24)
        mapping = self._read_map_csv(map_csv)
        assets = self._build_asset_lookup(use_spritesheet, assets_folder, spritesheet_path, spritesheet_map_json, grid_cols, grid_rows, map_csv)
        frames_codes = self._timeline_to_frame_codes(timeline, fps, mapping, rest_code or "X")
        print(f"[LipSyncNodes] Will render {len(frames_codes)} PNGs at {fps} fps → ~{len(frames_codes)/max(fps,1):.2f}s")
        transparent = (background.strip().lower() == "transparent")
        if transparent:
            bg_rgba = (0,0,0,0)
        else:
            from PIL import ImageColor
            bg_rgba = ImageColor.getcolor(background, "RGBA")
        import subprocess, tempfile
        work_dir = output_dir
        unique_base = _unique_basename(work_dir, output_basename)
        tmp_png_dir_obj = None
        if write_png:
            png_dir = os.path.join(work_dir, f"{unique_base}_png"); os.makedirs(png_dir, exist_ok=True)
        else:
            tmp_png_dir_obj = tempfile.TemporaryDirectory(prefix=f"{unique_base}_png_")
            png_dir = tmp_png_dir_obj.name
        mp4_path = os.path.join(work_dir, f"{unique_base}.mp4")
        prores_path = os.path.join(work_dir, f"{unique_base}.mov")
        scale = max(1, scale_percent) / 100.0
        W, H = width, height
        scaled_assets = {}
        for code, img in assets.items():
            w = max(1, int(img.width * scale)); h = max(1, int(img.height * scale))
            scaled_assets[code] = img.resize((w, h), Image.LANCZOS)
        need_png_seq = bool(write_png or write_mp4 or write_prores)
        for idx, code in enumerate(frames_codes):
            canvas = Image.new("RGBA", (W, H), bg_rgba)
            mouth_img = scaled_assets.get(code) or scaled_assets.get(rest_code)
            if mouth_img is None:
                mouth_img = Image.new("RGBA", (1,1), (0,0,0,0))
            px = (W - mouth_img.width)//2 + position_x
            py = (H - mouth_img.height)//2 + position_y
            canvas.alpha_composite(mouth_img, (px, py))
            if need_png_seq:
                canvas.save(os.path.join(png_dir, f"{unique_base}_{idx:06d}.png"))
        ff = _ffmpeg()
        video_out = ""
        pattern = os.path.join(png_dir, f"{unique_base}_%06d.png")
        if write_mp4 or write_prores:
            if not ff:
                print("[LipSyncNodes] ffmpeg not found — cannot write video. Install ffmpeg or `pip install imageio-ffmpeg`.")
            else:
                ok_mp4 = False
                ok_prores = False
                if write_mp4:
                    ok_mp4 = _encode_mp4_from_pngs(ff, fps, pattern, mp4_path)
                if write_prores:
                    ok_prores = _encode_prores_from_pngs(ff, fps, pattern, prores_path)
                if ok_mp4:
                    video_out = mp4_path
                elif ok_prores:
                    video_out = prores_path
                else:
                    print("[LipSyncNodes] ffmpeg encoding failed. See console for details.")
        if attach_audio and (write_mp4 or write_prores) and video_out:
            ff = _ffmpeg()
            if ff:
                meta = (timeline.get("meta") or {}) if isinstance(timeline, dict) else {}
                candidates = []
                if str(audio_source or "").strip():
                    candidates.append(str(audio_source).strip())
                cw = meta.get("audio_cached_wav")
                if cw:
                    candidates.append(str(cw))
                uf = meta.get("used_file")
                if uf:
                    candidates.append(str(uf))
                ap = meta.get("audio_path")
                if ap:
                    candidates.append(str(ap))
                chosen = ""
                from os.path import expanduser, expandvars
                for c in candidates:
                    p = expandvars(expanduser(str(c)))
                    if os.path.exists(p):
                        chosen = p; break
                if not chosen:
                    print("[LipSyncNodes] No readable audio to mux (attach_audio=True). Candidates tried:", candidates)
                else:
                    print(f"[LipSyncNodes] Muxing audio → {os.path.basename(video_out)}  audio={chosen}")
                    is_mp4 = video_out.lower().endswith(".mp4")
                    ok, err = _mux_audio_attach(ff, video_out, chosen, is_mp4, fps)
                    if ok:
                        print("[LipSyncNodes] Audio muxed successfully →", os.path.basename(video_out))
                    else:
                        print("[LipSyncNodes] Audio mux failed. ffmpeg said:\n" + str(err))
            else:
                print("[LipSyncNodes] ffmpeg not found — cannot mux audio. Install ffmpeg or `pip install imageio-ffmpeg`.")
        if write_png:
            png_out = png_dir
        else:
            png_out = ""
            if tmp_png_dir_obj is not None:
                try: tmp_png_dir_obj.cleanup()
                except Exception: pass
        return (video_out, png_out)

class PhonemesToCartoonVideoNode(LipSyncComposerNode):
    @classmethod
    def INPUT_TYPES(cls):
        return LipSyncComposerNode.INPUT_TYPES()
    RETURN_TYPES = LipSyncComposerNode.RETURN_TYPES
    RETURN_NAMES = LipSyncComposerNode.RETURN_NAMES
    FUNCTION = "compose"
    CATEGORY = "LipSyncNodes"

NODE_CLASS_MAPPINGS = {
    "AudioToPhonemesNode": AudioToPhonemesNode,
    "LipSyncComposerNode": LipSyncComposerNode,
    "PhonemesToCartoonVideoNode": PhonemesToCartoonVideoNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AudioToPhonemesNode": "Audio → Phonemes (Rhubarb)",
    "LipSyncComposerNode": "LipSync Composer",
    "PhonemesToCartoonVideoNode": "Phonemes → Cartoon Video",
}
