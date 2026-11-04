"""Cross-platform check of wakeforge wake words, through ovos-plugin-manager as ovos-dinkum-listener loads them.

    python scripts/platform_check.py golden --out test/golden/streams.json     # on the reference machine
    python scripts/platform_check.py check --golden test/golden/streams.json [--bench] [--json out.json]

The stream is every clip of ``test/clips`` and ``test/golden`` between 1.5 s of silence. ``golden`` records, per
model, the raw score of every 80 ms block (threshold above 1, so no detection resets the stream) and the blocks where
the model fires at its own threshold. ``check`` replays the stream on this machine and fails when a score is more
than ``--tolerance`` (0.002) away from the golden one or a detection appears, disappears or moves by more than one
block. ``--bench`` also reports the CPU each model costs per second of audio on one thread, beside openWakeWord's
hey_jarvis on the same audio (when the ``openwakeword`` package is installed).
"""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
SR, CHUNK = 16000, 1280
MODELS = {
    "wakehubert_hey_jarvis": "wakehubert_hey_jarvis",   # a ready model of the plugin, by name
    "thalovant": "https://huggingface.co/thalovant/thalovant-wakeword/resolve/v2/thalovant.onnx",
    "zoguilo": "https://huggingface.co/thalovant/zoguilo-wakeword/resolve/main/zoguilo.onnx",
}


def stream():
    gap = np.zeros(int(1.5 * SR), np.float32)
    parts, names = [gap], []
    for f in sorted((ROOT / "test" / "clips").glob("*.wav")) + sorted((ROOT / "test" / "golden").glob("*.wav")):
        w, sr = sf.read(str(f), dtype="float32", always_2d=True)
        assert sr == SR, f"{f}: {sr} Hz"
        parts += [w[:, 0], gap]
        names.append(f.name)
    y = np.concatenate(parts)
    y = y[:len(y) // CHUNK * CHUNK]
    return (np.clip(y, -1, 1) * 32767).astype(np.int16), names


def plugin(ref, threshold=None):
    from ovos_plugin_manager.wakewords import OVOSWakeWordFactory
    hw = {"module": "ovos-ww-plugin-wakeforge", "model": ref, "listen": True}
    if threshold is not None:
        hw["threshold"] = threshold
    return OVOSWakeWordFactory.create_hotword("ww", {"hotwords": {"ww": hw}, "listener": {"wake_word": "ww"}})


def run(ref, pcm):
    ww = plugin(ref, threshold=1.01)
    scores = []
    for k in range(len(pcm) // CHUNK):
        ww.update(pcm[k * CHUNK:(k + 1) * CHUNK].tobytes())
        scores.append(float(ww.last_score or 0.0))
    ww = plugin(ref)
    fired = []
    for k in range(len(pcm) // CHUNK):
        ww.update(pcm[k * CHUNK:(k + 1) * CHUNK].tobytes())
        if ww.found_wake_word():
            fired.append(k)
    return scores, fired, float(ww.threshold)


def cpu_name():
    try:
        if sys.platform == "darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        if sys.platform == "win32":
            return platform.processor()
        for line in open("/proc/cpuinfo"):
            if line.lower().startswith(("model name", "hardware", "cpu model")):
                return line.split(":", 1)[1].strip()
        parts = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith(("CPU implementer", "CPU part"))]
        return "ARM " + " ".join(sorted(set(parts)))
    except Exception:
        return platform.processor()


def bench(ref, pcm, seconds=60.0):
    """Process CPU per second of audio on one thread, over ``seconds`` of the stream looped."""
    reps = int(np.ceil(seconds * SR / len(pcm)))
    loop = np.tile(pcm, reps)[:int(seconds * SR) // CHUNK * CHUNK]
    ww = plugin(ref)
    for k in range(25):                                  # warm-up: 2 s
        ww.update(loop[k * CHUNK:(k + 1) * CHUNK].tobytes())
    t0, w0 = time.process_time(), time.perf_counter()
    for k in range(len(loop) // CHUNK):
        ww.update(loop[k * CHUNK:(k + 1) * CHUNK].tobytes())
        ww.found_wake_word()
    return {"cpu_ms_per_s": round((time.process_time() - t0) / seconds * 1000, 2),
            "wall_ms_per_s": round((time.perf_counter() - w0) / seconds * 1000, 2)}


def bench_openwakeword(pcm, seconds=60.0):
    try:
        import openwakeword
        from openwakeword.model import Model
        from openwakeword.utils import download_models
    except ImportError:
        return None
    d = Path(openwakeword.__file__).parent / "resources" / "models"
    if not (d / "hey_jarvis_v0.1.onnx").exists():
        download_models(["hey_jarvis"])
    m = Model(wakeword_models=[str(d / "hey_jarvis_v0.1.onnx")], inference_framework="onnx")
    reps = int(np.ceil(seconds * SR / len(pcm)))
    loop = np.tile(pcm, reps)[:int(seconds * SR) // CHUNK * CHUNK]
    for k in range(25):
        m.predict(loop[k * CHUNK:(k + 1) * CHUNK])
    t0, w0 = time.process_time(), time.perf_counter()
    for k in range(len(loop) // CHUNK):
        m.predict(loop[k * CHUNK:(k + 1) * CHUNK])
    return {"cpu_ms_per_s": round((time.process_time() - t0) / seconds * 1000, 2),
            "wall_ms_per_s": round((time.perf_counter() - w0) / seconds * 1000, 2)}


def info():
    import onnxruntime as ort
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu_name(),
            "python": platform.python_version(), "onnxruntime": ort.__version__, "numpy": np.__version__,
            "qemu_cpu": os.environ.get("QEMU_CPU", "")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["golden", "check"])
    ap.add_argument("--out", default=str(ROOT / "test" / "golden" / "streams.json"))
    ap.add_argument("--golden", default=str(ROOT / "test" / "golden" / "streams.json"))
    ap.add_argument("--tolerance", type=float, default=0.002)
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    pcm, names = stream()
    if a.cmd == "golden":
        out = {"info": info(), "clips": names, "blocks": len(pcm) // CHUNK, "models": {}}
        for name, ref in MODELS.items():
            scores, fired, thr = run(ref, pcm)
            out["models"][name] = {"ref": ref, "threshold": thr, "fired": fired, "scores": [round(s, 6) for s in scores]}
            print(name, "fires at blocks", fired)
        Path(a.out).write_text(json.dumps(out) + "\n")
        return
    golden = json.loads(Path(a.golden).read_text())
    res, ok = {"info": info(), "models": {}}, True
    for name, g in golden["models"].items():
        scores, fired, thr = run(g["ref"], pcm)
        diff = float(np.max(np.abs(np.array(scores) - np.array(g["scores"]))))
        moved = sorted(set(fired) ^ set(g["fired"]))
        close = all(any(abs(f - h) <= 1 for h in g["fired"]) for f in fired) and \
            all(any(abs(f - h) <= 1 for h in fired) for f in g["fired"]) and len(fired) == len(g["fired"])
        good = diff <= a.tolerance and close
        ok &= good
        r = {"max_score_diff": round(diff, 6), "detections": len(fired), "golden_detections": len(g["fired"]),
             "detections_moved": moved, "pass": good, "threshold": thr}
        if a.bench:
            r["bench"] = bench(g["ref"], pcm)
        res["models"][name] = r
        print(f"{name:24} max|score - golden| {diff:.2e}  detections {len(fired)}/{len(g['fired'])}  "
              f"{'PASS' if good else 'FAIL'}" + (f"  {r['bench']['cpu_ms_per_s']} ms CPU per audio s" if a.bench else ""))
    if a.bench:
        res["openwakeword_hey_jarvis"] = bench_openwakeword(pcm)
        print("openWakeWord hey_jarvis (onnx)", res["openwakeword_hey_jarvis"])
    print(json.dumps(res["info"]))
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a") as fh:
            fh.write(f"### {res['info']['cpu']} · {res['info']['machine']} · Python {res['info']['python']} · "
                     f"onnxruntime {res['info']['onnxruntime']} {res['info']['qemu_cpu']}\n\n"
                     "| model | max score diff | detections | result | CPU ms per audio s |\n|---|---|---|---|---|\n")
            for name, r in res["models"].items():
                fh.write(f"| {name} | {r['max_score_diff']:.1e} | {r['detections']}/{r['golden_detections']} | "
                         f"{'pass' if r['pass'] else 'FAIL'} | {r.get('bench', {}).get('cpu_ms_per_s', '')} |\n")
            if res.get("openwakeword_hey_jarvis"):
                fh.write(f"| openWakeWord hey_jarvis (onnx) | | | | {res['openwakeword_hey_jarvis']['cpu_ms_per_s']} |\n")
            fh.write("\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
