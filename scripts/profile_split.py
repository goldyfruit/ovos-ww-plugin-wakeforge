"""Where a wakeforge word's CPU goes on this machine, and what onnxruntime options change it.

Streams 60 s of noise through the OVOS plugin and splits the CPU between the streaming featurizer, the head and the
rest, then reruns with variants: 2 threads, the featurizer at ORT_ENABLE_BASIC, and on ARM64 the bf16 fast-math GEMM
(``mlas.enable_gemm_fastmath_arm64_bfloat16``), reporting each variant's largest score change against the default.
"""
import os
import platform
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import onnxruntime as ort

from ovos_plugin_manager.wakewords import OVOSWakeWordFactory
import ovos_ww_plugin_wakeforge.inference as inf

MODEL = "https://huggingface.co/thalovant/zoguilo-wakeword/resolve/main/zoguilo.onnx"
SECS = 60
pcm = (np.random.default_rng(0).standard_normal(SECS * 16000) * 0.05 * 32767).astype(np.int16)


def run(threads=1, extra=None, basic_featurizer=False):
    orig, orig_shared = inf.session_options, inf.shared_session

    def opts(t=1):
        o = orig(t)
        for k, v in (extra or {}).items():
            o.add_session_config_entry(k, v)
        return o
    inf.session_options = opts
    if basic_featurizer:
        inf.shared_session = lambda path, t=1, level=None, providers=("CPUExecutionProvider",): orig_shared(
            path, t, ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, providers)
    import ovos_ww_plugin_wakeforge.pretrained as pre
    pre.session_options, pre.shared_session = inf.session_options, inf.shared_session
    try:
        ww = OVOSWakeWordFactory.create_hotword("w", {"hotwords": {"w": {"module": "ovos-ww-plugin-wakeforge",
            "model": MODEL, "threshold": 1.01, "onnx_threads": threads}}, "listener": {"wake_word": "w"}})
    finally:
        inf.session_options, inf.shared_session = orig, orig_shared
        pre.session_options, pre.shared_session = orig, orig_shared
    eng = ww.engine
    t = {"trunk": 0.0, "head": 0.0}
    push, head = eng.stream.push, eng.head

    def tpush(x):
        a = time.perf_counter(); r = push(x); t["trunk"] += time.perf_counter() - a; return r

    class H:
        def run(self, *a, **k):
            b = time.perf_counter(); r = head.run(*a, **k); t["head"] += time.perf_counter() - b; return r

        def __getattr__(self, n):
            return getattr(head, n)
    eng.stream.push, eng.head = tpush, H()
    scores = []
    for i in range(0, 25 * 1280, 1280):
        ww.update(pcm[i:i + 1280].tobytes())
    t["trunk"] = t["head"] = 0.0
    c0, w0 = time.process_time(), time.perf_counter()
    for i in range(0, len(pcm) - 1279, 1280):
        ww.update(pcm[i:i + 1280].tobytes())
        scores.append(ww.last_score)
    cpu, wall = time.process_time() - c0, time.perf_counter() - w0
    return {"cpu": cpu / SECS * 1000, "wall": wall / SECS * 1000, "trunk": t["trunk"] / SECS * 1000,
            "head": t["head"] / SECS * 1000}, np.array(scores)


print(platform.platform(), platform.machine(), platform.processor(), "onnxruntime", ort.__version__)
base, ref = run()
print(f"default            CPU {base['cpu']:6.2f} ms/s (wall {base['wall']:.2f}) | featurizer {base['trunk']:.2f}  head {base['head']:.2f}  other {base['wall'] - base['trunk'] - base['head']:.2f}")
variants = [("2 threads", dict(threads=2)), ("featurizer BASIC", dict(basic_featurizer=True))]
if platform.machine().lower() in ("aarch64", "arm64"):
    variants.append(("bf16 fast-math GEMM", dict(extra={"mlas.enable_gemm_fastmath_arm64_bfloat16": "1"})))
for name, kw in variants:
    r, s = run(**kw)
    print(f"{name:18} CPU {r['cpu']:6.2f} ms/s (wall {r['wall']:.2f}) | featurizer {r['trunk']:.2f}  head {r['head']:.2f}  "
          f"| max score change {np.abs(s - ref).max():.1e}")
