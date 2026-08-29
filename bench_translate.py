"""Benchmark: Hy-MT translation speed GPU vs CPU."""
import time
import llama_cpp

MODEL = r"G:\work\hermes\vvp-translate\.models\HY-MT1.5-1.8B-Q4_K_M.gguf"
PROMPT = ("Translate the following English segment into Russian, without additional explanation.\n\n"
          "We are going to explore the mysterious caverns beneath the ancient castle, "
          "and I believe that somewhere down there we will finally find the lost treasure.")


def bench(llm, n=5):
    llm(PROMPT, max_tokens=128, temperature=0.3, echo=False)  # warmup
    t0 = time.perf_counter()
    outs = []
    for _ in range(n):
        out = llm(PROMPT, max_tokens=128, temperature=0.3, echo=False)
        outs.append("".join(c["text"] for c in out["choices"]))
    dt = (time.perf_counter() - t0) / n
    toks = sum(o.get("completion_tokens", 0) for o in [])  # not exposed; estimate
    return dt, outs[-1]


print("=" * 60)
print("GPU (n_gpu_layers=-1)")
llm_gpu = llama_cpp.Llama(model_path=MODEL, n_ctx=2048, n_gpu_layers=-1, verbose=False)
dt_g, sample = bench(llm_gpu)
print(f"  {dt_g:.2f} s/segment   sample: {sample[:80]!r}")
del llm_gpu

print("-" * 60)
print("CPU (n_gpu_layers=0)")
llm_cpu = llama_cpp.Llama(model_path=MODEL, n_ctx=2048, n_gpu_layers=0, verbose=False)
dt_c, _ = bench(llm_cpu)
print(f"  {dt_c:.2f} s/segment")
del llm_cpu

print("=" * 60)
print(f"SPEEDUP: {dt_c / dt_g:.1f}x")
