export const BENCH = {
  gpu: "RTX 5060 · 8GB VRAM",
  model: "Qwen2.5-14B-Instruct (Q4_K_M)",
  scaleMax: 20,
  rows: [
    { key: "baseline", label: "Ollama (auto)", value: 12.2 },
    { key: "ramer", label: "Ramer", value: 19.6 },
  ],
  deltaPct: "+61%",
};

export const STEPS = [
  {
    title: "Detect your GPU",
    body: "Reads real VRAM capacity and what's already using it — including whatever else is currently loaded.",
  },
  {
    title: "Test safe placements",
    body: "Runs a short real generation at several GPU/CPU splits, watching for the throughput cliff that shows up once a split overshoots physical VRAM.",
  },
  {
    title: "Pick the fastest one",
    body: "Not the split that looks safest on paper — the one that actually measured fastest on your hardware, right now.",
  },
  {
    title: "Reuse it, safely",
    body: "Caches the result. Before trusting it again, a quick check confirms it still holds — if something else has claimed the VRAM since, it falls back to Ollama's own safe default instead of guessing.",
  },
];

export const ISSUES = [
  {
    id: "#7629",
    quote:
      "“llama.cpp achieves 7.2 to 7.5 GB [utilization]… Ollama only reaches approximately 6GB” — on an RTX 2070, 8GB.",
    meta: "Opened Nov 2024",
    url: "https://github.com/ollama/ollama/issues/7629",
  },
  {
    id: "#3837",
    quote:
      "“The GPU vram fills ~7GB but the compute remains at 0-1%… the model is running purely on the CPU” — on an RTX 3070, 8GB.",
    meta: null,
    url: "https://github.com/ollama/ollama/issues/3837",
  },
];
