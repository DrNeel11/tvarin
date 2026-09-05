import BenchmarkChart from "./BenchmarkChart.jsx";
import EmailForm from "./EmailForm.jsx";
import { BENCH } from "../data.js";

export default function Hero() {
  return (
    <section className="hero">
      <span className="eyebrow">for 8–16GB consumer GPUs running Ollama</span>
      <h1>Get more performance from the VRAM you already have.</h1>
      <p className="sub">
        Ramer finds a faster GPU/CPU layer split for your local models automatically, and
        reuses it — instead of trusting Ollama's conservative default or guessing{" "}
        <code>num_gpu</code> by hand.
      </p>
      <BenchmarkChart bench={BENCH} />
      <EmailForm />
    </section>
  );
}
