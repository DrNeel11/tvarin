export default function Caveats() {
  return (
    <section className="block" id="caveats">
      <div className="kicker">read this before joining</div>
      <h2>What Ramer doesn't claim</h2>
      <div className="manpage">
        <div className="head">
          <span>RAMER(1)</span>
          <span>CAVEATS</span>
        </div>
        <dl>
          <dt>DOES NOT CREATE VRAM</dt>
          <dd>
            Ramer finds unused headroom inside the hardware you already have. It does not add
            memory, and it does not make a model that genuinely doesn't fit suddenly fit.
          </dd>
          <dt>NOT FOR EVERY MODEL SIZE</dt>
          <dd>
            Tested directly against a 32B model on this same 8GB card: best achievable
            placement was only +6.7% over Ollama's default, both too slow to use interactively.
            Past a certain size, no placement helps — that's a hardware limit, not a tuning
            problem.
          </dd>
          <dt>GAIN VARIES RUN TO RUN</dt>
          <dd>
            In real (not staged) usage, the measured gain over Ollama's default ranged from
            about −20% to +78% depending on what else was using the GPU at the time. It is a
            real average win, not a guaranteed one on any single request.
          </dd>
        </dl>
      </div>
    </section>
  );
}
