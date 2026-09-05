import { STEPS } from "../data.js";

export default function Pipeline() {
  return (
    <section className="block">
      <div className="kicker">how it works</div>
      <h2>Four steps, all local</h2>
      <p className="lede">
        No account, no cloud call, no telemetry. Everything happens against your own Ollama
        install.
      </p>
      <div className="pipeline">
        {STEPS.map((step, i) => (
          <div className="step" key={step.title}>
            <span className="n">{i + 1}</span>
            <div className="body">
              <h3>{step.title}</h3>
              <p>{step.body}</p>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
