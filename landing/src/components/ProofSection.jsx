import { ISSUES } from "../data.js";

export default function ProofSection() {
  return (
    <section className="block">
      <div className="kicker">you're not imagining it</div>
      <h2>This gap is already reported — just not fixed</h2>
      <p className="lede">
        Independent reports from Ollama's own issue tracker, on similarly constrained hardware:
      </p>
      <div className="proof">
        {ISSUES.map((issue) => (
          <div className="issue" key={issue.id}>
            <span className="id">{issue.id}</span>
            <div>
              <blockquote>{issue.quote}</blockquote>
              <div className="meta">
                {issue.meta ? `${issue.meta} · ` : null}
                <a href={issue.url} target="_blank" rel="noopener noreferrer">
                  {issue.url.replace("https://", "")}
                </a>
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
