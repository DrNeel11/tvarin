import { useState } from "react";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function EmailForm() {
  const [email, setEmail] = useState("");
  const [status, setStatus] = useState("idle"); // idle | joined | error

  function handleSubmit(e) {
    e.preventDefault();
    if (!EMAIL_RE.test(email)) {
      setStatus("error");
      return;
    }
    setStatus("joined");
  }

  return (
    <div className="cta">
      <form onSubmit={handleSubmit}>
        <input
          type="email"
          placeholder="you@example.com"
          aria-label="Email address"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          disabled={status === "joined"}
        />
        <button type="submit" disabled={status === "joined"}>
          {status === "joined" ? "Added." : "Join early access"}
        </button>
      </form>
      {status === "error" && (
        <p className="error">That doesn't look like an email address.</p>
      )}
      <p className="fineprint">One email when there's something to try. No pitch deck, no spam.</p>
    </div>
  );
}
