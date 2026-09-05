export default function BenchmarkChart({ bench }) {
  const ticks = [0, 5, 10, 15, 20];

  return (
    <div className="bench">
      <div className="bench-head">
        <span className="bench-title">
          measured — {bench.gpu} · {bench.model}
        </span>
        <span className="bench-delta">{bench.deltaPct}</span>
      </div>

      <div className="chart">
        {bench.rows.map((row) => (
          <div className={`row ${row.key}`} key={row.key}>
            <span className="label">{row.label}</span>
            <div className="track">
              <div
                className="fill"
                style={{ width: `${(row.value / bench.scaleMax) * 100}%` }}
              />
            </div>
            <span className="value">{row.value.toFixed(1)} tok/s</span>
          </div>
        ))}
      </div>

      <div className="scale">
        <span />
        <div className="ticks">
          {ticks.map((t, i) => (
            <span
              className="tick"
              key={t}
              style={{ left: `${(t / bench.scaleMax) * 100}%` }}
            >
              {i === ticks.length - 1 ? `${t} tok/s` : t}
            </span>
          ))}
        </div>
        <span />
      </div>

      <p className="bench-caption">
        Same model, same card, same request. Full sweep and methodology in the{" "}
        <a href="#caveats">what Ramer doesn't claim</a> section below.
      </p>
    </div>
  );
}
