import { useState } from "react";
import RiskForm from "./RiskForm";
import Recommendations from "./Recommendations";

export default function App() {
  const [result, setResult] = useState(null);

  const shouldRecommend = result?.risk_label === "Medium" || result?.risk_label === "High";

  return (
    <main className="page-shell">
      <section className="intro">
        <p className="eyebrow">FASHION RETURNS / RISK CHECK</p>
        <h1>Make the next order a little more certain.</h1>
        <p className="intro-copy">
          Enter the details you would normally know at checkout. The model estimates return risk
          and surfaces lower-return alternatives when the signal is worth a second look.
        </p>
      </section>

      <RiskForm onResult={setResult} />

      {result && (
        <section className="result-panel" aria-live="polite">
          <div>
            <p className="eyebrow">MODEL RESULT</p>
            <h2>Estimated return risk</h2>
          </div>
          <div className="risk-summary">
            <strong>{(result.return_probability * 100).toFixed(1)}%</strong>
            <span className={`risk-badge risk-${result.risk_label.toLowerCase()}`}>
              {result.risk_label}
            </span>
          </div>
        </section>
      )}

      {shouldRecommend && (
        <Recommendations
          category={result.category}
          fabric={result.fabric}
          fit_type={result.fit_type}
          price_usd={result.price_usd}
        />
      )}
    </main>
  );
}