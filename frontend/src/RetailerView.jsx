export default function RetailerView({ result }) {
  if (!result) {
    return (
      <section className="empty-state">
        <p className="eyebrow">RETAILER VIEW</p>
        <h1>Review order signals here.</h1>
        <p className="intro-copy">Submit an order in Customer Checkout View to see its internal assessment.</p>
      </section>
    );
  }

  return (
    <>
      <section className="intro compact-intro">
        <p className="eyebrow">RETAILER VIEW</p>
        <h1>Order assessment</h1>
        <p className="intro-copy">Review the current order signal and available alternatives.</p>
      </section>

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

      <section className="drivers-placeholder">
        <p className="eyebrow">RISK CONTEXT</p>
        <h2>Risk drivers: coming in Phase 2</h2>
        <p>Detailed explanations are not included in the current prediction response.</p>
      </section>
    </>
  );
}