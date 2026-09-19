import RiskForm from "./RiskForm";

export default function CustomerView({ result, onResult }) {
  return (
    <>
      <section className="intro">
        <p className="eyebrow">CHECKOUT</p>
        <h1>Make the next order a little more certain.</h1>
        <p className="intro-copy">
          Enter the details you would normally know at checkout and find options that may suit you even better.
        </p>
      </section>

      <RiskForm customerFacing onResult={onResult} />

      {result?.risk_label === "Low" && (
        <section className="positive-message" aria-live="polite">
          <h2>Great choice!</h2>
          <p>Your order looks like a good fit.</p>
        </section>
      )}
    </>
  );
}