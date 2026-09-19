export default function ViewToggle({ view, onChange }) {
  return (
    <div className="view-toggle" role="tablist" aria-label="Choose a view">
      <button
        className={view === "customer" ? "active" : ""}
        type="button"
        role="tab"
        aria-selected={view === "customer"}
        onClick={() => onChange("customer")}
      >
        Customer Checkout View
      </button>
      <button
        className={view === "retailer" ? "active" : ""}
        type="button"
        role="tab"
        aria-selected={view === "retailer"}
        onClick={() => onChange("retailer")}
      >
        Retailer View
      </button>
    </div>
  );
}