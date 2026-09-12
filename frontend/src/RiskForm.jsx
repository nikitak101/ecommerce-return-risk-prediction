import { useState } from "react";
import { API_BASE_URL } from "./config";

const defaults = {
  marketplace_region: "US",
  category: "dress",
  fabric: "cotton",
  price_usd: 45,
  discount_pct: 0.1,
  is_premium: false,
  size_ordered: 3,
  size_usual: 2,
  ordered_multiple_sizes: false,
  fit_type: "slim",
  customer_prior_orders: 2,
  customer_prior_return_rate: 0.25,
  avg_review_rating: 4.2,
  num_reviews: 120,
  reviews_read: 10,
  size_chart_viewed: true,
  model_shown: true,
  is_gift: false,
  device: "mobile",
  days_to_delivery: 4,
  return_shipping_free: true,
};

const enums = {
  marketplace_region: ["US", "UK", "DE", "FR", "ES", "IT", "NL", "PL", "CN"],
  category: ["dress", "accessories", "activewear", "jacket", "jeans", "knitwear", "lingerie", "shoes", "t_shirt", "trousers"],
  fabric: ["cotton", "blend", "cotton_stretch", "denim", "jersey", "linen", "polyester", "silk", "wool"],
  fit_type: ["slim", "regular", "oversized", "true_to_size"],
  device: ["mobile", "desktop", "tablet"],
};

const labels = {
  marketplace_region: "Marketplace region",
  category: "Category",
  fabric: "Fabric",
  price_usd: "Price (USD)",
  discount_pct: "Discount (decimal)",
  is_premium: "Premium item",
  size_ordered: "Size ordered",
  size_usual: "Usual size",
  ordered_multiple_sizes: "Ordered multiple sizes",
  fit_type: "Fit type",
  customer_prior_orders: "Prior orders",
  customer_prior_return_rate: "Prior return rate",
  avg_review_rating: "Average review rating",
  num_reviews: "Number of reviews",
  reviews_read: "Reviews read",
  size_chart_viewed: "Size chart viewed",
  model_shown: "Model shown",
  is_gift: "Gift order",
  device: "Device",
  days_to_delivery: "Days to delivery",
  return_shipping_free: "Free return shipping",
};

const booleanFields = new Set([
  "is_premium",
  "ordered_multiple_sizes",
  "size_chart_viewed",
  "model_shown",
  "is_gift",
  "return_shipping_free",
]);

const integerFields = new Set([
  "size_ordered",
  "size_usual",
  "customer_prior_orders",
  "num_reviews",
  "reviews_read",
  "days_to_delivery",
]);

function bucketRisk(probability) {
  if (probability < 0.3) return "Low";
  if (probability <= 0.6) return "Medium";
  return "High";
}

export default function RiskForm({ onResult }) {
  const [form, setForm] = useState(defaults);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateField(event) {
    const { name, type, value, checked } = event.target;
    setForm((current) => ({
      ...current,
      [name]: type === "checkbox" ? checked : value,
    }));
  }

  async function submit(event) {
    event.preventDefault();
    setLoading(true);
    setError("");

    const payload = Object.fromEntries(
      Object.entries(form).map(([name, value]) => [
        name,
        booleanFields.has(name)
          ? Boolean(value)
          : integerFields.has(name)
            ? Number.parseInt(value, 10)
            : ["price_usd", "discount_pct", "customer_prior_return_rate", "avg_review_rating"].includes(name)
              ? Number.parseFloat(value)
              : value,
      ]),
    );

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) throw new Error(`Prediction failed (${response.status})`);
      const data = await response.json();
      const risk_label = bucketRisk(data.return_probability);
      onResult({
        category: payload.category,
        fabric: payload.fabric,
        fit_type: payload.fit_type,
        price_usd: payload.price_usd,
        return_probability: data.return_probability,
        risk_label,
      });
    } catch (requestError) {
      setError(requestError.message || "Could not reach the prediction service.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <form className="risk-form" onSubmit={submit}>
      <div className="form-heading">
        <div>
          <p className="eyebrow">ORDER DETAILS</p>
          <h2>What are you ordering?</h2>
        </div>
        <span className="field-count">21 fields</span>
      </div>
      <div className="field-grid">
        {Object.keys(defaults).map((name) => (
          <label className={booleanFields.has(name) ? "field checkbox-field" : "field"} key={name}>
            {booleanFields.has(name) ? (
              <>
                <input type="checkbox" name={name} checked={form[name]} onChange={updateField} />
                <span>{labels[name]}</span>
              </>
            ) : (
              <>
                <span>{labels[name]}</span>
                {enums[name] ? (
                  <select name={name} value={form[name]} onChange={updateField}>
                    {enums[name].map((option) => <option key={option} value={option}>{option}</option>)}
                  </select>
                ) : (
                  <input
                    name={name}
                    type="number"
                    value={form[name]}
                    onChange={updateField}
                    min={name.includes("size_") ? 0 : undefined}
                    max={name.includes("size_") ? 6 : undefined}
                    step={name === "discount_pct" || name === "customer_prior_return_rate" ? "0.01" : name === "avg_review_rating" ? "0.1" : "any"}
                    required
                  />
                )}
              </>
            )}
          </label>
        ))}
      </div>
      {error && <p className="error-message">{error}</p>}
      <button className="submit-button" type="submit" disabled={loading}>
        {loading ? "Checking risk..." : "Check return risk"}
      </button>
    </form>
  );
}