import { useEffect, useState } from "react";
import { API_BASE_URL } from "./config";

export default function Recommendations({ category, fabric, fit_type, price_usd }) {
  const [recommendations, setRecommendations] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    setRecommendations(null);
    setError("");

    fetch(`${API_BASE_URL}/recommendations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ category, fabric, fit_type, price_usd }),
    })
      .then((response) => {
        if (!response.ok) throw new Error(`Recommendations failed (${response.status})`);
        return response.json();
      })
      .then((data) => {
        if (active) setRecommendations(data.recommendations || []);
      })
      .catch((requestError) => {
        if (active) setError(requestError.message || "Could not load alternatives.");
      });

    return () => { active = false; };
  }, [category, fabric, fit_type, price_usd]);

  return (
    <section className="recommendation-section" aria-live="polite">
      <div className="section-heading">
        <div>
          <p className="eyebrow">A LOWER-RETURN PATH</p>
          <h2>Alternatives worth comparing</h2>
        </div>
        <span className="section-note">Same category · more trustworthy history</span>
      </div>
      {recommendations === null && !error && <p className="muted-message">Finding comparable items...</p>}
      {error && <p className="error-message">{error}</p>}
      {recommendations?.length === 0 && <p className="muted-message">No safer alternatives found for this combination</p>}
      <div className="recommendation-grid">
        {recommendations?.map((item) => (
          <article className="recommendation-card" key={item.virtual_product_id}>
            <div className="card-topline">
              <span>{item.category}</span>
              <span className="return-rate">{(item.return_rate * 100).toFixed(1)}% returns</span>
            </div>
            <h3>{item.fabric.replaceAll("_", " ")}</h3>
            <p className="fit-label">{item.fit_type.replaceAll("_", " ")} fit</p>
            <dl>
              <div><dt>Average price</dt><dd>${item.avg_price.toFixed(2)}</dd></div>
              <div><dt>Average rating</dt><dd>{item.avg_rating.toFixed(1)} / 5</dd></div>
              <div><dt>Order history</dt><dd>{item.order_count} orders</dd></div>
            </dl>
          </article>
        ))}
      </div>
    </section>
  );
}