import { useState } from "react";
import Recommendations from "./Recommendations";
import CustomerView from "./CustomerView";
import RetailerView from "./RetailerView";
import ViewToggle from "./ViewToggle";

export default function App() {
  const [result, setResult] = useState(null);
  const [view, setView] = useState("customer");

  const shouldRecommend = result?.risk_label === "Medium" || result?.risk_label === "High";

  return (
    <main className="page-shell">
      <ViewToggle view={view} onChange={setView} />

      {view === "customer" ? (
        <CustomerView result={result} onResult={setResult} />
      ) : (
        <RetailerView result={result} />
      )}

      {result && shouldRecommend && (
        <Recommendations
          customerFacing={view === "customer"}
          category={result.category}
          fabric={result.fabric}
          fit_type={result.fit_type}
          price_usd={result.price_usd}
        />
      )}
    </main>
  );
}