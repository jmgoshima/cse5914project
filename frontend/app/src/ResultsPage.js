// src/CityPage.js
import React from "react";
import CityRecommendations from "./components/CityRecommendations";

export default function CityPage({ result, onBack }) {
  return (
    <div className="page">
      <button onClick={onBack} style={{ margin: "20px" }}>← Back</button>

      <CityRecommendations
        isWaiting={false}
        result={result}
      />
    </div>
  );
}