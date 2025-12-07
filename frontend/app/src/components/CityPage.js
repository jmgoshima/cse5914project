// src/CityPage.js
import React from "react";
import CityRecommendations from "./CityRecommendations";

export default function CityPage({ result, onBack }) {
  return (
    <div className="page">
      <button class="back-button">← Back</button>

      <CityRecommendations
        isWaiting={false}
        result={result}
      />
    </div>
  );
}