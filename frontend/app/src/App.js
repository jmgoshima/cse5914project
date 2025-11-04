import React, { useState } from "react";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import "./App.css";

const HOW_TO_STEPS = [
  "Type a clear question about your capstone or research task.",
  "Review the assistant's reply and any suggested resources.",
  "Ask follow-up questions until you have next steps you're ready to act on.",
];

const extractCityRecommendations = (payload) => {
  if (!payload || typeof payload !== "object") {
    return [];
  }

  const normalizeEntry = (entry, idx) => {
    if (!entry || typeof entry !== "object") {
      return null;
    }
    const cityName =
      entry.city ||
      entry.name ||
      entry.title ||
      entry.label ||
      `City ${idx + 1}`;
    const reasonText =
      entry.reason ||
      entry.summary ||
      entry.description ||
      entry.notes ||
      entry.details ||
      "";
    const identifier =
      entry.id ||
      entry.city ||
      entry.name ||
      `${cityName}-${idx}`;
    return {
      id: identifier,
      name: cityName,
      reason: reasonText,
      score: entry.score ?? entry.match ?? null,
      raw: entry,
    };
  };

  const extractReasoningArray = (obj) => {
    if (obj && Array.isArray(obj.reasoning)) {
      return obj.reasoning;
    }
    return null;
  };

  const reasoningSources = [
    payload,
    payload?.data,
    payload?.data?.data,
  ];

  for (const source of reasoningSources) {
    const entries = extractReasoningArray(source);
    if (entries) {
      return entries
        .map((entry, idx) => normalizeEntry(entry, idx))
        .filter(Boolean);
    }
  }

  const cityArrays = [
    payload.cities,
    payload?.data?.cities,
    payload?.data?.data?.cities,
  ].filter(Array.isArray);

  if (cityArrays.length > 0) {
    const cities = cityArrays[0];
    return cities
      .map((entry, idx) => {
        const normalized = normalizeEntry(entry, idx);
        if (normalized && !normalized.reason) {
          normalized.reason =
            entry.reason ||
            entry.summary ||
            entry.description ||
            entry.notes ||
            entry.explanation ||
            "";
        }
        if (normalized && !normalized.name) {
          normalized.name =
            entry.city ||
            entry.name ||
            `City ${idx + 1}`;
        }
        return normalized;
      })
      .filter(Boolean);
  }

  return [];
};

const isRecommendationReady = (payload) => {
  if (!payload || typeof payload !== "object") {
    return false;
  }

  if (typeof payload.ready === "boolean") {
    return payload.ready;
  }

  if (typeof payload?.data?.ready === "boolean") {
    return payload.data.ready;
  }

  return false;
};

function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! How can I help you today?" },
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [cityRecommendations, setCityRecommendations] = useState([]);

  const handleSend = async (message) => {
    if (!message.trim()) {
      return;
    }

    setMessages((prev) => [...prev, { sender: "user", text: message }]);
    setIsTyping(true);

    try {
      const response = await fetch("https://api.adviceslip.com/advice");
      const data = await response.json();
      const botReply =
        data?.slip?.advice ||
        data?.message ||
        data?.data?.message ||
        "Here's something to consider.";

      const recommendedCities = extractCityRecommendations(data);
      if (recommendedCities.length > 0) {
        setCityRecommendations(recommendedCities);
      } else if (isRecommendationReady(data)) {
        setCityRecommendations([]);
      }

      setMessages((prev) => [...prev, { sender: "bot", text: botReply }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Sorry, something went wrong. Try again!" },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="page">
      <div className="background-slideshow" aria-hidden="true">
        <span className="slide slide-one" />
        <span className="slide slide-two" />
        <span className="slide slide-three" />
      </div>

      <main className="home">
        <section className="hero">
          <span className="team-tag">Overly Trusting</span>
          <h1>Capstone AI Assistant</h1>
          <p>
            Join us in building a smarter project partner. Explore ideas, refine
            research, and capture next steps with confidence.
          </p>
        </section>

        <section className="content-grid">
          <div className="info-column">
            <div className="how-to-card">
              <h2>How to get the most out of it</h2>
              <ol>
                {HOW_TO_STEPS.map((step, index) => (
                  <li key={index}>
                    <span className="step-number">{index + 1}</span>
                    <p>{step}</p>
                  </li>
                ))}
              </ol>
            </div>

            <div className="recommendations-card">
              <h2>City recommendations</h2>
              <p className="recommendations-hint">
                Once the assistant finishes gathering details, your tailored
                cities will appear here.
              </p>
              {cityRecommendations.length > 0 ? (
                <ul className="city-list">
                  {cityRecommendations.map((city, idx) => {
                    const key = city?.id || `${city?.name ?? "city"}-${idx}`;
                    const reason =
                      city?.reason ||
                      city?.summary ||
                      city?.description ||
                      city?.notes ||
                      "Details coming soon.";
                    return (
                      <li key={key} className="city-item">
                        <div className="city-name">
                          {city?.name || "Unnamed city"}
                        </div>
                        {reason && <p className="city-reason">{reason}</p>}
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <div className="city-placeholder">
                  <span className="city-placeholder-dot" />
                  <span className="city-placeholder-text">
                    No recommendations yet - start chatting to unlock them.
                  </span>
                </div>
              )}
            </div>
          </div>

          <div className="chat-section">
            <h2>Preview the chat experience</h2>
            <p>
              We&apos;re still polishing the interface, but you can try the
              advice powered demo below.
            </p>
            <div className="chat-frame">
              <div className="chat-header">Assistant Demo</div>
              <div className="chat-body">
                <ChatWindow messages={messages} isTyping={isTyping} />
              </div>
              <div className="chat-footer">
                <ChatInput onSend={handleSend} />
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
