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

  if (Array.isArray(payload.cities)) {
    return payload.cities;
  }

  if (Array.isArray(payload?.data?.cities)) {
    return payload.data.cities;
  }

  if (Array.isArray(payload?.data?.data?.cities)) {
    return payload.data.data.cities;
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
