import React, { useState, useEffect } from "react";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import CityRecommendations from "./components/CityRecommendations";
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
  const [conversationID, setConversationID] = useState(null);

  // Initialize conversation when app starts
  useEffect(() => {
    const initializeConversation = async () => {
      try {
        const response = await fetch("http://localhost:8080/api/initialize", {
          method: "POST",
        });
        const data = await response.json();

        // Correctly read the nested conversationId
        const convId = data?.data?.conversationId;
        if (!convId) {
          console.error("No conversationId returned:", data);
          return;
        }

        setConversationID(convId);
        console.log("Conversation initialized:", convId);
      } catch (err) {
        console.error("Initialization failed:", err);
      }
    };

    initializeConversation();
  }, []);
  const [cityResult, setCityResult] = useState(null);
  const [isWaitingForCity, setIsWaitingForCity] = useState(false);

  const handleSend = async (message) => {
    if (!message.trim()) {
      return;
    }

    if (!conversationID) {
      console.error("No conversation ID yet; cannot send message.");
      return;
    }

    // Add user's message to chat window
    setMessages((prev) => [...prev, { sender: "user", text: message }]);
    setIsTyping(true);
    setIsWaitingForCity(true);

    try {
      const response = await fetch("http://localhost:8080/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          conversationId: conversationID, // matches backend key
          message: message,
        }),
      });

      const data = await response.json();
      // backend returns response in data.response
      const botReply = data?.data?.profile.notes.next_question || "Hmm, I didn’t understand that.";

      const recommendationPayload =
        data?.city_recommendations || data?.data?.city_recommendations || data;

      if (
        recommendationPayload?.raw_output ||
        recommendationPayload?.cities ||
        recommendationPayload?.profile_payload
      ) {
        setCityResult({
          header: recommendationPayload.header,
          raw_output: recommendationPayload.raw_output,
          cities: recommendationPayload.cities,
          profile_payload: recommendationPayload.profile_payload,
        });
      } else {
        setCityResult(null);
      }

      setMessages((prev) => [...prev, { sender: "bot", text: botReply }]);
    } catch (err) {
      console.error("Chat request failed:", err);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Sorry, something went wrong. Try again!" },
      ]);
    } finally {
      setIsTyping(false);
      setIsWaitingForCity(false);
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

        <section className="dashboard">
          <div className="how-to-row">
            <div className="how-to-heading">
              <h2>How to get the most out of it</h2>
              <p>Three quick steps to start a productive session.</p>
            </div>
            <ol className="how-to-steps">
              {HOW_TO_STEPS.map((step, index) => (
                <li key={index} className="how-to-step">
                  <span className="step-number">{index + 1}</span>
                  <p>{step}</p>
                </li>
              ))}
            </ol>
          </div>

          <div className="workspace">
            <div className="workspace-panel chat-panel">
              <header className="panel-header">
                <div>
                  <h2>Preview the chat experience</h2>
                  <p>
                    We&apos;re still polishing the interface, but you can try
                    the advice powered demo below.
                  </p>
                </div>
              </header>
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

            <div className="workspace-panel results-panel">
              <CityRecommendations
                isWaiting={isWaitingForCity && !cityResult}
                result={cityResult}
              />
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
