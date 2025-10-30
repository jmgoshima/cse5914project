import React, { useState } from "react";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import CityRecommendations from "./components/CityRecommendations";
import "./App.css";

const HOW_TO_STEPS = [
  "Type a clear question about your capstone or research task.",
  "Review the assistant's reply and any suggested resources.",
  "Ask follow-up questions until you have next steps you're ready to act on.",
];

function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! How can I help you today?" },
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [cityResult, setCityResult] = useState(null);
  const [isWaitingForCity, setIsWaitingForCity] = useState(false);

  const handleSend = async (message) => {
    if (!message.trim()) {
      return;
    }

    setMessages((prev) => [...prev, { sender: "user", text: message }]);
    setIsTyping(true);
    setIsWaitingForCity(true);

    try {
      const response = await fetch("https://api.adviceslip.com/advice");
      const data = await response.json();
      const botReply =
        data?.slip?.advice ||
        data?.message ||
        data?.data?.message ||
        "Here's something to consider.";

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
      console.error(err);
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
