import React, { useEffect, useRef } from "react";

function ChatWindow({ messages, isTyping }) {
  const endRef = useRef(null);

  // Scroll to the bottom when messages change
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="chat-window">
      {messages.map((msg, index) => (
        <div
          key={index}
          className={`message-row ${
            msg.sender === "user" ? "user-row" : "bot-row"
          }`}
        >
          {msg.sender === "bot" && (
            <div className="avatar bot-avatar">🤖</div>
          )}
          <div
            className={`message-bubble ${
              msg.sender === "user" ? "user-bubble" : "bot-bubble"
            }`}
          >
            <p>{msg.text}</p>
            <span className="timestamp">
              {new Date().toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })}
            </span>
          </div>
          {msg.sender === "user" && (
            <div className="avatar user-avatar">🧑</div>
          )}
        </div>
      ))}
      {isTyping && (
        <div className="message-row bot-row">
          <div className="avatar bot-avatar">🤖</div>
          <div className="message-bubble bot-bubble typing">
            <span>.</span><span>.</span><span>.</span>
          </div>
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}

export default ChatWindow;