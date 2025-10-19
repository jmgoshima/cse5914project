import React, { useState } from "react";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! How can I help you today?" },
  ]);

  const [isTyping, setIsTyping] = useState(false);

  const handleSend = (message) => {
    setMessages((prev) => [...prev, { sender: "user", text: message }]);
    setIsTyping(true);

    setTimeout(() => {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "I'm just a demo bot 🤖" },
      ]);
      setIsTyping(false);
    }, 1000);
  };

  return (
    <div className="chat-container">
      <ChatWindow messages={messages} isTyping={isTyping} />
      <ChatInput onSend={handleSend} />
    </div>
  );
}

export default App;