import React, { useState } from "react";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([
    { sender: "bot", text: "Hello! How can I help you today?" },
  ]);

  const [isTyping, setIsTyping] = useState(false);

  const handleSend = async (message) => {
    // Add user message
    setMessages((prev) => [...prev, { sender: "user", text: message }]);

    // show typing indicator
    setIsTyping(true);

    try {
      // Example call to your backend (or any test API)
      const response = await fetch("https://api.adviceslip.com/advice");
      const data = await response.json();

      // Use the advice as the bot's message
      const botReply = data.slip.advice;

      setMessages((prev) => [...prev, { sender: "bot", text: botReply }]);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Sorry, something went wrong 😢" },
      ]);
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className="chat-container">
      <ChatWindow messages={messages} isTyping={isTyping} />
      <ChatInput onSend={handleSend} />
    </div>
  );
}

export default App;