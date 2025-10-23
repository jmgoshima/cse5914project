// import React, { useState } from "react";
// import ChatWindow from "./components/ChatWindow";
// import ChatInput from "./components/ChatInput";
// import "./App.css";

// function App() {
//   const [messages, setMessages] = useState([
//     { sender: "bot", text: "Hello! How can I help you today?" },
//   ]);

//   const [isTyping, setIsTyping] = useState(false);

//   const handleSend = async (message) => {
//     // Add user message
//     setMessages((prev) => [...prev, { sender: "user", text: message }]);

//     // show typing indicator
//     setIsTyping(true);

//     try {
//       // Example call to your backend (or any test API)
//       const response = await fetch("https://api.adviceslip.com/advice");
//       const data = await response.json();

//       // Use the advice as the bot's message
//       const botReply = data.slip.advice;

//       setMessages((prev) => [...prev, { sender: "bot", text: botReply }]);
//     } catch (err) {
//       console.error(err);
//       setMessages((prev) => [
//         ...prev,
//         { sender: "bot", text: "Sorry, something went wrong 😢" },
//       ]);
//     } finally {
//       setIsTyping(false);
//     }
//   };

//   return (
//     <div className="chat-container">
//       <ChatWindow messages={messages} isTyping={isTyping} />
//       <ChatInput onSend={handleSend} />
//     </div>
//   );
// }

// export default App;

import React, { useState, useEffect } from "react";
import ChatWindow from "./components/ChatWindow";
import ChatInput from "./components/ChatInput";
import "./App.css";

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
        const response = await fetch("http://localhost:5000/api/initialize", {
          method: "POST",
        });
        const data = await response.json();
        setConversationID(data.conversationID);
        console.log("Conversation initialized:", data.conversationID);
      } catch (err) {
        console.error("Initialization failed:", err);
      }
    };

    initializeConversation();
  }, []);

  const handleSend = async (message) => {
    // Add user's message to chat window
    setMessages((prev) => [...prev, { sender: "user", text: message }]);
    setIsTyping(true);

    try {
      const response = await fetch("http://localhost:5000/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          conversationID: conversationID,
          userInput: message,
        }),
      });

      const data = await response.json();

      // Assuming backend returns: { reply: "...", metadata: {...} }
      const botReply = data.reply || "Hmm, I didn’t understand that.";

      setMessages((prev) => [...prev, { sender: "bot", text: botReply }]);
    } catch (err) {
      console.error("Chat request failed:", err);
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
