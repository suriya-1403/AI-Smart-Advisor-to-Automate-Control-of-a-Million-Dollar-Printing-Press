import React, { useState } from 'react';
import ChatBot from './components/ChatBot';
import './App.css';

function App() {
  const [showInfo, setShowInfo] = useState(false);

  return (
    <div className="App">
      <ChatBot />
    </div>
  );
}

export default App;
