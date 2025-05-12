import React, { useState, useRef, useEffect } from 'react';
import './ChatBot.css';

const ChatBot = () => {
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const chatMessagesRef = useRef(null);
  const textareaRef = useRef(null);
  const [documents, setDocuments] = useState([]);
  const [documentResultText, setDocumentResultText] = useState('');

  // API endpoint configuration
  const API_URL = 'http://localhost:8000/query';

  const suggestions = [
    {
      title: 'Find documents',
      description: 'with heavy ink coverage on coated media around 220 gsm'
    },
    {
      title: 'What\'s the recommended dryer power',
      description: 'and press speed for a coated glossy sheet with 50 GSM, 57% ink coverage, and an optical density of 96? Use the Quality mode.'
    },
    {
      title: 'Tell me about',
      description: 'event id 71?'
    },
    {
      title: 'Can you explain',
      description: 'the difference between coated and uncoated media?'
    }
  ];

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      const chatMessages = chatMessagesRef.current;
      const isScrolledToBottom = chatMessages.scrollHeight - chatMessages.clientHeight <= chatMessages.scrollTop + 100;

      if (isScrolledToBottom) {
        messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
      }
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = Math.min(textareaRef.current.scrollHeight, 200) + 'px';
    }
  }, [inputMessage]);

  // New function to fetch response from API
  const fetchResponse = async (query) => {
    setIsTyping(true);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      });

      const data = await response.json();

      if (data.status === 'success' && data.result) {
        // Format the response based on the result type
        let formattedResponse = '';

        if (data.result.type === 'document_search') {
          formattedResponse = `
### Documents Found
${data.result.document}

### Summary
${data.result.summary}`;
          setDocumentResultText(data.result.document);
        } else if (data.result.type === 'ruleset_evaluation') {
          formattedResponse = `
### Ruleset Evaluation
${data.result.explanation}`;
        } else {
          formattedResponse = "The server returned an unexpected response format.";
        }

        setMessages(prev => [...prev, {
          type: 'bot',
          content: formattedResponse,
          timestamp: new Date()
        }]);
      } else {
        // Handle error response
        setMessages(prev => [...prev, {
          type: 'bot',
          content: `Sorry, I encountered an error: ${data.message || 'Unknown error'}`,
          timestamp: new Date()
        }]);
      }
    } catch (error) {
      console.error('Error fetching response:', error);
      setMessages(prev => [...prev, {
        type: 'bot',
        content: `Sorry, I wasn't able to connect to the server. Please try again later.`,
        timestamp: new Date()
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const trimmedMessage = inputMessage.trim();
    if (!trimmedMessage) return;

    const userMessage = {
      type: 'user',
      content: trimmedMessage,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputMessage('');
    if (textareaRef.current) {
      textareaRef.current.style.height = '52px';
    }

    // Use real API instead of simulated response
    await fetchResponse(trimmedMessage);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  const handleSuggestionClick = async (suggestion) => {
    const fullQuestion = `${suggestion.title} ${suggestion.description}`;
    const userMessage = {
      type: 'user',
      content: fullQuestion,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    // Use real API instead of simulated response
    await fetchResponse(fullQuestion);
  };

  const formatTime = (date) => {
    return new Intl.DateTimeFormat('en-US', {
      hour: '2-digit',
      minute: '2-digit'
    }).format(date);
  };

  // Function to render message content with Markdown-like formatting
  const renderMessageContent = (content) => {
    // Simple Markdown parsing for headings and line breaks
    const formattedContent = content
      .replace(/^### (.*$)/gm, '<h3>$1</h3>')
      .replace(/\n\n/g, '<br/><br/>')
      .replace(/\n/g, '<br/>');

    return <div dangerouslySetInnerHTML={{ __html: formattedContent }} />;
  };

  function parseDocuments(rawText) {
    // Split by 'Result' and filter out empty strings
    const results = rawText.split(/Result \d+:/).filter(Boolean);
    return results.map(result => {
      const lines = result.split('\n').map(line => line.trim()).filter(Boolean);
      const doc = {};
      lines.forEach(line => {
        if (line.startsWith('Event ID:')) doc.eventId = line.replace('Event ID:', '').trim();
        else if (line.startsWith('Date:')) doc.date = line.replace('Date:', '').trim();
        else if (line.startsWith('Location:')) doc.location = line.replace('Location:', '').trim();
        else if (line.startsWith('Press Model:')) doc.pressModel = line.replace('Press Model:', '').trim();
        else if (line.startsWith('Ink Coverage:')) doc.inkCoverage = line.replace('Ink Coverage:', '').trim();
        else if (line.startsWith('Media:')) doc.media = line.replace('Media:', '').trim();
      });
      return doc;
    });
  }

  useEffect(() => {
    if (documentResultText) {
      setDocuments(parseDocuments(documentResultText));
    }
  }, [documentResultText]);

  return (
    <div className={`chatbot-container ${messages.length > 0 ? 'chat-active' : ''}`}>
      <div className={`gradient-balls ${messages.length > 0 ? 'chat' : 'home'} ${isTyping ? 'thinking' : ''}`}>
        <div className="gradient-ball ball-1"></div>
        <div className="gradient-ball ball-2"></div>
        <div className="gradient-ball ball-3"></div>
      </div>

      {messages.length === 0 && (
        <div className="chatbot-header">
          <div className="header-title">
            <h1>Hello <span>there!</span></h1>
          </div>
          <p className="header-subtitle">
            Ask me anything about HP printing systems and media optimization
          </p>
        </div>
      )}

      <div className="chat-content">
        {messages.length === 0 ? (
          <div className="suggestions-container">
            <div className="suggestions-grid">
              {suggestions.map((suggestion, index) => (
                <button
                  key={index}
                  className="suggestion-card"
                  onClick={() => handleSuggestionClick(suggestion)}
                >
                  <div className="suggestion-title">{suggestion.title}</div>
                  <div className="suggestion-description">{suggestion.description}</div>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="messages-container" ref={chatMessagesRef}>
            <div className="chat-messages">
              {messages.map((message, index) => (
                <div
                  key={index}
                  className={`message ${message.type}-message`}
                >
                  <div className="message-content">
                    <div className="message-header">
                      <span className="message-sender">
                        {message.type === 'bot' ? 'HP Assistant' : 'You'}
                      </span>
                    </div>
                    {renderMessageContent(message.content)}
                    <div className="message-time">
                      {formatTime(message.timestamp)}
                    </div>
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="message bot-message">
                  <div className="message-content">
                    <div className="message-header">
                      <span className="message-sender">HP Assistant</span>
                    </div>
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              )}
            </div>
            <div ref={messagesEndRef} />
          </div>
        )}

        <div className="input-form-container">
          <form onSubmit={handleSubmit} className="input-form">
            <textarea
              ref={textareaRef}
              rows={1}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Send a message..."
              className="chat-input"
              aria-label="Chat input"
            />
            <div className="input-buttons">
              <button
                type="submit"
                className="input-button"
                aria-label="Send message"
                disabled={!inputMessage.trim()}
              >
                <svg viewBox="0 0 24 24" className="send-icon">
                  <path fill="currentColor" d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                </svg>
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default ChatBot;
