import React, { useState, useRef, useEffect } from 'react';
import './ChatBot.css';

// Move these functions to the top of the file, before the ChatBot component
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

function parseDocumentSearch(rawText) {
  // Extract the "Found X document(s)..." line
  const foundLineMatch = rawText.match(/Found \d+ document\(s\) matching your query:/);
  const foundLine = foundLineMatch ? foundLineMatch[0] : '';

  // Remove the found line for further parsing
  const rest = rawText.replace(foundLine, '').trim();

  // Extract summary section (everything after 'Summary')
  const summaryIndex = rest.indexOf('Summary');
  let summarySection = '';
  let mainSection = rest;
  if (summaryIndex !== -1) {
    summarySection = rest.slice(summaryIndex + 'Summary'.length).trim();
    mainSection = rest.slice(0, summaryIndex).trim();
  }

  // Parse documents as before
  const documents = parseDocuments(mainSection);

  // Parse summary as bullet points (lines starting with - or *)
  const summaryLines = summarySection
    .split('\n')
    .map(line => line.trim())
    .filter(line =>
      line.startsWith('-') ||
      line.startsWith('*') ||
      line.match(/^[A-Za-z].*:/) ||
      line.match(/^[0-9]+\./) // include numbered lines
    );
  // Remove leading - or * and extra spaces
  const summary = summaryLines.map(line => line.replace(/^[-*]\s?/, '').trim());

  return { foundLine, documents, summary };
}

const ChatBot = () => {
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const chatMessagesRef = useRef(null);
  const textareaRef = useRef(null);
  const [documents, setDocuments] = useState([]);
  const [documentResultText, setDocumentResultText] = useState('');
  const [summary, setSummary] = useState([]);
  const [foundLine, setFoundLine] = useState('');
  const requestStartTimeRef = useRef(null);

  // API endpoint configuration
  const API_URL = 'http://0.0.0.0:8000/query';

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
      console.log('Backend response:', data);
      const responseTime = requestStartTimeRef.current ? Date.now() - requestStartTimeRef.current : null;
      const modelName = data.result && data.result.model ? data.result.model : '';

      if (data.status === 'success' && data.result) {
        // Format the response based on the result type
        let formattedResponse = '';

        if (data.result.type === 'document_search') {
          formattedResponse = `
### Documents Found
${data.result.document}

### Summary
${data.result.summary}`;
          setDocumentResultText(`${data.result.document}\n\nSummary\n${data.result.summary}`);
        } else if (data.result.type === 'ruleset_evaluation') {
          formattedResponse = `
          ### Ruleset Evaluation

          **Report**
          ${data.result.report}

          **Explanation**
          ${data.result.explanation}`;
        } else if (data.result.type === 'event_information') {
          formattedResponse = `
### Event Information

${data.result.summary || data.result.documents || 'Event details retrieved successfully.'}`;
        } else if (data.result.type === 'general_knowledge') {
          formattedResponse = `
### Knowledge Response

${data.result.final_answer || data.result.knowledge_response || 'Knowledge response retrieved successfully.'}`;
        } else {
          formattedResponse = "The server returned an unexpected response format.";
        }

        setMessages(prev => [...prev, {
          type: 'bot',
          content: formattedResponse,
          timestamp: new Date(),
          responseTime,
          modelName
        }]);
        console.log('Bot message:', {
          responseTime,
          modelName,
          requestStartTime: requestStartTimeRef.current,
          now: Date.now()
        });
        requestStartTimeRef.current = null;
      } else {
        // Handle error response
        setMessages(prev => [...prev, {
          type: 'bot',
          content: `Sorry, I encountered an error: ${data.message || 'Unknown error'}`,
          timestamp: new Date(),
          responseTime,
          modelName
        }]);
        requestStartTimeRef.current = null;
      }
    } catch (error) {
      console.error('Error fetching response:', error);
      setMessages(prev => [...prev, {
        type: 'bot',
        content: `Sorry, I wasn't able to connect to the server. Please try again later.`,
        timestamp: new Date(),
        responseTime: requestStartTimeRef.current ? Date.now() - requestStartTimeRef.current : null,
        modelName: ''
      }]);
      requestStartTimeRef.current = null;
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

    requestStartTimeRef.current = Date.now(); // Use ref for timing
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
    requestStartTimeRef.current = Date.now(); // Use ref for timing
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

  useEffect(() => {
    if (documentResultText) {
      console.log('RAW:', documentResultText); // Debug: log the raw backend response
      const { foundLine, documents, summary } = parseDocumentSearch(documentResultText);
      setFoundLine(foundLine);
      setDocuments(documents);
      setSummary(summary);
    }
  }, [documentResultText]);

  // Add this component above ChatBot
  function DocumentResultsTable({ foundLine, documents, summary }) {
    return (
      <div className="results-table-container">
        {foundLine && <div className="results-found-line">{foundLine}</div>}
        <div className="results-table-scroll">
          <table className="results-table-pro">
            <thead>
              <tr>
                <th>Event ID</th>
                <th>Date</th>
                <th>Location</th>
                <th>Press Model</th>
                <th>Ink Coverage</th>
                <th>Media</th>
              </tr>
            </thead>
            <tbody>
              {documents.map((doc, idx) => (
                <tr key={idx}>
                  <td>{doc.eventId}</td>
                  <td>{doc.date}</td>
                  <td>{doc.location}</td>
                  <td>{doc.pressModel}</td>
                  <td>{doc.inkCoverage}</td>
                  <td>{doc.media}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {summary && summary.length > 0 && (
          <div className="summary-card-pro">
            <h4>Summary</h4>
            <ul>
              {summary.map((item, idx) => (
                <li key={idx} dangerouslySetInnerHTML={{ __html: item.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>') }} />
              ))}
            </ul>
          </div>
        )}
      </div>
    );
  }

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
                    {message.type === 'bot' && message.content.includes('Documents Found') && documents.length > 0 ? (
                      <DocumentResultsTable foundLine={foundLine} documents={documents} summary={summary} />
                    ) : (
                      renderMessageContent(message.content)
                    )}
                    <div className="message-time">
                      {formatTime(message.timestamp)}
                      {message.type === 'bot' && message.responseTime != null && (
                        <span style={{ marginLeft: 8, color:' rgba(255, 255, 255, 0.5)', fontSize: '0.75rem' }}>
                          {(message.responseTime / 1000).toFixed(2)} s
                        </span>
                      )}
                      {message.modelName && (
                            <span style={{ marginLeft: 8, color: 'rgb(0, 150, 214)', fontSize: '0.75rem' }}>
                              {message.modelName}
                            </span>
                          )}
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
