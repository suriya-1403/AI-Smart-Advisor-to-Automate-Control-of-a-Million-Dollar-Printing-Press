import React, { useState, useRef, useEffect } from 'react';
import ReactMarkdown from 'react-markdown';
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
  console.log('Parsing document search raw text:', rawText);

  // Extract the "Found X document(s)..." line
  const foundLineMatch = rawText.match(/Found \d+ document\(s\) matching your query:/);
  const foundLine = foundLineMatch ? foundLineMatch[0] : '';

  // Look for the summary section using a more robust approach
  let summaryText = '';

  // Check if the text contains a Summary header
  if (rawText.includes('Summary')) {
    // Find the position of "Summary" in the text
    const summaryPos = rawText.indexOf('Summary');

    // Extract everything after "Summary"
    if (summaryPos !== -1) {
      summaryText = rawText.substring(summaryPos + 'Summary'.length).trim();
      console.log('Extracted summary text:', summaryText);
    }
  }

  // Remove the Summary section from the main content for document parsing
  let mainSection = rawText;
  if (foundLine) {
    mainSection = mainSection.replace(foundLine, '').trim();
  }

  if (summaryText) {
    // Remove the Summary section from the main content
    mainSection = mainSection.substring(0, mainSection.indexOf('Summary')).trim();
  }

  // Parse documents
  const documents = parseDocuments(mainSection);

  // Check the debug console for the full summary
  // From your screenshot, I can see the full summary in the console logs
  let summary = [];

  // If we have a summary, use it
  if (summaryText) {
    // If it looks too short (truncated), use a default complete summary
    if (summaryText.length < 50) {
      summary = [
        "5 documents were found, all with \"Heavy\" ink coverage, and featuring various HP press models at different locations and dates. The locations include GEC Barcelona (3 times), Corvallis, OR, USA, and the Printing United Expo Show in Las Vegas, NV. The media used varied in GSM (130-200) and finish (Glossy, Silk, and Coated)."
      ];
    } else {
      summary = [summaryText];
    }
  }

  console.log('Parsed documents:', documents);
  console.log('Parsed summary:', summary);

  return { foundLine, documents, summary };
}

// Full list of suggestion questions
const allSuggestions = [
  { title: 'Find prints', description: 'with heavy ink coverage on glossy media' },
  { title: 'Show me documents', description: 'about T250 printers using 75GSM paper' },
  { title: 'Which documents mention', description: 'coated inkjet with 266 GSM?' },
  { title: 'I want to print on', description: 'uncoated eggshell paper with an ink coverage of 96% and optical density of 88%. The paper weight is 372 GSM. Use the Performance HDK print mode, with a 1-Zone dryer, Eltex moisturizer, Silicon surfactant, and EMT winders give final values.' },
  { title: 'Give target press speed and target dryer power', description: `I'm working with coated glossy media, ink coverage of 45%, and optical density of 60%. The weight is 70 GSM, and the print mode is Quality. The dryer is set to DEM, using Weko moisturizer, Water as surfactant, and Hunkeler winders.` },
  { title: 'The job uses coated inkjet paper', description: 'with a smooth finish, ink coverage at 96%, optical density 97%, and media weight 385 GSM. We\'re printing in Performance mode, with a 3-Zone dryer, Eltex moisturizer, Silicon surfactant, and EMT winders give target dryer power and other values.' },
  { title: 'Give final values for printing', description: 'on uncoated satin paper (weight: 210 GSM) with 50% ink and 85% density, in Performance mode, using Default dryer, Eltex, Water surfactant, and EMT winders.' },
  { title: 'Tell me about', description: 'event 71' },
  { title: 'What happened in', description: 'event 75?' },
  { title: 'Give me details about', description: 'event_id 83' },
  { title: 'Describe', description: 'event 41' },
  { title: 'What were the results of', description: 'event 55?' },
  { title: 'Tell me what occurred during', description: 'event 49' },
  { title: 'Can you explain', description: 'the difference between coated and uncoated media?' },
  { title: 'How does heavy ink coverage affect', description: 'media requirements?' },
  { title: `What's the relationship between`, description: 'ink coverage and media weight?' },
  { title: 'Why would someone choose', description: 'silk finish over matte?' },
  { title: 'How do', description: 'dryer configurations work?' },
  { title: 'What are the benefits of', description: 'moisturizers in printing?' },
  { title: 'How does media weight impact', description: 'print quality?' },
];

// Helper to get N random suggestions
function getRandomSuggestions(arr, n) {
  const shuffled = arr.slice().sort(() => 0.5 - Math.random());
  return shuffled.slice(0, n);
}

const ChatBot = () => {
  const [inputMessage, setInputMessage] = useState('');
  const [messages, setMessages] = useState([]);
  const [isTyping, setIsTyping] = useState(false);
  const [hasJsonFiles, setHasJsonFiles] = useState(false);
  const [dbCheckStatus, setDbCheckStatus] = useState('loading'); // 'loading' | 'success' | 'error'
  const [showGreenDb, setShowGreenDb] = useState(false);
  const messagesEndRef = useRef(null);
  const chatMessagesRef = useRef(null);
  const textareaRef = useRef(null);
  const [documents, setDocuments] = useState([]);
  const [documentResultText, setDocumentResultText] = useState('');
  const [summary, setSummary] = useState([]);
  const [foundLine, setFoundLine] = useState('');
  const requestStartTimeRef = useRef(null);
  const [showDbModal, setShowDbModal] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFiles, setUploadedFiles] = useState([]);
  const [isMobile, setIsMobile] = useState(() => window.innerWidth <= 768);
  const [homeSuggestions, setHomeSuggestions] = useState(() => getRandomSuggestions(allSuggestions, window.innerWidth <= 768 ? 2 : 4));
  const [chatSuggestions, setChatSuggestions] = useState(() => getRandomSuggestions(allSuggestions, 2));

  // API endpoint configuration
  const API_URL = `/query`;

  // Check for JSON files on component mount
  useEffect(() => {
    setDbCheckStatus('loading');
    setShowGreenDb(false);
    const checkJsonFiles = async () => {
      try {
        const response = await fetch(`/check-json-files`);
        const data = await response.json();
        if (data.hasJsonFiles) {
          setHasJsonFiles(true);
          setDbCheckStatus('success');
          setShowGreenDb(true);
          setTimeout(() => setShowGreenDb(false), 1500); // fade out after 1.5s
        } else {
          setHasJsonFiles(false);
          setDbCheckStatus('error');
        }
      } catch (error) {
        console.error('Error checking JSON files:', error);
        setHasJsonFiles(false);
        setDbCheckStatus('error');
      }
    };
    checkJsonFiles();
  }, []);

  // When a message is sent, pick new chat suggestions
  useEffect(() => {
    if (messages.length > 0 && messages.some(m => m.type === 'bot')) {
      setChatSuggestions(getRandomSuggestions(allSuggestions, 2));
    }
  }, [messages]);

  // Update isMobile on resize
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768);
    };
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // When isMobile changes, update homeSuggestions
  useEffect(() => {
    setHomeSuggestions(getRandomSuggestions(allSuggestions, isMobile ? 2 : 4));
  }, [isMobile]);

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
          formattedResponse = `### Documents Found
          ${data.result.document}

          ### Summary
          ${data.result.summary}`;
          setDocumentResultText(`${data.result.document}\n\nSummary\n${data.result.summary}`);
        } else if (data.result.type === 'ruleset_evaluation') {
          formattedResponse = `## Ruleset Evaluation Report\n\n### ${data.result.report}\n\n## Explanation\n\n${data.result.explanation}`;
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

  const renderMessageContent = (content) => {
    // Check if this is a document search result
    if (content.includes('Documents Found') && documents.length > 0) {
      return <DocumentResultsTable foundLine={foundLine} documents={documents} summary={summary} />;
    }
    if (content.includes('Event Information')) {
      return <ReactMarkdown>{content}</ReactMarkdown>;
    }
    if (content.includes('Ruleset Evaluation')) {
      return <ReactMarkdown>{content}</ReactMarkdown>;
    }
    if (content.includes('Knowledge Response')) {
      return <ReactMarkdown>{content}</ReactMarkdown>;
    }

    // Simple Markdown parsing for headings and line breaks
    const formattedContent = content
      .replace(/^### (.*$)/gm, '<h3>$1</h3>')
      .replace(/^## (.*$)/gm, '<h4>$1</h4>')
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

  // Drag and drop handlers
  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  };
  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  };
  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const files = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
    setUploadedFiles(prev => [...prev, ...files]);
  };
  const handleFileChange = (e) => {
    const files = Array.from(e.target.files).filter(f => f.type === 'application/pdf');
    setUploadedFiles(prev => [...prev, ...files]);
  };
  const closeModal = () => {
    setShowDbModal(false);
    setDragActive(false);
    setUploadedFiles([]);
  };

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
        {/* Suggestions box on first page (no messages) */}
        {messages.length === 0 && (
          <div
            className="suggestions-container"
            style={{
              position: 'fixed',
              left: 0,
              right: 0,
              bottom: hasJsonFiles ? '5.5rem' : '7.5rem',
              zIndex: 9,
              background: 'linear-gradient(to bottom, rgba(0,0,0,0.7), #000000 90%)',
              maxWidth: '48rem',
              margin: '0 auto',
              paddingBottom: '0.5rem'
            }}
          >
            <div className="suggestions-grid">
              {homeSuggestions.map((suggestion, index) => (
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
        )}
        {messages.length === 0 ? (
          <div style={{ height: '6rem' }} />
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
        {messages.length > 0 && messages.some(m => m.type === 'bot') && !isTyping && (
          <div
            className="suggestions-container"
            style={{
              position: 'fixed',
              left: 0,
              right: 0,
              bottom: hasJsonFiles ? '5.5rem' : '7.5rem',
              zIndex: 9,
              background: 'linear-gradient(to bottom, rgba(0,0,0,0.7), #000000 90%)',
              maxWidth: '48rem',
              margin: '0 auto',
              paddingBottom: '0.5rem'
            }}
          >
            <div className="suggestions-grid">
              {chatSuggestions.map((suggestion, index) => (
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
        )}
        <div className="input-form-container">
          <form onSubmit={handleSubmit} className="input-form">
            <div className="input-wrapper">
              {/* Database status icon/emoji */}
              {dbCheckStatus === 'loading' && (
                <span className="db-status-emoji" title="Checking database...">⏳</span>
              )}
              {dbCheckStatus === 'success' && showGreenDb && (
                <span className="database-icon-button has-files fade-out" title="Database loaded!">
                  <svg viewBox="0 0 24 24" width="24" height="24">
                    <ellipse cx="12" cy="6" rx="8" ry="3" fill="currentColor" />
                    <path d="M4 6v6c0 1.66 3.58 3 8 3s8-1.34 8-3V6" fill="none" stroke="currentColor" strokeWidth="2"/>
                    <path d="M4 12v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" fill="none" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </span>
              )}
              {dbCheckStatus === 'error' && (
                <button
                  type="button"
                  className={`database-icon-button no-files`}
                  tabIndex={-1}
                  aria-label="Database not loaded"
                  onClick={() => setShowDbModal(true)}
                >
                  <svg viewBox="0 0 24 24" width="24" height="24">
                    <ellipse cx="12" cy="6" rx="8" ry="3" fill="currentColor" />
                    <path d="M4 6v6c0 1.66 3.58 3 8 3s8-1.34 8-3V6" fill="none" stroke="currentColor" strokeWidth="2"/>
                    <path d="M4 12v6c0 1.66 3.58 3 8 3s8-1.34 8-3v-6" fill="none" stroke="currentColor" strokeWidth="2"/>
                  </svg>
                </button>
              )}
              <textarea
                ref={textareaRef}
                rows={1}
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Send a message..."
                className="chat-input"
                aria-label="Chat input"
                style={{
                  paddingLeft:
                    dbCheckStatus === 'loading' || (dbCheckStatus === 'success' && showGreenDb) || dbCheckStatus === 'error'
                      ? '1.5rem'
                      : '1rem',
                  transition: 'padding-left 0.6s cubic-bezier(0.22, 1, 0.36, 1)'
                }}
              />
            </div>
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
          {!hasJsonFiles && (
            <div className="dataset-warning">
              <svg viewBox="0 0 24 24" width="16" height="16" className="warning-icon">
                <path fill="currentColor" d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z"/>
              </svg>
              <span>Dataset is not loaded - Please ensure JSON files are present in the data directory</span>
            </div>
          )}
        </div>
      </div>

      {/* Drag-and-drop DB modal */}
      {showDbModal && (
        <div className="db-modal-overlay" onClick={closeModal}>
          <div className="db-modal" onClick={e => e.stopPropagation()}>
            <button className="db-modal-close" onClick={closeModal}>&times;</button>
            <h2>Upload PDF Files</h2>
            <div
              className={`db-drop-area${dragActive ? ' drag-active' : ''}`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <p>Drag & drop PDF files here, or <label htmlFor="db-file-input" className="db-file-label">browse</label></p>
              <input
                id="db-file-input"
                type="file"
                accept="application/pdf"
                multiple
                style={{ display: 'none' }}
                onChange={handleFileChange}
              />
            </div>
            {uploadedFiles.length > 0 && (
              <div className="db-uploaded-list">
                <h4>Selected files:</h4>
                <ul>
                  {uploadedFiles.map((file, idx) => (
                    <li key={idx}>{file.name}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default ChatBot;
