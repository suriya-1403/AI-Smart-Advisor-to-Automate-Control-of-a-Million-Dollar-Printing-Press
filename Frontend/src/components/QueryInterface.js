import React, { useState, useEffect } from 'react';
import { processQuery, setupSSE } from '../services/api';

const QueryInterface = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    // Set up SSE connection
    const eventSource = setupSSE((data) => {
      setResults(prev => prev + '\n' + data.message);
    });

    return () => {
      eventSource.close();
    };
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResults('');

    try {
      const response = await processQuery(query);
      setResults(response.result);
    } catch (err) {
      setError('Error processing query. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="query-interface">
      <form onSubmit={handleSubmit}>
        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query here..."
          rows={4}
          className="query-input"
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Processing...' : 'Submit Query'}
        </button>
      </form>

      {error && <div className="error">{error}</div>}

      {results && (
        <div className="results">
          <h3>Results:</h3>
          <pre>{results}</pre>
        </div>
      )}
    </div>
  );
};

export default QueryInterface;
