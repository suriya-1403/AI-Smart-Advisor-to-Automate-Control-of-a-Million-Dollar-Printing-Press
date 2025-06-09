import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000'; // Adjust this to match your backend port

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const processQuery = async (query) => {
  try {
    const response = await api.post('/query', { query });
    return response.data;
  } catch (error) {
    console.error('Error processing query:', error);
    throw error;
  }
};

export const setupSSE = (onMessage) => {
  const eventSource = new EventSource(`${API_BASE_URL}/sse`);

  eventSource.onmessage = (event) => {
    onMessage(JSON.parse(event.data));
  };

  eventSource.onerror = (error) => {
    console.error('SSE Error:', error);
    eventSource.close();
  };

  return eventSource;
};

export default api;
