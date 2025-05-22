// mcpService.js
class MCPService {
  constructor(serverUrl = 'http://0.0.0.0:8000/sse') {
    this.serverUrl = serverUrl;
    this.eventSource = null;
    this.messageHandlers = [];
    this.connected = false;
  }

  // Connect to the MCP server
  connect() {
    return new Promise((resolve, reject) => {
      try {
        // Close any existing connection
        if (this.eventSource) {
          this.disconnect();
        }

        console.log('Connecting to MCP server:', this.serverUrl);

        // Create a new EventSource connection
        const eventSource = new EventSource(this.serverUrl);
        this.eventSource = eventSource;

        // Set up the event handlers
        eventSource.onopen = () => {
          console.log('SSE connection opened to MCP server');
          this.connected = true;
          resolve();
        };

        eventSource.onerror = (error) => {
          console.error('Error with SSE connection to MCP server:', error);
          this.connected = false;
          reject(error);
        };

        // Process incoming messages directly without using the parser
        eventSource.onmessage = (event) => {
          console.log('Raw SSE message:', event.data);
          try {
            const data = JSON.parse(event.data);
            console.log('Parsed MCP event:', data);
            this.handleMCPEvent(data);
          } catch (e) {
            console.error('Error parsing MCP event:', e);
          }
        };
      } catch (error) {
        console.error('Failed to connect to MCP server:', error);
        reject(error);
      }
    });
  }

  // Disconnect from the MCP server
  disconnect() {
    if (this.eventSource) {
      this.eventSource.close();
      this.eventSource = null;
      this.connected = false;
      console.log('Disconnected from MCP server');
    }
  }

  // Add a message handler
  addMessageHandler(handler) {
    this.messageHandlers.push(handler);
    return () => {
      this.messageHandlers = this.messageHandlers.filter(h => h !== handler);
    };
  }

  // Handle MCP events
  handleMCPEvent(data) {
    // Log the received data
    console.log('Processing MCP event:', data);

    // Check for error responses
    if (data.error) {
      console.error('MCP server returned error:', data.error);
      const errorData = {
        task_type: "error",
        document: `Error: ${data.error.message || "Unknown error"}`,
        explanation: `Error: ${data.error.message || "Unknown error"}`
      };
      this.messageHandlers.forEach(handler => handler(errorData));
      return;
    }

    // Handle successful responses
    if (data.result) {
      let content = '';

      // Extract content based on response format
      if (data.result.content && Array.isArray(data.result.content)) {
        // Handle tool call responses
        content = data.result.content[0]?.text || JSON.stringify(data.result);
      } else {
        // Handle other response formats
        content = JSON.stringify(data.result);
      }

      // Create a processed response
      const processedData = {
        task_type: this.determineTaskType(content),
        document: content,
        explanation: content
      };

      console.log('Sending to handlers:', processedData);
      this.messageHandlers.forEach(handler => handler(processedData));
    } else {
      // Pass through other response types
      this.messageHandlers.forEach(handler => handler(data));
    }
  }

  // Determine the task type based on content
  determineTaskType(content) {
    if (typeof content !== 'string') {
      return 'unknown';
    }

    // Determine task type based on content
    if (content.includes('Document Search Results') || content.includes('event id')) {
      return 'document_search';
    } else if (content.includes('Ruleset Evaluation') || content.includes('media coverage')) {
      return 'ruleset_evaluation';
    }
    return 'unknown';
  }

  // Send a query via WebSocket or fetch depending on server capabilities
  async sendQuery(query) {
    // Make sure we're connected
    if (!this.connected) {
      try {
        await this.connect();
      } catch (error) {
        console.error('Failed to connect to MCP server:', error);
        throw error;
      }
    }

    try {
      // Determine if this is a ruleset query
      const isRulesetQuery = query.includes('media coating') ||
                            query.includes('ink coverage') ||
                            query.includes('GSM') ||
                            query.includes('optical density');

      const toolName = isRulesetQuery ? 'evaluate_ruleset' : 'find_document';

      // Create a unique message ID based on timestamp
      const messageId = Date.now().toString();

      // Format the JSON-RPC request
      const jsonRpcRequest = {
        jsonrpc: "2.0",
        method: "call_tool",
        params: {
          name: toolName,
          arguments: {
            query: query
          }
        },
        id: messageId
      };

      // The SSE endpoint is only for receiving events, not sending commands
      // We need to use the command endpoint for sending requests
      const commandUrl = this.serverUrl.replace('/sse', '/command');

      console.log('Sending request to command endpoint:', commandUrl, jsonRpcRequest);

      // Try using fetch first with Streamable HTTP transport
      const response = await fetch(commandUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(jsonRpcRequest)
      });

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}: ${response.statusText}`);
      }

      // Note: The response will actually come through the SSE connection
      console.log('Command sent successfully');
      return true;
    } catch (error) {
      console.error('Error sending query:', error);

      // Fall back to URL parameter method if fetch fails
      console.log('Trying alternative message method...');

      try {
        // In SSE, we can try to append the message to the URL as a query parameter
        const url = new URL(this.serverUrl);
        url.searchParams.append('message', JSON.stringify({
          jsonrpc: "2.0",
          method: "call_tool",
          params: {
            name: isRulesetQuery ? 'evaluate_ruleset' : 'find_document',
            arguments: {
              query: query
            }
          },
          id: Date.now().toString()
        }));

        const paramResponse = await fetch(url.toString());
        if (!paramResponse.ok) {
          throw new Error(`URL parameter method failed: ${paramResponse.status}`);
        }

        return true;
      } catch (fallbackError) {
        console.error('All message methods failed:', fallbackError);
        throw error; // Throw the original error
      }
    }
  }

  // Process document search query
  async searchDocuments(query) {
    return this.sendQuery(query);
  }

  // Process ruleset evaluation query
  async evaluateRuleset(parameters) {
    return this.sendQuery(parameters);
  }
}

// Create a singleton instance
const mcpService = new MCPService();
export default mcpService;
