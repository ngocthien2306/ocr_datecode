/**
 * Message Bubble Component
 * Displays individual chat messages with tool execution info
 */

import React from 'react';
import { Message } from '../../types/agent';

interface MessageBubbleProps {
  message: Message;
}

export const MessageBubble: React.FC<MessageBubbleProps> = ({ message }) => {
  const formatTime = (date: Date) => {
    return new Date(date).toLocaleTimeString('vi-VN', {
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const formatMessageContent = (content: string) => {
    // Preserve line breaks and formatting
    return content.split('\n').map((line, index) => (
      <React.Fragment key={index}>
        {line}
        {index < content.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };

  return (
    <div className={`message-bubble ${message.role}`}>
      <div className="message-content">
        {formatMessageContent(message.content)}
      </div>

      {message.toolCalls && message.toolCalls.length > 0 && (
        <div className="message-tools">
          <div style={{ fontWeight: 600, marginBottom: 4, fontSize: '11px', opacity: 0.8 }}>
            🔧 Tools used:
          </div>
          {message.toolCalls.map((tool, index) => (
            <div key={index} className="tool-item">
              <span className="tool-item-icon">⚙️</span>
              <span>{tool.tool}</span>
            </div>
          ))}
        </div>
      )}

      <div className="message-timestamp">
        {formatTime(message.timestamp)}
      </div>
    </div>
  );
};

/**
 * Typing Indicator Component
 */
export const TypingIndicator: React.FC = () => {
  return (
    <div className="typing-indicator">
      <div className="typing-dots">
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
        <div className="typing-dot"></div>
      </div>
      <span style={{ fontSize: '13px', color: '#9ca3af' }}>Agent đang trả lời...</span>
    </div>
  );
};
