/**
 * Agent Chat Widget Component
 * Main floating chat interface for AI Agent
 */

import React, { useState, useRef, useEffect } from 'react';
import { Message, QuickAction } from '../../types/agent';
import { agentService } from '../../services/agentService';
import { MessageBubble, TypingIndicator } from './MessageBubble';
import { ServiceStatusBar } from './ServiceStatusBar';
import ConfirmDialog from '../shared/ConfirmDialog';
import {
  BotIcon,
  TrashIcon,
  MaximizeIcon,
  RestoreIcon,
  MinimizeIcon,
  WindowIcon,
  CloseIcon,
  SendIcon,
  LoadingIcon,
  SearchIcon,
  DocumentIcon,
  HelpIcon
} from './AgentIcons';
import '../../styles/AgentChat.css';

interface AgentChatWidgetProps {
  initialOpen?: boolean;
}

const QUICK_ACTIONS: QuickAction[] = [
  {
    label: 'Check Status',
    icon: 'search',
    message: 'Camera service có đang chạy không?'
  },
  {
    label: 'View Logs',
    icon: 'document',
    message: 'Cho tôi xem logs của camera service (50 dòng cuối)'
  },
  {
    label: 'Help',
    icon: 'help',
    message: 'Tôi cần giúp đỡ về camera service'
  }
];

export const AgentChatWidget: React.FC<AgentChatWidgetProps> = ({ initialOpen = false }) => {
  const [isOpen, setIsOpen] = useState(initialOpen);
  const [isMinimized, setIsMinimized] = useState(false);
  const [isMaximized, setIsMaximized] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      id: 'welcome',
      role: 'assistant',
      content: 'Xin chào! 👋 Tôi là Service Management Assistant.\n\nTôi có thể giúp bạn:\n• Kiểm tra trạng thái services\n• Start/Stop camera service\n• Xem và phân tích logs\n• Troubleshoot các vấn đề\n\nBạn cần giúp gì?',
      timestamp: new Date(),
      status: 'success'
    }
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const [clearError, setClearError] = useState<string | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const sessionId = useRef(`session_${Date.now()}`);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const handleSend = async () => {
    if (!inputValue.trim() || isSending) return;

    const userMessage: Message = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: inputValue.trim(),
      timestamp: new Date(),
      status: 'success'
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsTyping(true);
    setIsSending(true);

    // Auto-resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
    }

    try {
      const response = await agentService.chat({
        message: userMessage.content,
        agent_id: 'service_management',
        session_id: sessionId.current
      });

      const assistantMessage: Message = {
        id: `assistant_${Date.now()}`,
        role: 'assistant',
        content: response.response,
        timestamp: new Date(response.timestamp),
        toolCalls: response.tool_calls,
        status: 'success'
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error: any) {
      console.error('Chat error:', error);

      const errorMessage: Message = {
        id: `error_${Date.now()}`,
        role: 'system',
        content: `❌ Lỗi: ${error.message || 'Không thể kết nối với agent'}`,
        timestamp: new Date(),
        status: 'error'
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
      setIsSending(false);
    }
  };

  const handleQuickAction = (action: QuickAction) => {
    setInputValue(action.message);
    textareaRef.current?.focus();
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setInputValue(e.target.value);

    // Auto-resize textarea
    e.target.style.height = 'auto';
    e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
  };

  const toggleMinimize = () => {
    setIsMinimized(!isMinimized);
  };

  const toggleOpen = () => {
    setIsOpen(!isOpen);
    if (!isOpen) {
      setIsMinimized(false);
    }
  };

  const handleClearChat = async () => {
    try {
      await agentService.clearConversation(sessionId.current);

      // Reset messages to welcome message only
      setMessages([{
        id: 'welcome',
        role: 'assistant',
        content: 'Xin chào! 👋 Tôi là Service Management Assistant.\n\nTôi có thể giúp bạn:\n• Kiểm tra trạng thái services\n• Start/Stop camera service\n• Xem và phân tích logs\n• Troubleshoot các vấn đề\n\nBạn cần giúp gì?',
        timestamp: new Date(),
        status: 'success'
      }]);

      // Generate new session ID
      sessionId.current = `session_${Date.now()}`;
      setClearError(null);
    } catch (error: any) {
      console.error('Failed to clear chat:', error);
      setClearError(error.message);
    }
  };

  const toggleMaximize = () => {
    setIsMaximized(!isMaximized);
    if (isMinimized) {
      setIsMinimized(false);
    }
  };

  const getQuickActionIcon = (iconName: string) => {
    switch (iconName) {
      case 'search':
        return <SearchIcon size={16} />;
      case 'document':
        return <DocumentIcon size={16} />;
      case 'help':
        return <HelpIcon size={16} />;
      default:
        return null;
    }
  };

  if (!isOpen) {
    return (
      <button className="agent-toggle-btn" onClick={toggleOpen} title="Open AI Assistant">
        <BotIcon size={28} />
      </button>
    );
  }

  return (
    <>
      <div className={`agent-chat-widget ${isMinimized ? 'minimized' : ''} ${isMaximized ? 'maximized' : ''}`}>
        {/* Header */}
        <div className="agent-chat-header">
        <div className="agent-chat-title">
          <span className="agent-icon">
            <BotIcon size={24} />
          </span>
          <div className="agent-title-text">
            <h3>Service Assistant</h3>
            {!isMinimized && <p>AI-powered service management</p>}
          </div>
        </div>

        <div className="agent-header-actions">
          <button
            className="agent-header-btn"
            onClick={() => setShowClearConfirm(true)}
            title="Clear chat history"
            disabled={isSending}
          >
            <TrashIcon size={18} />
          </button>
          {!isMinimized && (
            <button
              className="agent-header-btn"
              onClick={toggleMaximize}
              title={isMaximized ? 'Restore size' : 'Maximize to full screen'}
            >
              {isMaximized ? <RestoreIcon size={18} /> : <MaximizeIcon size={18} />}
            </button>
          )}
          <button
            className="agent-header-btn"
            onClick={toggleMinimize}
            title={isMinimized ? 'Maximize' : 'Minimize'}
          >
            {isMinimized ? <WindowIcon size={18} /> : <MinimizeIcon size={18} />}
          </button>
          <button
            className="agent-header-btn"
            onClick={toggleOpen}
            title="Close"
          >
            <CloseIcon size={18} />
          </button>
        </div>
      </div>

      {!isMinimized && (
        <>
          {/* Service Status Bar */}
          <ServiceStatusBar />

          {/* Messages Area */}
          <div className="agent-messages">
            {messages.map(message => (
              <MessageBubble key={message.id} message={message} />
            ))}

            {isTyping && <TypingIndicator />}

            <div ref={messagesEndRef} />
          </div>

          {/* Quick Actions */}
          <div className="agent-quick-actions">
            {QUICK_ACTIONS.map((action, index) => (
              <button
                key={index}
                className="quick-action-btn"
                onClick={() => handleQuickAction(action)}
                disabled={isSending}
              >
                {getQuickActionIcon(action.icon)}
                <span>{action.label}</span>
              </button>
            ))}
          </div>

          {/* Input Area */}
          <div className="agent-input-area">
            <div className="agent-input-wrapper">
              <textarea
                ref={textareaRef}
                className="agent-textarea"
                placeholder="Type your message... (Enter to send, Shift+Enter for new line)"
                value={inputValue}
                onChange={handleInputChange}
                onKeyDown={handleKeyDown}
                disabled={isSending}
                rows={1}
              />
              <button
                className="agent-send-btn"
                onClick={handleSend}
                disabled={!inputValue.trim() || isSending}
                title="Send message"
              >
                {isSending ? <LoadingIcon size={20} /> : <SendIcon size={20} />}
              </button>
            </div>
          </div>
        </>
      )}
      </div>

      {/* Clear Chat Confirmation Dialog */}
      <ConfirmDialog
        isOpen={showClearConfirm}
        onClose={() => setShowClearConfirm(false)}
        onConfirm={handleClearChat}
        title="Xóa lịch sử chat?"
        message="Bạn có chắc muốn xóa toàn bộ lịch sử trò chuyện? Agent sẽ không nhớ các cuộc hội thoại trước đó."
        confirmText="Xóa"
        cancelText="Hủy"
        type="warning"
      />

      {/* Error Dialog for Clear Chat */}
      {clearError && (
        <ConfirmDialog
          isOpen={!!clearError}
          onClose={() => setClearError(null)}
          onConfirm={() => setClearError(null)}
          title="Lỗi"
          message={`Không thể xóa lịch sử chat: ${clearError}`}
          confirmText="OK"
          cancelText=""
          type="danger"
        />
      )}
    </>
  );
};
