import { useState, useRef, useEffect, useCallback } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import rehypeSanitize, { defaultSchema } from 'rehype-sanitize';
import { Send, Activity, Stethoscope, AlertTriangle, Brain, ChevronDown, ChevronRight, Trash2 } from 'lucide-react';
import './index.css';

// Tách nội dung <think>...</think> ra khỏi phần trả lời chính
function parseThinkingContent(rawText) {
  if (!rawText) return { thinking: '', answer: '', isThinking: false };

  const thinkOpenTag = '<think>';
  const thinkCloseTag = '</think>';
  const openIdx = rawText.indexOf(thinkOpenTag);

  if (openIdx === -1) {
    return { thinking: '', answer: rawText, isThinking: false };
  }

  const closeIdx = rawText.indexOf(thinkCloseTag, openIdx);

  if (closeIdx === -1) {
    // Chưa đóng </think> → đang trong quá trình suy luận
    const thinkContent = rawText.slice(openIdx + thinkOpenTag.length);
    const beforeThink = rawText.slice(0, openIdx);
    return { thinking: thinkContent, answer: beforeThink, isThinking: true };
  }

  // Đã đóng </think>
  const thinkContent = rawText.slice(openIdx + thinkOpenTag.length, closeIdx);
  const afterThink = rawText.slice(closeIdx + thinkCloseTag.length);
  const beforeThink = rawText.slice(0, openIdx);
  return { thinking: thinkContent, answer: beforeThink + afterThink, isThinking: false };
}

// Component hiển thị phần suy luận của AI (collapsible)
function ThinkingBlock({ content, isThinking }) {
  const [isOpen, setIsOpen] = useState(isThinking);

  useEffect(() => {
    if (isThinking) setIsOpen(true);
  }, [isThinking]);

  if (!content) return null;

  return (
    <div className={`thinking-block ${isThinking ? 'active' : 'done'}`}>
      <button className="thinking-toggle" onClick={() => setIsOpen(!isOpen)}>
        <Brain size={14} />
        <span>{isThinking ? 'AI đang suy luận...' : 'Xem quá trình suy luận'}</span>
        {isThinking && <span className="thinking-pulse" />}
        {isOpen ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
      </button>
      {isOpen && (
        <div className="thinking-content">
          <pre>{content}</pre>
        </div>
      )}
    </div>
  );
}

// Component hiển thị 1 tin nhắn bot với markdown
function BotMessage({ msg }) {
  const { thinking, answer, isThinking } = parseThinkingContent(msg.content);

  const formatMarkdown = (text) => {
    if (!text) return '';

    let f = text.replace(/\r\n/g, '\n');

    // ===== Fix inline markdown structures =====
    // LLM thường output markdown structures mà không có newline phía trước.
    // Chèn \n\n trước các cấu trúc block-level khi chúng xuất hiện inline.

    // 1. Horizontal rules: --- khi đứng inline (trước nó không phải newline)
    f = f.replace(/([^\n])\s*(\n?---)\s*/g, '$1\n\n---\n\n');

    // 2. Headings: # ## ### #### khi đứng inline
    f = f.replace(/([^\n])\s*(#{1,4}\s+)/g, '$1\n\n$2');

    // 2b. Tách các heading đặc thù hay bị dính inline
    const commonHeadings = [
      "Giải thích chi tiết",
      "Khuyến nghị an toàn",
      "Khuyến cáo",
      "Kết luận",
      "Tóm tắt",
      "Lưu ý quan trọng",
      "Lưu ý an toàn",
      "Cảnh báo",
      "Phân tích"
    ];
    for (const h of commonHeadings) {
      // Tìm "# Heading" theo sau là khoảng trắng và 1 từ (để chắc chắn nó bị dính)
      const regex = new RegExp(`(#{1,4}\\s+${h})\\s+(?=\\S)`, 'gi');
      f = f.replace(regex, '$1\n\n');
    }
    
    // Tách heading nếu nó kết thúc bằng dấu hai chấm
    f = f.replace(/(#{1,4}\s+[^:\n]+:)\s+(?=\S)/g, '$1\n\n');

    // 3. Numbered list: "1. " "2. " khi đứng inline (trước nó là ký tự thường, không phải \n)
    f = f.replace(/([^\n\d])\s+(\d+\.\s+)/g, '$1\n\n$2');

    // 3b. Tách heading numbered item khỏi sub-items: "1. Title: - **Sub**"
    f = f.replace(/(\d+\.\s+[^:\n]+:)\s+(- )/g, '$1\n$2');

    // 4. Dash list items trước bold: "- **Text**:" — pattern phổ biến nhất của model
    f = f.replace(/([^\n])\s+(- \*\*)/g, '$1\n$2');

    // 5. Dash list items trước text thường: ". - Text" hoặc ": - Text"
    f = f.replace(/([.!?:;)）。])\s+(- [^\s*-])/g, '$1\n$2');

    // ===== Đảm bảo single \n → \n\n để ReactMarkdown hiểu là paragraph break =====
    f = f.replace(/(?<!\n)\n(?!\n)/g, '\n\n');

    // Thu gọn 3+ newlines → 2
    f = f.replace(/\n{3,}/g, '\n\n');

    // Trim leading whitespace/newlines
    f = f.replace(/^\s+/, '');

    // Citation badges
    f = f.replace(/\[(\d+)\]/g, '<span class="citation-badge">[$1]</span>');

    return f;
  };

  return (
    <>
      <ThinkingBlock content={thinking} isThinking={isThinking} />
      {answer.trim() && (
        <div className="msg-content">
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            rehypePlugins={[
              rehypeRaw,
              [rehypeSanitize, {
                ...defaultSchema,
                attributes: {
                  ...defaultSchema.attributes,
                  span: [...(defaultSchema.attributes.span || []), 'className', 'class']
                }
              }]
            ]}
          >
            {formatMarkdown(answer)}
          </ReactMarkdown>
        </div>
      )}
      {/* Khi đang stream mà chưa có answer (vẫn đang think), hiện placeholder */}
      {msg._isStreaming && !answer.trim() && !isThinking && (
        <div className="msg-content">
          <span className="stream-cursor" />
        </div>
      )}
    </>
  );
}

const initialWelcomeMessage = [
  {
    id: 1,
    role: 'bot',
    content: 'Chào bạn, mình là Trợ lý Y tế AI. Bạn cần tư vấn thông tin gì về sức khỏe hay thuốc men hôm nay?',
    metadata: null
  }
];

function App() {
  const [messages, setMessages] = useState(() => {
    const saved = localStorage.getItem('chat_messages');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        // Clear any dangling streaming flags
        return parsed.map(msg => ({ ...msg, _isStreaming: false }));
      } catch (e) {
        console.error("Error parsing saved messages:", e);
      }
    }
    return initialWelcomeMessage;
  });
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);   // Đang chờ server phản hồi
  const [isStreaming, setIsStreaming] = useState(false); // Đang nhận token
  const [stats, setStats] = useState(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);
  const abortControllerRef = useRef(null);

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, isLoading, isStreaming, scrollToBottom]);

  useEffect(() => {
    localStorage.setItem('chat_messages', JSON.stringify(messages));
  }, [messages]);

  useEffect(() => {
    // Fetch stats on load
    axios.get('http://localhost:8000/api/stats')
      .then(res => setStats(res.data.stats))
      .catch(err => console.error("Could not fetch stats:", err));
  }, []);

  const handleInput = (e) => {
    setInput(e.target.value);
    // Auto resize textarea
    if (textareaRef.current) {
      textareaRef.current.style.height = '24px';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 120)}px`;
    }
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading || isStreaming) return;

    const userMsg = input.trim();
    setInput('');
    if (textareaRef.current) textareaRef.current.style.height = '24px';

    const userMsgId = Date.now();
    const botMsgId = userMsgId + 1;

    setMessages(prev => [...prev, { id: userMsgId, role: 'user', content: userMsg }]);
    setIsLoading(true);

    // Hỗ trợ abort request
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await fetch('http://localhost:8000/api/chat/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, is_emergency: isEmergency }),
        signal: controller.signal
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');

      let isFirstMetadata = true;
      let currentContent = "";
      let buffer = "";
      let messageMetadata = null;

      // RAF batching: gom nhiều token cùng 1 frame để giảm re-render
      let pendingContent = null;
      let rafId = null;

      const flushContent = () => {
        if (pendingContent !== null) {
          const contentToSet = pendingContent;
          setMessages(prev => {
            const newMessages = [...prev];
            const lastMsg = newMessages[newMessages.length - 1];
            if (lastMsg && lastMsg.id === botMsgId) {
              newMessages[newMessages.length - 1] = {
                ...lastMsg,
                content: contentToSet
              };
            }
            return newMessages;
          });
          pendingContent = null;
        }
        rafId = null;
      };

      const scheduleUpdate = (content) => {
        pendingContent = content;
        if (!rafId) {
          rafId = requestAnimationFrame(flushContent);
        }
      };

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const parts = buffer.split('\n\n');
        buffer = parts.pop() || "";

        for (const part of parts) {
          const trimmed = part.trim();
          if (!trimmed.startsWith('data: ')) continue;

          try {
            const data = JSON.parse(trimmed.slice(6));

            if (data.type === 'metadata') {
              if (data.data.type === 'emergency' || data.data.type === 'out_of_scope' || data.data.type === 'insufficient_evidence') {
                // Flush pending trước
                if (rafId) { cancelAnimationFrame(rafId); flushContent(); }
                setMessages(prev => [...prev, {
                  id: botMsgId,
                  role: 'bot',
                  content: data.data.message,
                  metadata: { type: data.data.type }
                }]);
                if (data.data.type === 'emergency') {
                  setIsEmergency(true);
                }
                setIsLoading(false);
                return;
              }

              if (isFirstMetadata) {
                messageMetadata = {
                  type: data.data.type,
                  category: data.data.category,
                  risk: data.data.risk_level,
                  route: data.data.route,
                  sources: data.data.sources,
                  disclaimer: data.data.disclaimer
                };
                if (data.data.risk_level === 'critical' || data.data.risk_level === 'high') {
                  setIsEmergency(true);
                }
                setMessages(prev => [...prev, {
                  id: botMsgId,
                  role: 'bot',
                  content: '',
                  metadata: messageMetadata,
                  _isStreaming: true
                }]);
                setIsLoading(false);
                setIsStreaming(true);
                isFirstMetadata = false;
              }
            } else if (data.type === 'token') {
              currentContent += data.content;
              scheduleUpdate(currentContent);
            } else if (data.type === 'error') {
              throw new Error(data.content);
            }
          } catch (e) {
            if (e.message && !e.message.includes('JSON')) throw e;
            console.warn("SSE parse warning:", e.message, trimmed.slice(0, 100));
          }
        }
      }

      // Flush bất kì token còn lại
      if (rafId) { cancelAnimationFrame(rafId); }
      flushContent();

      // Chèn disclaimer vào cuối nếu có
      setMessages(prev => {
        const newMessages = [...prev];
        const lastMsgIndex = newMessages.length - 1;
        const lastMsg = newMessages[lastMsgIndex];

        if (lastMsg && lastMsg.id === botMsgId) {
          let finalContent = lastMsg.content;
          if (lastMsg.metadata && lastMsg.metadata.disclaimer) {
            if (!finalContent.includes(lastMsg.metadata.disclaimer)) {
              finalContent = finalContent + "\n\n---\n" + lastMsg.metadata.disclaimer;
            }
          }
          newMessages[lastMsgIndex] = {
            ...lastMsg,
            content: finalContent,
            _isStreaming: false
          };
        }
        return newMessages;
      });

    } catch (error) {
      if (error.name === 'AbortError') return;
      setMessages(prev => [...prev, {
        id: Date.now() + 1,
        role: 'bot',
        content: `**Lỗi kết nối:** Không thể gọi tới server. Vui lòng đảm bảo FastAPI backend đang chạy.\n\nChi tiết: ${error.message}`,
        metadata: { type: 'error' }
      }]);
    } finally {
      setIsLoading(false);
      setIsStreaming(false);
      abortControllerRef.current = null;
    }
  };

  const [isEmergency, setIsEmergency] = useState(false);

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1><Activity size={28} /> Health AI Assistant</h1>
        <div className="stats">
          <button
            className="clear-btn"
            onClick={() => {
              if (window.confirm("Bạn có chắc chắn muốn xóa toàn bộ lịch sử trò chuyện?")) {
                localStorage.removeItem('chat_messages');
                setMessages(initialWelcomeMessage);
              }
            }}
            title="Xóa lịch sử trò chuyện"
            style={{
              padding: '8px',
              borderRadius: '8px',
              border: '1px solid var(--border)',
              background: 'transparent',
              color: 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              transition: 'all 0.2s ease',
              marginRight: '8px',
              width: '38px',
              height: '38px',
              boxSizing: 'border-box'
            }}
            onMouseOver={(e) => {
              e.currentTarget.style.color = '#ef4444';
              e.currentTarget.style.borderColor = 'rgba(239, 68, 68, 0.4)';
              e.currentTarget.style.background = 'rgba(239, 68, 68, 0.05)';
            }}
            onMouseOut={(e) => {
              e.currentTarget.style.color = 'var(--text-muted)';
              e.currentTarget.style.borderColor = 'var(--border)';
              e.currentTarget.style.background = 'transparent';
            }}
          >
            <Trash2 size={18} />
          </button>
          <button
            className={`sos-btn ${isEmergency ? 'active' : ''}`}
            onClick={() => setIsEmergency(!isEmergency)}
            title="Bật/Tắt chế độ cấp cứu"
          >
            🚨 SOS
          </button>
          {stats ? (
            <span>📚 {stats.vi_count} tài liệu | RAG + GGUF Engine</span>
          ) : (
            <span>Đang kết nối...</span>
          )}
        </div>
      </header>

      <main className="chat-window">
        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            {msg.role === 'bot' ? (
              <BotMessage msg={msg} />
            ) : (
              <div className="msg-content">
                {msg.content}
              </div>
            )}

            {msg.role === 'bot' && msg.metadata && msg.metadata.type !== 'error' && msg.metadata.route && (
              <div className="metadata">
                {msg.metadata.risk === 'critical' || msg.metadata.risk === 'high' ? (
                  <span style={{color: '#ef4444', display: 'flex', alignItems: 'center', gap: '4px'}}>
                    <AlertTriangle size={12} /> Nguy cơ: {msg.metadata.risk.toUpperCase()}
                  </span>
                ) : (
                  <span>Danh mục: {msg.metadata.category}</span>
                )}
                <span>•</span>
                <span>Luồng: {msg.metadata.route === 'general_qa' ? '🤖 Local LLM' : '🔍 RAG Pipeline'}</span>
              </div>
            )}

            {msg.role === 'bot' && msg.metadata && msg.metadata.sources && msg.metadata.sources.length > 0 && (
              <div className="sources-container">
                <details className="sources-details">
                  <summary className="sources-summary">
                    🔍 Xem {msg.metadata.sources.length} tài liệu đối chiếu y khoa
                  </summary>
                  <div className="sources-list">
                    {msg.metadata.sources.map((src) => (
                      <div key={src.index} className="source-card">
                        <div className="source-title" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '8px' }}>
                          <strong>[{src.index}] {src.title}</strong>
                          {src.publish_date && (
                            <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 'normal' }}>
                              📅 {src.publish_date.split('T')[0]}
                            </span>
                          )}
                        </div>
                        <p className="source-content">{src.content}</p>
                      </div>
                    ))}
                  </div>
                </details>
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="typing-indicator">
            <div className="dot"></div>
            <div className="dot"></div>
            <div className="dot"></div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      <footer className="input-area">
        <div className="input-wrapper">
          <Stethoscope size={24} color="var(--text-muted)" />
          <textarea
            ref={textareaRef}
            value={input}
            onChange={handleInput}
            onKeyDown={handleKeyDown}
            placeholder="Hỏi về triệu chứng, thuốc, hoặc lời khuyên y tế..."
            rows={1}
            disabled={isLoading || isStreaming}
          />
          <button onClick={handleSend} disabled={!input.trim() || isLoading || isStreaming}>
            <Send size={20} />
          </button>
        </div>
      </footer>
    </div>
  );
}

export default App;
