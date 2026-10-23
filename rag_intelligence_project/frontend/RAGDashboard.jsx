import React, { useState } from 'react';

export default function RAGDashboard() {
  const [query, setQuery] = useState('');
  const [chatHistory, setChatHistory] = useState([]);
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    setUploadStatus('Uploading & indexing...');

    try {
      const res = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      setUploadStatus(`Indexed: ${data.message} (${data.chunks_created} chunks)`);
    } catch (err) {
      setUploadStatus('Upload failed. Ensure backend is running.');
    }
  };

  const handleQuery = async (e) => {
    e.preventDefault();
    if (!query.trim() || loading) return;

    const userQ = query;
    setQuery('');
    setLoading(true);
    setChatHistory(prev => [...prev, { sender: 'user', text: userQ }]);

    try {
      const res = await fetch('http://localhost:8000/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: userQ, top_k: 3 }),
      });
      const data = await res.json();

      setChatHistory(prev => [...prev, { sender: 'ai', text: data.answer }]);
      setSources(data.sources || []);
    } catch (err) {
      setChatHistory(prev => [...prev, { sender: 'ai', text: 'Error fetching response from backend.' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ display: 'flex', height: '100vh', fontFamily: 'Inter, sans-serif', backgroundColor: '#f8fafc' }}>
      {/* Main Chat Area */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', borderRight: '1px solid #e2e8f0' }}>
        <header style={{ padding: '16px 24px', borderBottom: '1px solid #e2e8f0', background: '#fff' }}>
          <h2 style={{ margin: 0, fontSize: '1.25rem', color: '#0f172a' }}>Enterprise Document Intelligence (RAG)</h2>
          <p style={{ margin: '4px 0 0', fontSize: '0.85rem', color: '#64748b' }}>FastAPI + ChromaDB + Grounded LLM Citations</p>
        </header>

        <div style={{ flex: 1, overflowY: 'auto', padding: '24px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {chatHistory.map((msg, idx) => (
            <div key={idx} style={{
              alignSelf: msg.sender === 'user' ? 'flex-end' : 'flex-start',
              maxWidth: '75%',
              padding: '12px 16px',
              borderRadius: '8px',
              backgroundColor: msg.sender === 'user' ? '#2563eb' : '#fff',
              color: msg.sender === 'user' ? '#fff' : '#1e293b',
              boxShadow: '0 1px 3px rgba(0,0,0,0.08)',
              whiteSpace: 'pre-wrap',
              fontSize: '0.95rem'
            }}>
              {msg.text}
            </div>
          ))}
          {loading && <div style={{ color: '#64748b', fontSize: '0.9rem' }}>Searching vector index & generating response...</div>}
        </div>

        <form onSubmit={handleQuery} style={{ padding: '16px 24px', background: '#fff', borderTop: '1px solid #e2e8f0', display: 'flex', gap: '12px' }}>
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a question across indexed documents..."
            style={{ flex: 1, padding: '10px 14px', borderRadius: '6px', border: '1px solid #cbd5e1', outline: 'none' }}
          />
          <button type="submit" disabled={loading} style={{ padding: '10px 20px', borderRadius: '6px', background: '#2563eb', color: '#fff', border: 'none', cursor: 'pointer', fontWeight: 600 }}>
            Send
          </button>
        </form>
      </div>

      {/* Right Drawer: Ingestion & Retrieved Context Sources */}
      <div style={{ width: '380px', padding: '24px', backgroundColor: '#fff', overflowY: 'auto' }}>
        <h3 style={{ marginTop: 0, fontSize: '1.1rem', color: '#0f172a' }}>Document Management</h3>
        <input type="file" accept=".pdf,.txt" onChange={handleFileUpload} style={{ marginBottom: '8px', width: '100%' }} />
        {uploadStatus && <div style={{ fontSize: '0.85rem', color: '#059669', marginBottom: '16px' }}>{uploadStatus}</div>}

        <hr style={{ border: 'none', borderTop: '1px solid #e2e8f0', margin: '20px 0' }} />

        <h3 style={{ fontSize: '1.1rem', color: '#0f172a' }}>Retrieved Context & Citations</h3>
        {sources.length === 0 ? (
          <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Ask a question to view retrieved document chunks and relevance scores.</p>
        ) : (
          sources.map((s, idx) => (
            <div key={idx} style={{ padding: '12px', border: '1px solid #e2e8f0', borderRadius: '6px', marginBottom: '12px', backgroundColor: '#f8fafc' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.8rem', fontWeight: 600, color: '#2563eb' }}>
                <span>{s.source} (p. {s.page})</span>
                <span>Score: {s.relevance_score}</span>
              </div>
              <p style={{ fontSize: '0.85rem', color: '#475569', marginTop: '6px', lineHeight: '1.4' }}>
                {s.content}
              </p>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
