import React, { useState, useRef, useEffect } from 'react';
import { signOut, fetchAuthSession } from 'aws-amplify/auth';
import { getConfig } from './aws-config';
import STLViewer from './STLViewer';

const DESIGN_PARAMS = [
  { name: 'ride_height', label: 'Ride Height', tooltip: 'Ground Clearance', min: 0.03, max: 0.10, step: 0.005, defaultValue: 0.05, unit: 'm' },
  { name: 'diffuser_angle', label: 'Diffuser Angle', tooltip: 'Underbody Exit', min: 0, max: 20, step: 1, defaultValue: 10, unit: '°' },
  { name: 'rear_slant', label: 'Rear Slant', tooltip: 'Roofline / Sportiness', min: 0, max: 35, step: 1, defaultValue: 25, unit: '°' },
  { name: 'front_overhang', label: 'Front Overhang', tooltip: 'Nose Length', min: 0.5, max: 1.2, step: 0.05, defaultValue: 0.85, unit: 'm' },
  { name: 'boat_tail_angle', label: 'Boat Tail Angle', tooltip: 'Rear Taper', min: 0, max: 25, step: 1, defaultValue: 12, unit: '°' },
];

const BODY_STYLES = [
  { value: 'sedan', label: 'Sedan', defaults: { ride_height: 0.05, diffuser_angle: 5, rear_slant: 15, front_overhang: 0.85, boat_tail_angle: 5 } },
  { value: 'sport', label: 'Sports', defaults: { ride_height: 0.04, diffuser_angle: 12, rear_slant: 30, front_overhang: 0.75, boat_tail_angle: 10 } },
  { value: 'suv', label: 'SUV', defaults: { ride_height: 0.08, diffuser_angle: 3, rear_slant: 10, front_overhang: 0.90, boat_tail_angle: 3 } },
  { value: 'hatchback', label: 'Hatchback', defaults: { ride_height: 0.05, diffuser_angle: 5, rear_slant: 35, front_overhang: 0.70, boat_tail_angle: 5 } },
  { value: 'mini_suv', label: 'Mini SUV', defaults: { ride_height: 0.07, diffuser_angle: 5, rear_slant: 18, front_overhang: 0.75, boat_tail_angle: 5 } },
];

const SAMPLE_QUESTIONS = {
  'Aerodynamics': [
    'Compare the top 5 variants by drag coefficient',
    'Which variant has the lowest Cd value?',
    'Which variants have Cd below 0.27?',
    'Compare run_50 and run_100 across all aero KPIs',
    'Compare run_20, run_80, run_140 by lift-to-drag ratio',
    'Which variant has the best lift-to-drag ratio?',
  ],
  'Visualization': [
    'Show me the surface pressure heatmap for run_0',
    'Generate surface friction heatmap for run_1',
    'Generate flow field slices for run_2',
    'Show velocity cross-sections for run_3',
  ],
  'Structural': [
    'What are the structural properties of run_0?',
    'Compute geometry metrics for run_1',
    'Compare weight and stiffness for the top 3 variants by Cd',
    'What is the vertex count and surface area for run_2?',
  ],
  'Cost': [
    'Estimate the manufacturing cost for run_0 using aluminum',
    'What is the cost breakdown for run_1 with steel?',
    'Compare manufacturing costs of run_5 vs run_8 in aluminum',
    'Which material gives the lowest cost for run_3?',
    'Estimate cost for run_9 using carbon fiber',
  ],
  'Geometry': [
    'Add side mirrors to run_0 and evaluate aerodynamics',
    'Add a rear spoiler to run_1',
    'Extend the bonnet of run_2 by 50mm',
    'Generate a design concept for a low-drag sedan',
  ],
  'General': [
    'How many design variants are cached?',
    'Summarize the best performing variant overall',
    'Recommend the best variant balancing cost and aerodynamics',
    'What agents are available in the system?',
  ],
};

// Section header for left panel
const PanelSection = ({ children }) => (
  <div style={{
    padding: '8px 12px 4px',
    fontSize: 10,
    fontWeight: 700,
    color: '#484f58',
    textTransform: 'uppercase',
    letterSpacing: '0.8px',
    borderTop: '1px solid #21262d',
    marginTop: 4,
  }}>
    {children}
  </div>
);

const Chat = ({ user, signOut: parentSignOut }) => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [websocket, setWebsocket] = useState(null);
  const messagesEndRef = useRef(null);
  const heartbeatRef = useRef(null);
  const [loadingMessage, setLoadingMessage] = useState('Sending to AI...');
  const loadingMessageRef = useRef(null);
  const [stlFile, setStlFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);
  const [stlViewerUrl, setStlViewerUrl] = useState(null);
  const [paramValues, setParamValues] = useState(
    Object.fromEntries(DESIGN_PARAMS.map(p => [p.name, p.defaultValue]))
  );
  const [bodyStyle, setBodyStyle] = useState('sedan');
  const [questionCategory, setQuestionCategory] = useState('Aerodynamics');
  const typewriterRef = useRef(null);
  const [uploadedS3Path, setUploadedS3Path] = useState(null);
  const [loadingProgress, setLoadingProgress] = useState(0);

  // New state for Design Explorer layout
  const [kpiData, setKpiData] = useState(null);
  const [showSampleQs, setShowSampleQs] = useState(false);
  const [variantInput, setVariantInput] = useState('125');
  // Keep activeTab for internal handler compatibility
  const [activeTab, setActiveTab] = useState('agent');

  const loadingMessages = [
    'Connecting to orchestrator...',
    'Discovering specialist agents...',
    'Routing to the right agent...',
    'Agent is processing your request...',
    'Running analysis...',
    'Generating visualizations...',
    'Synthesizing results...',
    'Preparing response...',
  ];

  // --- KPI extraction ---
  const extractKPIs = (text) => {
    const cd = text.match(/\bCd[:\s=]+([0-9.]+)/i)?.[1];
    const cs = text.match(/\bCs[:\s=]+([0-9.-]+)/i)?.[1];
    const cl = text.match(/\bCl[:\s=]+([0-9.-]+)/i)?.[1];
    const cmy = text.match(/\bCmy[:\s=]+([0-9.-]+)/i)?.[1];
    if (cd || cs || cl || cmy) return { cd, cs, cl, cmy };
    return null;
  };

  // --- Text formatting ---
  const formatText = (text) => {
    if (!text) return null;
    const lines = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n').split('\n');
    const elements = [];
    let i = 0;
    while (i < lines.length) {
      const line = lines[i];
      if (line.includes('|') && i + 1 < lines.length && /^\s*\|?\s*[-:]+[-|\s:]+\s*$/.test(lines[i + 1])) {
        const tableLines = [];
        let j = i;
        while (j < lines.length && lines[j].includes('|')) { tableLines.push(lines[j]); j++; }
        const parseRow = (row) => row.split('|').map(c => c.trim()).filter(c => c.length > 0);
        const headers = parseRow(tableLines[0]);
        const rows = tableLines.slice(2).map(parseRow);
        elements.push(
          <div key={`tbl-${i}`} style={{ overflowX: 'auto', margin: '8px 0' }}>
            <table style={{ borderCollapse: 'collapse', width: '100%', fontSize: 12, fontFamily: 'monospace' }}>
              <thead>
                <tr style={{ backgroundColor: '#21262d' }}>
                  {headers.map((h, hi) => (
                    <th key={hi} style={{ padding: '6px 10px', textAlign: 'left', color: '#e6edf3', fontWeight: 700, fontSize: 11, whiteSpace: 'nowrap', borderBottom: '1px solid #FF9900' }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row, ri) => (
                  <tr key={ri} style={{ backgroundColor: ri % 2 === 0 ? '#161b22' : '#0d1117' }}>
                    {row.map((cell, ci) => (
                      <td key={ci} style={{ padding: '5px 10px', borderBottom: '1px solid #21262d', color: '#c9d1d9', whiteSpace: 'nowrap', fontWeight: ci === 0 ? 600 : 400 }}>{cell}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        i = j; continue;
      }
      const parts = [];
      let lastIndex = 0;
      const boldRegex = /\*\*(.*?)\*\*/g;
      let match;
      while ((match = boldRegex.exec(line)) !== null) {
        if (match.index > lastIndex) parts.push(line.substring(lastIndex, match.index));
        parts.push(<strong key={`b-${i}-${match.index}`} style={{ color: '#e6edf3' }}>{match[1]}</strong>);
        lastIndex = match.index + match[0].length;
      }
      if (lastIndex < line.length) parts.push(line.substring(lastIndex));
      if (/^\d+\.\s/.test(line)) {
        const m = line.match(/^(\d+\.)\s(.*)/);
        elements.push(<div key={i} style={{ marginBottom: 3 }}><strong style={{ color: '#FF9900' }}>{m[1]}</strong> {m[2]}</div>);
      } else if (/^[-•]\s/.test(line)) {
        elements.push(<div key={i} style={{ marginBottom: 3, paddingLeft: 10, borderLeft: '2px solid #30363d' }}>{line.substring(2)}</div>);
      } else {
        elements.push(<div key={i} style={{ marginBottom: line.trim() ? 3 : 6 }}>{parts.length > 0 ? parts : (line || '\u00A0')}</div>);
      }
      i++;
    }
    return elements;
  };

  const extractImages = (text) => {
    const tagged = [...text.matchAll(/\[IMAGE\]([^\[]+)\[\/IMAGE\]/gi)].map(m => m[1].trim());
    const unclosed = [...text.matchAll(/\[IMAGE\](https?:\/\/[^\s\[]+)/gi)].map(m => m[1].trim()).filter(u => !tagged.includes(u));
    const urls = (text.match(/(https?:\/\/[^\s)\]]+\.(?:jpg|jpeg|png|gif|webp|svg)[^\s)\]]*)/gi) || []).map(u => u.replace(/&amp;/g, '&'));
    return [...new Set([...tagged, ...unclosed, ...urls])];
  };

  const cleanImageTags = (text) => text
    .replace(/\[IMAGE\][^\[]+\[\/IMAGE\]/gi, '')
    .replace(/\[IMAGE\](https?:\/\/[^\s\[]*)/gi, '')
    .replace(/(https?:\/\/[^\s)\]]+\.(?:jpg|jpeg|png|gif|webp|svg)[^\s)\]]*)/gi, '')
    .trim();

  const extractSTLUrls = (text) => {
    const closed = [...text.matchAll(/\[STL\]([^\[]+)\[\/STL\]/gi)].map(m => m[1].trim());
    const unclosed = [...text.matchAll(/\[STL\](https?:\/\/[^\s\[]+)/gi)].map(m => m[1].trim()).filter(u => !closed.includes(u));
    return [...new Set([...closed, ...unclosed])];
  };

  const cleanSTLTags = (text) => text
    .replace(/\[STL\][^\[]+\[\/STL\]/gi, '')
    .replace(/\[STL\](https?:\/\/[^\s\[]*)/gi, '')
    .trim();

  // --- WebSocket ---
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  useEffect(() => {
    const connectWebSocket = async () => {
      try {
        const session = await fetchAuthSession();
        const accessToken = session.tokens?.accessToken?.toString();
        const cfg = getConfig();
        const wsUrl = `${cfg.websocketUrl}?token=${accessToken}`;
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;
        ws.onopen = () => {
          setWebsocket(ws);
          if (heartbeatRef.current) clearInterval(heartbeatRef.current);
          heartbeatRef.current = setInterval(() => {
            if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ action: 'ping', userId: user.username }));
          }, 25000);
        };
        ws.onmessage = (event) => {
          let data;
          try {
            data = JSON.parse(event.data);
          } catch (err) {
            console.error('Invalid WebSocket response:', err);
            clearTimers(); setLoading(false);
            setMessages(prev => [...prev, { text: 'Error: The server returned an invalid response.', sender: 'System', timestamp: new Date(), type: 'error' }]);
            return;
          }
          if (data.action === 'pong') return;
          if (data.type === 'error' || data.status === 'error' || data.error) {
            clearTimers(); setLoading(false);
            const errorText = data.error || data.response || data.message || 'Unknown error';
            setMessages(prev => [...prev, { text: `Error: ${errorText}`, sender: 'System', timestamp: new Date(), type: 'error' }]);
            return;
          }
          if (data.type === 'processing') {
            setLoadingMessage(data.message || 'Processing your request...');
            setLoadingProgress(prev => Math.min(prev + 10, 85));
            return;
          }
          if (data.type === 'chunk') {
            setMessages(prev => {
              const msgs = [...prev];
              const last = msgs[msgs.length - 1];
              if (last && last.type === 'agent' && last.streaming) {
                last.text += data.chunk;
              } else {
                msgs.push({ text: data.chunk, sender: 'Agent', timestamp: new Date(), type: 'agent', streaming: true, images: [] });
              }
              return msgs;
            });
            return;
          }
          if (data.type === 'tool_start') { setLoadingMessage('Executing analysis...'); setLoadingProgress(prev => Math.min(prev + 10, 85)); return; }
          if (data.type === 'stl_uploaded') {
            clearTimers(); setLoading(false);
            const s3Path = data.s3Path || '';
            const fileName = data.fileName || 'uploaded.stl';
            setUploadedS3Path(s3Path);
            setMessages(prev => [...prev, { text: '', sender: 'Agent', timestamp: new Date(), type: 'stl_action_card', s3Path, fileName }]);
            setActiveTab('agent');
            return;
          }
          if (data.type === 'complete' || data.type === 'response') {
            clearTimers(); setLoading(false);
            const raw = (data.response || 'No response from agent')
              .replace(/&#39;/g, "'").replace(/&quot;/g, '"').replace(/&amp;/g, '&').replace(/&lt;/g, '<').replace(/&gt;/g, '>');
            const images = extractImages(raw);
            const stlUrls = extractSTLUrls(raw);
            const cleanText = cleanSTLTags(cleanImageTags(raw));
            if (stlUrls.length > 0) { setStlViewerUrl(stlUrls[0]); setActiveTab('3dview'); }
            // Extract KPIs from response
            const kpis = extractKPIs(raw);
            if (kpis) setKpiData(kpis);
            const hasTable = cleanText.split('\n').some(line => /^\s*\|[\s:-]+\|/.test(line));
            const hasImages = images.length > 0;
            const stlUrl = stlUrls.length > 0 ? stlUrls[0] : null;
            if (hasTable || hasImages) {
              setMessages(prev => {
                const msgs = [...prev];
                const last = msgs[msgs.length - 1];
                if (last && last.streaming) {
                  last.text = cleanText; last.streaming = false; last.images = images; last.stlUrl = stlUrl;
                  delete last._fullText; delete last._charIndex;
                } else {
                  msgs.push({ text: cleanText, sender: 'Agent', timestamp: new Date(), type: 'agent', streaming: false, images, stlUrl });
                }
                return [...msgs];
              });
            } else {
              startTypewriter(cleanText, images, stlUrl);
            }
            return;
          }
          if (data.response) {
            clearTimers(); setLoading(false);
            const images = extractImages(data.response);
            const stlUrls = extractSTLUrls(data.response);
            const cleanText = cleanSTLTags(cleanImageTags(data.response));
            if (stlUrls.length > 0) { setStlViewerUrl(stlUrls[0]); setActiveTab('3dview'); }
            const kpis = extractKPIs(data.response);
            if (kpis) setKpiData(kpis);
            const stlUrl = stlUrls.length > 0 ? stlUrls[0] : null;
            setMessages(prev => [...prev, { text: cleanText, sender: 'Agent', timestamp: new Date(), type: 'agent', images, stlUrl }]);
          }
        };
        ws.onerror = () => { setLoading(false); };
        ws.onclose = () => {
          setWebsocket(null);
          if (heartbeatRef.current) { clearInterval(heartbeatRef.current); heartbeatRef.current = null; }
          reconnectTimerRef.current = setTimeout(() => { connectWebSocket(); }, 3000);
        };
      } catch (err) { console.error('WebSocket connect failed:', err); }
    };
    connectWebSocket();
    return () => {
      clearTimers();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const clearTimers = () => {
    if (heartbeatRef.current) { clearInterval(heartbeatRef.current); heartbeatRef.current = null; }
    if (loadingMessageRef.current) { clearInterval(loadingMessageRef.current); loadingMessageRef.current = null; }
    if (typewriterRef.current) { clearInterval(typewriterRef.current); typewriterRef.current = null; }
    setLoadingProgress(100);
    setTimeout(() => setLoadingProgress(0), 300);
  };

  const startTypewriter = (fullText, images, stlUrl = null) => {
    if (typewriterRef.current) { clearInterval(typewriterRef.current); typewriterRef.current = null; }
    setMessages(prev => {
      const msgs = [...prev];
      const last = msgs[msgs.length - 1];
      if (last && last.streaming) {
        last.text = ''; last._fullText = fullText; last._charIndex = 0; last.images = images; last.stlUrl = stlUrl;
      } else {
        msgs.push({ text: '', sender: 'Agent', timestamp: new Date(), type: 'agent', streaming: true, images, stlUrl, _fullText: fullText, _charIndex: 0 });
      }
      return msgs;
    });
    const charsPerTick = fullText.length > 1000 ? 12 : fullText.length > 500 ? 8 : 4;
    typewriterRef.current = setInterval(() => {
      setMessages(prev => {
        const msgs = [...prev];
        const last = msgs[msgs.length - 1];
        if (!last || !last.streaming || !last._fullText) { clearInterval(typewriterRef.current); typewriterRef.current = null; return msgs; }
        const nextIndex = Math.min(last._charIndex + charsPerTick, last._fullText.length);
        last.text = last._fullText.substring(0, nextIndex);
        last._charIndex = nextIndex;
        if (nextIndex >= last._fullText.length) {
          last.streaming = false; delete last._fullText; delete last._charIndex;
          clearInterval(typewriterRef.current); typewriterRef.current = null;
        }
        return [...msgs];
      });
    }, 16);
  };

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages]);

  const sendMessage = async (overrideText, displayText) => {
    if (isProcessing) return;
    const text = overrideText || input;
    const shown = displayText || text;
    if (!text.trim()) return;
    setMessages(prev => [...prev, { text: shown, sender: user.username || 'You', timestamp: new Date(), type: 'user' }]);
    setLoading(true);
    setLoadingMessage('Connecting to orchestrator...');
    setLoadingProgress(5);
    if (!overrideText) setInput('');
    let msgIndex = 0;
    loadingMessageRef.current = setInterval(() => {
      msgIndex = (msgIndex + 1) % loadingMessages.length;
      setLoadingMessage(loadingMessages[msgIndex]);
      setLoadingProgress(prev => Math.min(prev + Math.floor(Math.random() * 8 + 5), 90));
    }, 4000);
    try {
      const session = await fetchAuthSession();
      const accessToken = session.tokens?.accessToken?.toString();
      if (websocket && websocket.readyState === WebSocket.OPEN) {
        websocket.send(JSON.stringify({ message: text, userId: user.username, bearerToken: accessToken }));
      } else { throw new Error('WebSocket not connected'); }
    } catch (err) {
      console.error('Send error:', err);
      setMessages(prev => [...prev, { text: 'Error: Could not reach the agent', sender: 'System', timestamp: new Date(), type: 'error' }]);
      setLoading(false); clearTimers();
    }
  };

  const handleEvaluateDesign = () => {
    const styleName = BODY_STYLES.find(s => s.value === bodyStyle)?.label || 'Sedan';
    const paramStr = DESIGN_PARAMS.map(p => `${p.label}: ${paramValues[p.name].toFixed(p.step < 1 ? 3 : 0)}${p.unit}`).join(', ');
    sendMessage(`Generate a ${styleName} car design with these parameters: ${paramStr}. Ensure it has visible wheels and a defined cabin. What would be the expected aerodynamic performance?`);
  };

  const handleSTLUpload = async (file) => {
    if (!file || !file.name.toLowerCase().endsWith('.stl')) { alert('Please select an STL file'); return; }
    setStlFile(file);
    setUploadProgress('uploading');
    setStlViewerUrl(URL.createObjectURL(file));
    if (websocket && websocket.readyState === WebSocket.OPEN) {
      setLoading(true); setLoadingMessage('Uploading STL file to storage...');
      try {
        const session = await fetchAuthSession();
        const accessToken = session.tokens?.accessToken?.toString();
        const reader = new FileReader();
        reader.onload = () => {
          const base64 = reader.result.split(',')[1];
          websocket.send(JSON.stringify({ userId: user.username, bearerToken: accessToken, stlFile: { name: file.name, data: base64, size: file.size } }));
          setUploadProgress('sent');
          setMessages(prev => [...prev, { text: `Uploaded STL: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`, sender: user.username || 'You', timestamp: new Date(), type: 'user' }]);
        };
        reader.readAsDataURL(file);
      } catch (err) { console.error('Upload error:', err); setUploadProgress(null); setLoading(false); clearTimers(); }
    }
  };

  const handleDrop = (e) => { e.preventDefault(); setDragOver(false); if (e.dataTransfer.files[0]) handleSTLUpload(e.dataTransfer.files[0]); };
  const handleSignOut = async () => { try { await signOut(); if (parentSignOut) parentSignOut(); } catch (err) { console.log('Sign out error:', err); } };
  const isProcessing = loading || messages.some(msg => msg.streaming);

  const toS3Uri = (urlOrUri) => {
    if (!urlOrUri) return urlOrUri;
    if (urlOrUri.startsWith('s3://')) return urlOrUri;
    // Convert presigned HTTPS URL to s3:// URI — never expose credentials in queries
    try {
      const u = new URL(urlOrUri);
      if (u.hostname.includes('.s3.') && u.hostname.includes('amazonaws.com')) {
        const bucket = u.hostname.split('.s3.')[0];
        const key = u.pathname.startsWith('/') ? u.pathname.slice(1) : u.pathname;
        return `s3://${bucket}/${key}`;
      }
    } catch (_) {}
    return urlOrUri;
  };

  const handleSTLAction = (action, s3Path, fileName) => {
    const uri = toS3Uri(s3Path);
    const label = fileName && fileName !== 'generated.stl' ? fileName : 'this design';
    const display = {
      aero:         `Aerodynamic KPI analysis for ${label}`,
      surface:      `Surface pressure heatmap for ${label}`,
      slices:       `Flow field slices for ${label}`,
      structural:   `Structural evaluation for ${label}`,
      cost_aluminum:`Manufacturing cost estimate (aluminum) for ${label}`,
      cost_steel:   `Manufacturing cost estimate (steel) for ${label}`,
      cost_carbon:  `Manufacturing cost estimate (carbon fiber) for ${label}`,
      full:         `Full pipeline analysis for ${label}`,
    };
    const queries = {
      aero:         `Run aerodynamic analysis on the uploaded geometry at ${uri}`,
      surface:      `Show surface pressure and friction heatmap for the geometry at ${uri}`,
      slices:       `Show velocity flow field slices for the geometry at ${uri}`,
      structural:   `Evaluate structural feasibility for the uploaded geometry at ${uri} using aluminum`,
      cost_aluminum:`Estimate manufacturing cost for the uploaded geometry at ${uri} using aluminum`,
      cost_steel:   `Estimate manufacturing cost for the uploaded geometry at ${uri} using steel`,
      cost_carbon:  `Estimate manufacturing cost for the uploaded geometry at ${uri} using carbon fiber`,
      full:         `Run full pipeline analysis (aero, structural, and cost) on the uploaded geometry at ${uri} using aluminum`,
    };
    sendMessage(queries[action] || queries.aero, display[action] || display.aero);
  };

  // ─────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────
  return (
    <>
      <style>{`
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: #484f58; }
        @keyframes pulse { 0%, 80%, 100% { opacity: 0.2; } 40% { opacity: 1; } }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }
        input[type="range"] { -webkit-appearance: none; appearance: none; background: #30363d; border-radius: 3px; height: 4px; outline: none; }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance: none; appearance: none; width: 14px; height: 14px; border-radius: 50%; background: #FF9900; cursor: pointer; box-shadow: 0 0 6px rgba(255,153,0,0.5); }
        input[type="range"]::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: #FF9900; cursor: pointer; border: none; }
        .msg-row { animation: fadeIn 0.2s ease; }
        .q-item:hover { border-color: #FF9900 !important; color: #e6edf3 !important; background: rgba(255,153,0,0.08) !important; }
        .action-btn:hover { border-color: #FF9900 !important; background: rgba(255,153,0,0.1) !important; }
        .preset-btn:hover { border-color: #FF9900 !important; color: #FF9900 !important; }
        .style-btn:hover { border-color: #FF9900 !important; color: #FF9900 !important; }
      `}</style>

      <div style={{
        height: '100vh', display: 'flex', flexDirection: 'column',
        background: '#0d1117', color: '#e6edf3', overflow: 'hidden',
        fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
      }}>

        {/* ═══ TOP HEADER ═══ */}
        <header style={{
          height: 48, background: '#161b22', borderBottom: '1px solid #30363d',
          display: 'flex', alignItems: 'center', padding: '0 16px', gap: 12,
          flexShrink: 0, zIndex: 10,
        }}>
          <span style={{ fontSize: 20 }}>🏎</span>
          <span style={{ fontSize: 15, fontWeight: 700, color: '#e6edf3', letterSpacing: '-0.3px' }}>
            Car Design Space Explorer
          </span>
          <span style={{
            fontSize: 11, color: '#58a6ff', background: 'rgba(88,166,255,0.1)',
            border: '1px solid rgba(88,166,255,0.25)', borderRadius: 10, padding: '2px 8px', fontWeight: 500,
          }}>
            Amazon Bedrock AgentCore
          </span>
          <div style={{ flex: 1 }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12 }}>
            <div style={{
              width: 7, height: 7, borderRadius: '50%',
              background: websocket ? '#3fb950' : '#f85149',
              boxShadow: websocket ? '0 0 6px #3fb950' : '0 0 6px #f85149',
            }} />
            <span style={{ color: websocket ? '#3fb950' : '#f85149' }}>
              {websocket ? 'Connected' : 'Reconnecting...'}
            </span>
          </div>
          <div style={{ width: 1, height: 20, background: '#30363d' }} />
          <div
            onClick={handleSignOut}
            style={{
              display: 'flex', alignItems: 'center', gap: 8, padding: '4px 10px',
              borderRadius: 6, background: '#21262d', border: '1px solid #30363d',
              cursor: 'pointer', transition: 'border-color 0.15s',
            }}
            onMouseEnter={e => e.currentTarget.style.borderColor = '#484f58'}
            onMouseLeave={e => e.currentTarget.style.borderColor = '#30363d'}
          >
            <div style={{
              width: 22, height: 22, borderRadius: 4,
              background: 'linear-gradient(135deg, #FF9900, #FF6600)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontSize: 11, fontWeight: 700, color: '#fff',
            }}>
              {(user.username || 'U').charAt(0).toUpperCase()}
            </div>
            <span style={{ fontSize: 13, color: '#c9d1d9' }}>{user.username || 'User'}</span>
            <span style={{ fontSize: 11, color: '#484f58' }}>Sign out</span>
          </div>
        </header>

        {/* ═══ 3-COLUMN BODY ═══ */}
        <div style={{ flex: 1, display: 'flex', overflow: 'hidden' }}>

          {/* ── LEFT PANEL: Design Controls ── */}
          <div style={{
            width: 240, background: '#161b22', borderRight: '1px solid #30363d',
            display: 'flex', flexDirection: 'column', overflow: 'hidden', flexShrink: 0,
          }}>
            <div style={{ flex: 1, overflowY: 'auto', paddingBottom: 16 }}>

              {/* Body Style */}
              <PanelSection>Body Style</PanelSection>
              <div style={{ padding: '6px 12px 8px', display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {BODY_STYLES.map(s => (
                  <button
                    key={s.value}
                    className="style-btn"
                    onClick={() => { setBodyStyle(s.value); setParamValues(prev => ({ ...prev, ...s.defaults })); }}
                    style={{
                      padding: '4px 10px', borderRadius: 5, cursor: 'pointer', fontSize: 11, fontWeight: 500,
                      transition: 'all 0.15s', border: bodyStyle === s.value ? '1px solid #FF9900' : '1px solid #30363d',
                      background: bodyStyle === s.value ? 'rgba(255,153,0,0.12)' : '#21262d',
                      color: bodyStyle === s.value ? '#FF9900' : '#8b949e',
                    }}
                  >
                    {s.label}
                  </button>
                ))}
              </div>

              {/* Parameters */}
              <PanelSection>Parameters</PanelSection>
              <div style={{ padding: '8px 12px' }}>
                {DESIGN_PARAMS.map(p => (
                  <div key={p.name} style={{ marginBottom: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 5 }}>
                      <label style={{ fontSize: 11, color: '#8b949e', lineHeight: 1 }}>{p.label}</label>
                      <span style={{
                        fontSize: 11, fontFamily: 'monospace', fontWeight: 700, color: '#FF9900',
                        background: 'rgba(255,153,0,0.1)', border: '1px solid rgba(255,153,0,0.2)',
                        borderRadius: 4, padding: '1px 6px', lineHeight: 1.6,
                      }}>
                        {paramValues[p.name].toFixed(p.step < 1 ? 3 : 0)}{p.unit}
                      </span>
                    </div>
                    <input
                      type="range" min={p.min} max={p.max} step={p.step}
                      value={paramValues[p.name]}
                      onChange={e => setParamValues(prev => ({ ...prev, [p.name]: parseFloat(e.target.value) }))}
                      style={{ width: '100%', cursor: 'pointer' }}
                    />
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
                      <span style={{ fontSize: 10, color: '#484f58' }}>{p.min}{p.unit}</span>
                      <span style={{ fontSize: 10, color: '#484f58' }}>{p.max}{p.unit}</span>
                    </div>
                  </div>
                ))}
                <div style={{ display: 'flex', gap: 6, marginTop: 4 }}>
                  <button
                    onClick={handleEvaluateDesign}
                    disabled={isProcessing}
                    style={{
                      flex: 1, padding: '7px 10px', borderRadius: 6, border: 'none',
                      background: isProcessing ? '#30363d' : 'linear-gradient(135deg, #FF9900 0%, #FF6600 100%)',
                      color: 'white', fontWeight: 700, fontSize: 12, cursor: isProcessing ? 'not-allowed' : 'pointer',
                      boxShadow: isProcessing ? 'none' : '0 2px 8px rgba(255,153,0,0.3)',
                    }}
                  >
                    Evaluate Design
                  </button>
                  <button
                    onClick={() => { setParamValues(Object.fromEntries(DESIGN_PARAMS.map(p => [p.name, p.defaultValue]))); setBodyStyle('sedan'); }}
                    style={{
                      padding: '7px 10px', borderRadius: 6, border: '1px solid #30363d',
                      background: '#21262d', color: '#8b949e', fontSize: 12, cursor: 'pointer',
                    }}
                  >
                    Reset
                  </button>
                </div>
              </div>

              {/* Quick Presets */}
              <PanelSection>Quick Presets</PanelSection>
              <div style={{ padding: '6px 12px 8px', display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                {[
                  { label: 'Low Drag', values: { ride_height: 0.04, diffuser_angle: 15, rear_slant: 12, front_overhang: 0.75, boat_tail_angle: 8 } },
                  { label: 'High Downforce', values: { ride_height: 0.035, diffuser_angle: 18, rear_slant: 30, front_overhang: 0.9, boat_tail_angle: 20 } },
                  { label: 'Balanced', values: { ride_height: 0.05, diffuser_angle: 10, rear_slant: 25, front_overhang: 0.85, boat_tail_angle: 12 } },
                  { label: 'Easy Mfg', values: { ride_height: 0.06, diffuser_angle: 5, rear_slant: 20, front_overhang: 0.8, boat_tail_angle: 10 } },
                ].map((preset, i) => (
                  <button
                    key={i}
                    className="preset-btn"
                    onClick={() => setParamValues(prev => ({ ...prev, ...preset.values }))}
                    style={{
                      padding: '4px 10px', borderRadius: 5, border: '1px solid #30363d',
                      background: '#21262d', color: '#8b949e', fontSize: 11, cursor: 'pointer', transition: 'all 0.15s',
                    }}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>

              {/* WindsorML Variant Loader */}
              <PanelSection>WindsorML Variant</PanelSection>
              <div style={{ padding: '6px 12px 8px' }}>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  <div style={{
                    flex: 1, display: 'flex', alignItems: 'center', gap: 4,
                    background: '#21262d', border: '1px solid #30363d', borderRadius: 6, padding: '5px 10px',
                  }}>
                    <span style={{ fontSize: 11, color: '#484f58', fontFamily: 'monospace', whiteSpace: 'nowrap' }}>run_</span>
                    <input
                      type="number" min={1} max={355} value={variantInput}
                      onChange={e => setVariantInput(e.target.value)}
                      style={{
                        flex: 1, background: 'transparent', border: 'none', outline: 'none',
                        color: '#e6edf3', fontSize: 13, fontFamily: 'monospace', width: '100%',
                      }}
                    />
                  </div>
                  <button
                    onClick={() => { if (variantInput) sendMessage(`Show me the 3D model for run_${variantInput}`); }}
                    disabled={isProcessing}
                    style={{
                      padding: '6px 12px', borderRadius: 6, border: 'none',
                      background: isProcessing ? '#30363d' : '#1f6feb',
                      color: 'white', fontSize: 12, fontWeight: 600, cursor: isProcessing ? 'not-allowed' : 'pointer',
                    }}
                  >
                    Load
                  </button>
                </div>
                <div style={{ fontSize: 10, color: '#484f58', marginTop: 5 }}>
                  355 variants available (run_1 – run_355)
                </div>
              </div>

              {/* Upload Geometry */}
              <PanelSection>Upload Geometry</PanelSection>
              <div style={{ padding: '6px 12px 8px' }}>
                <div
                  onDragOver={e => { e.preventDefault(); setDragOver(true); }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  style={{
                    border: `1px dashed ${dragOver ? '#FF9900' : '#30363d'}`,
                    borderRadius: 8, padding: '14px 10px', cursor: 'pointer', textAlign: 'center',
                    background: dragOver ? 'rgba(255,153,0,0.06)' : 'transparent', transition: 'all 0.2s',
                  }}
                >
                  <div style={{ fontSize: 22, marginBottom: 4, opacity: 0.6 }}>📐</div>
                  <div style={{ fontSize: 11, color: stlFile ? '#3fb950' : '#6e7681', lineHeight: 1.4 }}>
                    {stlFile
                      ? <><span style={{ color: '#3fb950', fontWeight: 600 }}>✓ {stlFile.name}</span><br /><span style={{ fontSize: 10, color: '#484f58' }}>{(stlFile.size / 1024).toFixed(1)} KB</span></>
                      : 'Drop .STL file or click to browse'
                    }
                  </div>
                  <input ref={fileInputRef} type="file" accept=".stl" style={{ display: 'none' }}
                    onChange={e => { if (e.target.files[0]) handleSTLUpload(e.target.files[0]); }} />
                </div>
              </div>

            </div>
          </div>

          {/* ── CENTER: 3D Viewport ── */}
          <div style={{
            flex: 1, display: 'flex', flexDirection: 'column',
            background: '#0d1117', position: 'relative', overflow: 'hidden',
          }}>
            {/* Viewport toolbar */}
            <div style={{
              height: 38, background: '#161b22', borderBottom: '1px solid #30363d',
              display: 'flex', alignItems: 'center', padding: '0 14px', gap: 10, flexShrink: 0,
            }}>
              <span style={{ fontSize: 11, color: '#8b949e', fontFamily: 'monospace' }}>
                {stlFile ? stlFile.name : stlViewerUrl ? 'Model loaded' : '— no model —'}
              </span>
              {stlViewerUrl && (
                <span style={{
                  fontSize: 10, color: '#3fb950', background: 'rgba(63,185,80,0.1)',
                  border: '1px solid rgba(63,185,80,0.3)', borderRadius: 4, padding: '1px 6px', fontWeight: 600,
                }}>LIVE</span>
              )}
              <div style={{ flex: 1 }} />
              <span style={{ fontSize: 10, color: '#484f58' }}>
                Left drag: rotate · Right drag: pan · Scroll: zoom
              </span>
            </div>

            {/* 3D Viewer */}
            <div style={{ flex: 1, position: 'relative', overflow: 'hidden' }}>
              {stlViewerUrl ? (
                <STLViewer url={stlViewerUrl} />
              ) : (
                <div style={{
                  height: '100%', display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center', userSelect: 'none',
                }}>
                  <div style={{ fontSize: 72, opacity: 0.15, marginBottom: 20 }}>🏎</div>
                  <div style={{ fontSize: 15, fontWeight: 600, color: '#484f58', marginBottom: 8 }}>
                    No model loaded
                  </div>
                  <div style={{ fontSize: 12, color: '#30363d', maxWidth: 260, textAlign: 'center', lineHeight: 1.7 }}>
                    Load a WindsorML variant from the left panel, upload an STL file, or ask the AI to generate a design concept.
                  </div>
                  <div style={{ marginTop: 24, display: 'flex', gap: 8 }}>
                    {['run_0', 'run_1', 'run_2'].map(v => (
                      <button
                        key={v}
                        onClick={() => sendMessage(`Show me the 3D model for ${v}`)}
                        style={{
                          padding: '6px 14px', borderRadius: 6, border: '1px solid #30363d',
                          background: '#161b22', color: '#8b949e', fontSize: 12, cursor: 'pointer',
                          transition: 'all 0.15s', fontFamily: 'monospace',
                        }}
                        onMouseEnter={e => { e.currentTarget.style.borderColor = '#FF9900'; e.currentTarget.style.color = '#FF9900'; }}
                        onMouseLeave={e => { e.currentTarget.style.borderColor = '#30363d'; e.currentTarget.style.color = '#8b949e'; }}
                      >
                        {v}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* KPI overlay bar */}
            {kpiData && (
              <div style={{
                position: 'absolute', bottom: 0, left: 0, right: 0,
                background: 'rgba(13,17,23,0.92)', backdropFilter: 'blur(10px)',
                borderTop: '1px solid #30363d', padding: '10px 16px',
                display: 'flex', gap: 0, alignItems: 'stretch', flexShrink: 0,
              }}>
                <div style={{ fontSize: 9, color: '#484f58', textTransform: 'uppercase', letterSpacing: '1px', fontWeight: 700, display: 'flex', alignItems: 'center', marginRight: 20, whiteSpace: 'nowrap' }}>
                  KPI Results
                </div>
                {[
                  { label: 'Drag Cd', value: kpiData.cd, good: v => parseFloat(v) < 0.30 },
                  { label: 'Side Cs', value: kpiData.cs, good: v => Math.abs(parseFloat(v)) < 0.05 },
                  { label: 'Lift Cl', value: kpiData.cl, good: v => parseFloat(v) < 0 },
                  { label: 'Pitch Cmy', value: kpiData.cmy, good: v => Math.abs(parseFloat(v)) < 0.10 },
                ].filter(k => k.value).map((kpi, i, arr) => (
                  <div key={kpi.label} style={{
                    display: 'flex', flexDirection: 'column', gap: 2, paddingRight: 24,
                    marginRight: 24, borderRight: i < arr.length - 1 ? '1px solid #21262d' : 'none',
                  }}>
                    <span style={{ fontSize: 9, color: '#6e7681', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
                      {kpi.label}
                    </span>
                    <span style={{
                      fontSize: 22, fontFamily: 'monospace', fontWeight: 700, lineHeight: 1,
                      color: kpi.good(kpi.value) ? '#3fb950' : '#FF9900',
                    }}>
                      {kpi.value}
                    </span>
                  </div>
                ))}
                <div style={{ flex: 1 }} />
                <button
                  onClick={() => setKpiData(null)}
                  style={{ background: 'none', border: 'none', color: '#484f58', cursor: 'pointer', fontSize: 16, padding: '0 4px', alignSelf: 'center' }}
                  title="Dismiss"
                >
                  ×
                </button>
              </div>
            )}
          </div>

          {/* ── RIGHT PANEL: AI Assistant ── */}
          <div style={{
            width: 480, background: '#161b22', borderLeft: '1px solid #30363d',
            display: 'flex', flexDirection: 'column', overflow: 'hidden', flexShrink: 0,
          }}>
            {/* Panel header */}
            <div style={{
              height: 38, borderBottom: '1px solid #30363d',
              display: 'flex', alignItems: 'center', padding: '0 12px', gap: 8, flexShrink: 0,
            }}>
              <div style={{
                width: 22, height: 22, borderRadius: 4,
                background: 'linear-gradient(135deg, #FF9900, #FF6600)',
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 10, fontWeight: 700, color: '#fff',
              }}>AI</div>
              <span style={{ fontSize: 13, fontWeight: 600, color: '#e6edf3' }}>AI Assistant</span>
              <div style={{ flex: 1 }} />
              <button
                onClick={() => setShowSampleQs(s => !s)}
                style={{
                  padding: '3px 9px', borderRadius: 4, border: `1px solid ${showSampleQs ? 'rgba(255,153,0,0.4)' : '#30363d'}`,
                  background: showSampleQs ? 'rgba(255,153,0,0.12)' : '#21262d',
                  color: showSampleQs ? '#FF9900' : '#6e7681', fontSize: 11, cursor: 'pointer', transition: 'all 0.15s',
                }}
              >
                {showSampleQs ? '▲' : '▼'} Examples
              </button>
            </div>

            {/* Sample questions (collapsible) */}
            {showSampleQs && (
              <div style={{ borderBottom: '1px solid #30363d', background: '#0d1117', flexShrink: 0 }}>
                {/* Category tabs */}
                <div style={{ padding: '8px 12px 6px', display: 'flex', gap: 5, flexWrap: 'wrap' }}>
                  {Object.keys(SAMPLE_QUESTIONS).map(cat => (
                    <button
                      key={cat}
                      onClick={() => setQuestionCategory(cat)}
                      style={{
                        padding: '3px 9px', borderRadius: 10, border: 'none',
                        background: questionCategory === cat ? 'rgba(255,153,0,0.15)' : '#21262d',
                        color: questionCategory === cat ? '#FF9900' : '#6e7681',
                        fontSize: 11, fontWeight: questionCategory === cat ? 600 : 400, cursor: 'pointer',
                      }}
                    >
                      {cat}
                    </button>
                  ))}
                </div>
                {/* Questions list */}
                <div style={{ maxHeight: 180, overflowY: 'auto', padding: '4px 12px 10px', display: 'flex', flexDirection: 'column', gap: 4 }}>
                  {(SAMPLE_QUESTIONS[questionCategory] || []).map((q, i) => (
                    <div
                      key={i}
                      className="q-item"
                      onClick={() => { setInput(q); setShowSampleQs(false); }}
                      style={{
                        padding: '6px 10px', borderRadius: 5, background: '#21262d',
                        fontSize: 12, color: '#8b949e', cursor: 'pointer', lineHeight: 1.4,
                        transition: 'all 0.15s', border: '1px solid transparent',
                      }}
                    >
                      {q}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Messages */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '12px', display: 'flex', flexDirection: 'column', gap: 10 }}>
              {messages.length === 0 && !loading && (
                <div style={{ textAlign: 'center', padding: '32px 12px', color: '#484f58' }}>
                  <div style={{ fontSize: 32, marginBottom: 10, opacity: 0.5 }}>💬</div>
                  <div style={{ fontSize: 12, lineHeight: 1.7, maxWidth: 260, margin: '0 auto' }}>
                    Ask about aerodynamic performance, structural feasibility, manufacturing costs, or design modifications.
                  </div>
                  <div style={{ marginTop: 16, display: 'flex', flexDirection: 'column', gap: 5 }}>
                    {['Compare top 5 by drag coefficient', 'Estimate cost for run_125', 'Show surface pressure for run_0'].map((q, i) => (
                      <div
                        key={i}
                        className="q-item"
                        onClick={() => sendMessage(q)}
                        style={{
                          padding: '6px 10px', borderRadius: 5, background: '#21262d',
                          fontSize: 11, color: '#6e7681', cursor: 'pointer', lineHeight: 1.4,
                          transition: 'all 0.15s', border: '1px solid transparent', textAlign: 'left',
                        }}
                      >
                        {q}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {messages.map((msg, index) => (
                <div key={index} className="msg-row" style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  {/* Sender row */}
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{
                      width: 18, height: 18, borderRadius: 3, flexShrink: 0,
                      display: 'flex', alignItems: 'center', justifyContent: 'center',
                      fontSize: 9, fontWeight: 700, color: '#fff',
                      background: msg.type === 'user'
                        ? '#21262d'
                        : msg.type === 'error'
                          ? '#da3633'
                          : 'linear-gradient(135deg, #FF9900, #FF6600)',
                      border: msg.type === 'user' ? '1px solid #30363d' : 'none',
                    }}>
                      {msg.type === 'user' ? (user.username || 'U').charAt(0).toUpperCase() : msg.type === 'error' ? '!' : 'AI'}
                    </div>
                    <span style={{
                      fontSize: 11, fontWeight: 600,
                      color: msg.type === 'user' ? '#8b949e' : msg.type === 'error' ? '#f85149' : '#FF9900',
                    }}>
                      {msg.type === 'user' ? (user.username || 'You') : msg.type === 'error' ? 'System' : 'AI Assistant'}
                    </span>
                    <span style={{ fontSize: 10, color: '#484f58' }}>
                      {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    </span>
                  </div>

                  {/* Message body */}
                  <div style={{
                    marginLeft: 24, fontSize: 12, lineHeight: 1.65,
                    color: msg.type === 'error' ? '#f85149' : msg.type === 'user' ? '#c9d1d9' : '#e6edf3',
                  }}>
                    {msg.type === 'stl_action_card' ? (
                      <div>
                        <div style={{ marginBottom: 8, color: '#8b949e', fontSize: 12 }}>
                          <span style={{ color: '#FF9900', fontWeight: 600 }}>{msg.fileName}</span> uploaded. Choose analysis:
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
                          {[
                            { key: 'aero', label: 'Aerodynamic Analysis', desc: 'Cd, Cs, Cl, Cmy' },
                            { key: 'surface', label: 'Surface Pressure Heatmap', desc: 'Cp & Cfx visualization' },
                            { key: 'slices', label: 'Flow Field Slices', desc: 'Velocity cross-sections' },
                            { key: 'structural', label: 'Structural Evaluation', desc: 'Weight, stiffness' },
                            { key: 'cost_aluminum', label: 'Cost — Aluminum', desc: 'Manufacturing breakdown' },
                            { key: 'cost_steel', label: 'Cost — Steel', desc: 'Manufacturing breakdown' },
                            { key: 'cost_carbon', label: 'Cost — Carbon Fiber', desc: 'Manufacturing breakdown' },
                            { key: 'full', label: 'Full Pipeline', desc: 'Aero + Structural + Cost' },
                          ].map(action => (
                            <button
                              key={action.key}
                              className="action-btn"
                              onClick={() => handleSTLAction(action.key, msg.s3Path, msg.fileName)}
                              disabled={isProcessing}
                              style={{
                                padding: '6px 10px', borderRadius: 5, border: '1px solid #30363d',
                                background: '#21262d', color: '#e6edf3', fontSize: 12, cursor: isProcessing ? 'not-allowed' : 'pointer',
                                textAlign: 'left', opacity: isProcessing ? 0.5 : 1, transition: 'all 0.15s', display: 'flex', gap: 8, alignItems: 'center',
                              }}
                            >
                              <span style={{ fontWeight: 600 }}>{action.label}</span>
                              <span style={{ color: '#6e7681', fontSize: 11 }}>{action.desc}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    ) : msg.type === 'agent' ? (
                      <>
                        {formatText(msg.text)}
                        {msg.streaming && (
                          <span style={{ display: 'inline-flex', gap: 3, marginLeft: 4, verticalAlign: 'middle' }}>
                            {[0, 0.2, 0.4].map((d, i) => (
                              <span key={i} style={{ width: 4, height: 4, borderRadius: '50%', background: '#FF9900', display: 'inline-block', animation: `pulse 1.2s ease-in-out infinite ${d}s` }} />
                            ))}
                          </span>
                        )}
                      </>
                    ) : msg.text}

                    {msg.images && msg.images.length > 0 && (
                      <div style={{ marginTop: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
                        {msg.images.map((url, i) => (
                          <img key={i} src={url} alt="Visualization"
                            style={{ maxWidth: '100%', borderRadius: 6, border: '1px solid #30363d' }}
                            onError={e => { e.target.style.display = 'none'; }} />
                        ))}
                      </div>
                    )}

                    {msg.stlUrl && !msg.streaming && (
                      <div style={{ marginTop: 10, borderTop: '1px solid #21262d', paddingTop: 8 }}>
                        <div style={{ color: '#8b949e', fontSize: 11, marginBottom: 6 }}>Analyse this geometry:</div>
                        <div style={{ display: 'flex', flexWrap: 'wrap', gap: 5 }}>
                          {[
                            { key: 'aero', label: 'Aero KPIs', desc: 'Cd, Cs, Cl, Cmy' },
                            { key: 'surface', label: 'Surface Heatmap', desc: 'Cp & Cfx' },
                            { key: 'slices', label: 'Flow Slices', desc: 'Velocity field' },
                            { key: 'structural', label: 'Structural', desc: 'Mesh metrics' },
                            { key: 'cost_aluminum', label: 'Cost (Al)', desc: 'Manufacturing' },
                          ].map(action => (
                            <button
                              key={action.key}
                              className="action-btn"
                              onClick={() => handleSTLAction(action.key, msg.stlUrl, 'generated.stl')}
                              disabled={isProcessing}
                              style={{
                                padding: '4px 8px', borderRadius: 4, border: '1px solid #30363d',
                                background: '#21262d', color: '#e6edf3', fontSize: 11, cursor: isProcessing ? 'not-allowed' : 'pointer',
                                opacity: isProcessing ? 0.5 : 1, transition: 'all 0.15s',
                              }}
                            >
                              <span style={{ fontWeight: 600 }}>{action.label}</span>
                              <span style={{ color: '#6e7681', marginLeft: 4 }}>{action.desc}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              ))}

              {/* Loading state */}
              {loading && !messages.some(m => m.streaming) && (
                <div className="msg-row" style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                    <div style={{ width: 18, height: 18, borderRadius: 3, background: 'linear-gradient(135deg, #FF9900, #FF6600)', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 9, fontWeight: 700, color: '#fff' }}>AI</div>
                    <span style={{ fontSize: 11, fontWeight: 600, color: '#FF9900' }}>AI Assistant</span>
                  </div>
                  <div style={{ marginLeft: 24 }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 7, marginBottom: 6 }}>
                      {[0, 0.2, 0.4].map((d, i) => (
                        <div key={i} style={{ width: 5, height: 5, borderRadius: '50%', background: '#FF9900', animation: `pulse 1.2s ease-in-out infinite ${d}s` }} />
                      ))}
                      <span style={{ fontSize: 11, color: '#6e7681', fontStyle: 'italic' }}>{loadingMessage}</span>
                    </div>
                    {loadingProgress > 0 && (
                      <div style={{ width: '100%', height: 2, background: '#21262d', borderRadius: 2, overflow: 'hidden' }}>
                        <div style={{ height: '100%', background: 'linear-gradient(90deg, #FF9900, #FF6600)', width: `${loadingProgress}%`, transition: 'width 0.5s ease', borderRadius: 2 }} />
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Input bar */}
            <div style={{ padding: '10px 12px', borderTop: '1px solid #30363d', flexShrink: 0 }}>
              <div style={{
                display: 'flex', gap: 8, alignItems: 'flex-end',
                background: '#21262d', border: '1px solid #30363d',
                borderRadius: 8, padding: '7px 10px', transition: 'border-color 0.15s',
              }}
                onFocus={e => e.currentTarget.style.borderColor = '#FF9900'}
                onBlur={e => e.currentTarget.style.borderColor = '#30363d'}
              >
                <input
                  type="text" value={input}
                  onChange={e => setInput(e.target.value)}
                  onKeyPress={e => e.key === 'Enter' && !e.shiftKey && !isProcessing && sendMessage()}
                  placeholder="Ask about designs, aero, cost..."
                  disabled={isProcessing}
                  style={{
                    flex: 1, background: 'transparent', border: 'none', outline: 'none',
                    fontSize: 13, color: '#e6edf3', fontFamily: 'inherit',
                    '::placeholder': { color: '#484f58' },
                  }}
                />
                <button
                  onClick={() => sendMessage()}
                  disabled={isProcessing || !input.trim()}
                  style={{
                    width: 28, height: 28, borderRadius: 5, border: 'none', flexShrink: 0,
                    background: isProcessing || !input.trim() ? '#30363d' : 'linear-gradient(135deg, #FF9900, #FF6600)',
                    color: 'white', cursor: isProcessing || !input.trim() ? 'not-allowed' : 'pointer',
                    display: 'flex', alignItems: 'center', justifyContent: 'center',
                    fontSize: 13, transition: 'all 0.15s',
                    boxShadow: isProcessing || !input.trim() ? 'none' : '0 2px 8px rgba(255,153,0,0.35)',
                  }}
                >
                  {isProcessing
                    ? <div style={{ width: 14, height: 14, border: '2px solid rgba(255,255,255,0.3)', borderTop: '2px solid #fff', borderRadius: '50%', animation: 'spin 1s linear infinite' }} />
                    : '▶'
                  }
                </button>
              </div>
            </div>

          </div>
        </div>
      </div>
    </>
  );
};

export default Chat;
