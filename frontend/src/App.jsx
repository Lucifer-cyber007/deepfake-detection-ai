import { useState, useRef, useEffect } from 'react'
import './index.css'

function App() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [analysisData, setAnalysisData] = useState(null)
  const [logs, setLogs] = useState([])
  const fileInputRef = useRef(null)
  const logEndRef = useRef(null)

  const addLog = (msg) => {
    const time = new Date().toLocaleTimeString([], { hour12: false })
    setLogs(prev => [...prev, `[${time}] ${msg}`])
  }

  useEffect(() => {
    if (logEndRef.current) {
      logEndRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [logs])

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0]
    if (selectedFile) {
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setAnalysisData(null)
      setLogs([])
      addLog(`File loaded: ${selectedFile.name} (${(selectedFile.size / 1024 / 1024).toFixed(2)} MB)`)
    }
  }

  const runAnalysis = async () => {
    if (!file) return
    setLoading(true)
    setAnalysisData(null)
    setLogs([])

    addLog("Initializing NeuralGuard Analysis Engine...")
    addLog("Loading EfficientNet-B3 Weights...")
    addLog("Probing video streams...")
    
    const formData = new FormData()
    formData.append('file', file)

    try {
      addLog("Running High-Resolution Temporal Inference...")
      const response = await fetch('http://localhost:8000/predict/video', {
        method: 'POST',
        body: formData,
      })
      const data = await response.json()
      
      if (data.success) {
        addLog("Analysis complete. Compiling thermal signature...")
        setAnalysisData(data.prediction)
        addLog(`Found ${data.prediction.suspicious_frames.length} suspicious anomalies.`)
      } else {
        addLog("Error: Backend sequence failed.")
      }
    } catch (error) {
      addLog("Critical: Backend connection lost.")
    } finally {
      setLoading(false)
    }
  }

  const getHeatmapColor = (val) => {
    if (val > 0.7) return '#ef4444' // Danger
    if (val > 0.4) return '#f59e0b' // Warning
    return '#10b981' // Success
  }

  return (
    <div className="dashboard">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo-section">
          <h2 style={{ fontSize: '1.2rem', color: 'var(--accent)' }}>NeuralGuard</h2>
          <p style={{ fontSize: '0.7rem', opacity: 0.6 }}>SYSTEM V1.4.2</p>
        </div>

        <section>
          <p className="section-title">Input</p>
          <div className="dropzone" onClick={() => fileInputRef.current.click()}>
            <p style={{ fontSize: '0.8rem' }}>Drag & drop or Click to upload</p>
            <input 
              type="file" 
              ref={fileInputRef} 
              style={{ display: 'none' }} 
              accept="video/*"
              onChange={handleFileChange}
            />
          </div>
          <button className="btn btn-secondary" style={{ marginTop: '0.5rem', fontSize: '0.8rem' }}>
            Demo Mode
          </button>
        </section>

        <section>
          <p className="section-title">Actions</p>
          <button className="btn btn-primary" onClick={runAnalysis} disabled={loading || !file}>
            {loading ? 'ANALYZING...' : 'ANALYZE'}
          </button>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', marginTop: '0.5rem' }}>
            <button className="btn btn-secondary" style={{ fontSize: '0.7rem' }}>Export CSV</button>
            <button className="btn btn-secondary" style={{ fontSize: '0.7rem' }}>Export ZIP</button>
          </div>
        </section>

        <footer style={{ marginTop: 'auto', fontSize: '0.7rem', opacity: 0.5 }}>
          <p>● Local Backend Connected</p>
          <p>MP4, MOV supported</p>
        </footer>
      </aside>

      {/* Main Workspace */}
      <main className="workspace">
        <div className="card">
          <div className="video-container">
            {preview ? (
              <video src={preview} controls className="video-preview" />
            ) : (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: '#444' }}>
                No video preview
              </div>
            )}
          </div>

          {analysisData && analysisData.heatmap && (
            <div className="analysis-results">
              <p className="section-title" style={{ marginTop: '1.5rem' }}>Timeline Heatmap</p>
              <div className="heatmap">
                {analysisData.heatmap.map((point, i) => (
                  <div 
                    key={i} 
                    className="heatmap-segment" 
                    style={{ backgroundColor: getHeatmapColor(point.val) }}
                    title={`TS: ${point.ts.toFixed(1)}s | Score: ${point.val.toFixed(2)}`}
                  ></div>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card">
          <p className="section-title">Suspicious Frames Detected</p>
          {analysisData && analysisData.suspicious_frames ? (
            <div className="frames-grid">
              {analysisData.suspicious_frames.map((frame, i) => (
                <div key={i} className="frame-card">
                  <img src={`data:image/jpeg;base64,${frame.thumbnail}`} className="frame-thumb" />
                  <div className="frame-info">
                    <span style={{ color: 'var(--text-secondary)' }}>#{i+1} • {frame.timestamp.toFixed(1)}s</span>
                    <span style={{ color: getHeatmapColor(frame.score), fontWeight: 'bold' }}>{frame.score.toFixed(2)}</span>
                  </div>
                </div>
              ))}
              {analysisData.suspicious_frames.length === 0 && (
                <p style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>No major anomalies detected.</p>
              )}
            </div>
          ) : (
            <p style={{ fontSize: '0.9rem', opacity: 0.5 }}>Analysis results will appear here.</p>
          )}
        </div>
      </main>

      {/* Overview Panel */}
      <aside className="overview">
        <section className="gauge-container card">
          <p className="section-title">Overall Score</p>
          <div className="score-display" style={{ color: analysisData ? getHeatmapColor(analysisData.overall_score) : 'var(--text-secondary)' }}>
            {analysisData ? analysisData.overall_score.toFixed(2) : '--'}
          </div>
          <p style={{ fontSize: '0.8rem', fontWeight: 600 }}>
            {analysisData ? (analysisData.overall_score > 0.5 ? 'HIGH RISK' : 'LOW RISK') : 'Awaiting Data'}
          </p>
        </section>

        <section className="card" style={{ flexGrow: 1, display: 'flex', flexDirection: 'column' }}>
          <p className="section-title">Processing Logs</p>
          <div className="log-container">
            {logs.map((log, i) => (
              <div key={i} className="log-entry">{log}</div>
            ))}
            <div ref={logEndRef} />
          </div>
        </section>

        <section className="card">
          <p className="section-title">Filters</p>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            <span style={{ fontSize: '0.7rem' }}>Min Probability</span>
            <input type="range" style={{ flexGrow: 1 }} />
            <span style={{ fontSize: '0.7rem' }}>0.70</span>
          </div>
        </section>
      </aside>
    </div>
  )
}

export default App
