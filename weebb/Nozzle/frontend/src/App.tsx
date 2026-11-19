import React, { useState, useEffect, useCallback } from 'react';
import { LiveCamera } from './components/LiveCamera';
import {
  Scan,
  UploadCloud,
  Camera,
  Sparkles,
  Eye,
  Monitor,
  Boxes,
  Layers,
  PieChart,
  Zap,
  CheckCircle2,
  Loader2,
  ClipboardList,
} from 'lucide-react';

// ─── Types ───────────────────────────────────────────────────────────────────
interface BreakdownRow {
  type: string;
  class: string;
  count: number;
  percentage: number;
}

interface ApiResult {
  success: boolean;
  latency_ms: number;
  total_count: number;
  undetected: number;
  unique_types: number;
  occupancy_rate: number;
  total_slots: number;
  model_version: string;
  breakdown: BreakdownRow[];
  annotated_image: string;   // base64 data:image/jpeg;base64,...
}

interface DemoResult {
  imageUrl: string;
  breakdown: BreakdownRow[];
  latency_ms: number;
  total_count: number;
  occupancy_rate: number;
  model_version: string;
}

const API_BASE = (import.meta as any).env?.VITE_API_URL ?? 'http://localhost:8000';

// ─── App ─────────────────────────────────────────────────────────────────────
export const App: React.FC = () => {
  const [inputMode, setInputMode] = useState<'camera' | 'upload'>('camera');
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [uploadPreview, setUploadPreview] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [demoResult, setDemoResult] = useState<DemoResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [apiError, setApiError] = useState<string | null>(null);

  const [apiUrl, setApiUrl] = useState<string>(() => {
    return localStorage.getItem('nozzle_api_url') || (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';
  });
  const [showUrlModal, setShowUrlModal] = useState(false);
  const [tempUrl, setTempUrl] = useState(apiUrl);
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);

  // Check backend health
  const checkHealth = useCallback(async (urlToCheck: string) => {
    try {
      const res = await fetch(`${urlToCheck}/health`, {
        headers: { 'ngrok-skip-browser-warning': '1' },
      });
      if (res.ok) {
        setBackendOnline(true);
      } else {
        setBackendOnline(false);
      }
    } catch {
      setBackendOnline(false);
    }
  }, []);

  useEffect(() => {
    checkHealth(apiUrl);
  }, [apiUrl, checkHealth]);

  const saveApiUrl = (newUrl: string) => {
    const clean = newUrl.trim().replace(/\/+$/, '');
    setApiUrl(clean);
    localStorage.setItem('nozzle_api_url', clean);
    setShowUrlModal(false);
    checkHealth(clean);
  };

  // ── Call real backend API with a File object ───────────────────────────────
  const analyzeFile = async (file: File) => {
    setIsAnalyzing(true);
    setDemoResult(null);
    setApiError(null);
    try {
      const form = new FormData();
      form.append('file', file);

      const res = await fetch(`${apiUrl}/api/detect`, {
        method: 'POST',
        headers: {
          'ngrok-skip-browser-warning': '1',
        },
        body: form,
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}: ${res.statusText}`);
      const data: ApiResult = await res.json();

      setDemoResult({
        imageUrl:       data.annotated_image,
        breakdown:      data.breakdown,
        latency_ms:     data.latency_ms,
        total_count:    data.total_count,
        occupancy_rate: data.occupancy_rate,
        model_version:  data.model_version,
      });
      setBackendOnline(true);
    } catch (err: any) {
      setApiError(err.message ?? 'Connection failed — check your backend / ngrok URL');
      setBackendOnline(false);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ── Call real backend API with a base64 camera frame ──────────────────────
  const analyzeBase64 = async (base64: string) => {
    setIsAnalyzing(true);
    setDemoResult(null);
    setApiError(null);
    try {
      const res = await fetch(`${apiUrl}/api/detect-frame`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': '1',
        },
        body: JSON.stringify({ image_base64: base64 }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}: ${res.statusText}`);
      const data: ApiResult = await res.json();

      setDemoResult({
        imageUrl:       data.annotated_image,
        breakdown:      data.breakdown,
        latency_ms:     data.latency_ms,
        total_count:    data.total_count,
        occupancy_rate: data.occupancy_rate,
        model_version:  data.model_version,
      });
      setBackendOnline(true);
    } catch (err: any) {
      setApiError(err.message ?? 'Connection failed — check your backend / ngrok URL');
      setBackendOnline(false);
    } finally {
      setIsAnalyzing(false);
    }
  };

  // ── Camera snapshot handler ───────────────────────────────────────────────
  const handleFrameCapture = async (base64: string) => {
    setCapturedImage(base64);
    await analyzeBase64(base64);
  };

  // ── File upload handler ───────────────────────────────────────────────────
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setUploadPreview(url);
    setDemoResult(null);
    setApiError(null);
  };

  const handleAnalyzeUpload = async () => {
    if (!selectedFile) return;
    await analyzeFile(selectedFile);
  };

  // ── Load Sample ───────────────────────────────────────────────────────────
  const loadSample = async () => {
    try {
      const res = await fetch('/sample.jpeg');
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setInputMode('upload');
      setUploadPreview(url);
      setSelectedFile(new File([blob], 'sample.jpeg'));
      setDemoResult(null);
    } catch {
      alert('Sample image not found. Drop any tray photo in instead!');
    }
  };

  const totalNozzles = demoResult?.breakdown.reduce((s, r) => s + r.count, 0) ?? 0;
  const displayImage = inputMode === 'camera' ? capturedImage : uploadPreview;

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col antialiased">

      {/* ── Navbar ── */}
      <header className="border-b border-slate-800 bg-[#0b0f19]/80 sticky top-0 z-40 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="h-10 w-10 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-sky-500/20">
              <Scan className="w-5 h-5 text-white" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-lg tracking-tight bg-gradient-to-r from-white via-slate-200 to-sky-400 bg-clip-text text-transparent">
                  NozzleVision AI
                </span>
                <span className="text-[10px] font-bold tracking-wider px-2 py-0.5 rounded-full bg-sky-500/10 text-sky-400 border border-sky-500/20 uppercase">
                  Production
                </span>
              </div>
              <p className="text-xs text-slate-400">Industrial Tray Inspection System</p>
            </div>
          </div>
          <button
            onClick={() => {
              setTempUrl(apiUrl);
              setShowUrlModal(true);
            }}
            title="Click to configure backend URL (ngrok / localhost)"
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition cursor-pointer ${
              backendOnline === true
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 hover:bg-emerald-500/20'
                : backendOnline === false
                ? 'bg-rose-500/10 text-rose-400 border-rose-500/30 hover:bg-rose-500/20'
                : 'bg-amber-500/10 text-amber-400 border-amber-500/30 hover:bg-amber-500/20'
            }`}
          >
            <span
              className={`h-2 w-2 rounded-full ${
                backendOnline === true
                  ? 'bg-emerald-400 animate-pulse'
                  : backendOnline === false
                  ? 'bg-rose-400'
                  : 'bg-amber-400'
              }`}
            ></span>
            <span>
              {backendOnline === true
                ? 'Backend: Online'
                : backendOnline === false
                ? 'Backend: Offline (Click to set ngrok)'
                : 'Connecting...'}
            </span>
          </button>
        </div>
      </header>

      {/* ── Modal to change Backend / ngrok URL ── */}
      {showUrlModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="bg-[#111827] border border-slate-700 rounded-2xl max-w-md w-full p-6 shadow-2xl space-y-4">
            <h3 className="text-base font-bold text-white flex items-center gap-2">
              <Zap className="w-5 h-5 text-sky-400" /> Configure Backend API URL
            </h3>
            <p className="text-xs text-slate-400">
              Paste your public ngrok URL when running on Vercel, or leave as localhost for local testing.
            </p>
            <input
              type="text"
              value={tempUrl}
              onChange={(e) => setTempUrl(e.target.value)}
              placeholder="https://xxxx.ngrok-free.app or http://localhost:8000"
              className="w-full px-3 py-2 bg-slate-900 border border-slate-700 rounded-xl text-xs text-white focus:outline-none focus:border-sky-500 font-mono"
            />
            <div className="flex justify-end gap-2 pt-2">
              <button
                onClick={() => setShowUrlModal(false)}
                className="px-4 py-2 bg-slate-800 hover:bg-slate-700 text-slate-300 text-xs font-semibold rounded-xl"
              >
                Cancel
              </button>
              <button
                onClick={() => saveApiUrl(tempUrl)}
                className="px-4 py-2 bg-sky-500 hover:bg-sky-400 text-white text-xs font-semibold rounded-xl shadow-lg shadow-sky-500/20"
              >
                Save & Connect
              </button>
            </div>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 flex-1 w-full space-y-6">

        {/* ── KPI Stat Cards ── */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { icon: Boxes,    label: 'Total Nozzles',  value: demoResult ? String(totalNozzles) : '--', color: 'sky' },
            { icon: Layers,   label: 'Unique Types',   value: demoResult ? String(demoResult.breakdown.length) : '--', color: 'purple' },
            { icon: PieChart, label: 'Tray Capacity',  value: demoResult ? `${Math.round((totalNozzles / 77) * 100)}%` : '--%', color: 'emerald' },
            { icon: Zap,      label: 'Inference',      value: demoResult ? `${demoResult.latency_ms} ms` : '-- ms', color: 'amber', sub: demoResult ? `Model ${demoResult.model_version}` : 'Connect model' },
          ].map(({ icon: Icon, label, value, color, sub }) => (
            <div key={label} className="glass-card p-5 rounded-2xl border border-slate-800 flex items-center justify-between">
              <div>
                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">{label}</p>
                <h3 className={`text-3xl font-black mt-1 text-${color}-400`}>{value}</h3>
                <p className="text-[11px] text-slate-500 mt-0.5">{sub ?? 'Live Inspection'}</p>
              </div>
              <div className={`p-3 bg-${color}-500/10 rounded-xl text-${color}-400`}>
                <Icon className="w-6 h-6" />
              </div>
            </div>
          ))}
        </div>

        {/* ── Main Grid ── */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">

          {/* Left: Input Controls */}
          <div className="lg:col-span-5 space-y-5">
            <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-5">

              {/* Mode Switcher */}
              <div className="flex items-center justify-between">
                <div className="flex bg-slate-800/80 p-1 rounded-xl border border-slate-700/60">
                  {(['camera', 'upload'] as const).map((mode) => (
                    <button
                      key={mode}
                      onClick={() => setInputMode(mode)}
                      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                        inputMode === mode
                          ? 'bg-sky-500 text-white shadow-md'
                          : 'text-slate-400 hover:text-white'
                      }`}
                    >
                      {mode === 'camera' ? <Camera className="w-3.5 h-3.5" /> : <UploadCloud className="w-3.5 h-3.5" />}
                      {mode === 'camera' ? 'Live Camera' : 'File Upload'}
                    </button>
                  ))}
                </div>
                <button
                  onClick={loadSample}
                  className="text-xs text-sky-400 hover:text-sky-300 font-medium flex items-center gap-1 transition"
                >
                  <Sparkles className="w-3.5 h-3.5" /> Sample Tray
                </button>
              </div>

              {/* Camera Mode */}
              {inputMode === 'camera' && (
                <LiveCamera onFrameCapture={handleFrameCapture} isLoading={isAnalyzing} />
              )}

              {/* Upload Mode */}
              {inputMode === 'upload' && (
                <div className="space-y-4">
                  <label className="border-2 border-dashed border-slate-700 hover:border-sky-500/60 rounded-xl p-6 text-center cursor-pointer transition bg-slate-800/20 hover:bg-slate-800/40 block">
                    <input type="file" accept="image/*" className="hidden" onChange={handleFileChange} />
                    {uploadPreview ? (
                      <div className="space-y-2">
                        <img src={uploadPreview} alt="Preview" className="max-h-52 mx-auto rounded-lg object-contain shadow" />
                        <p className="text-xs text-slate-400">{selectedFile?.name ?? 'Uploaded image'}</p>
                        <p className="text-xs text-sky-400 mt-1">Click to change image</p>
                      </div>
                    ) : (
                      <div className="space-y-3 py-4">
                        <div className="h-12 w-12 rounded-full bg-slate-800 flex items-center justify-center mx-auto text-slate-500">
                          <UploadCloud className="w-6 h-6" />
                        </div>
                        <div>
                          <p className="text-sm font-medium text-slate-200">Click to upload or drag & drop</p>
                          <p className="text-xs text-slate-500 mt-0.5">PNG, JPG, JPEG up to 20 MB</p>
                        </div>
                      </div>
                    )}
                  </label>

                  <button
                    onClick={handleAnalyzeUpload}
                    disabled={!selectedFile || isAnalyzing}
                    className="w-full py-3.5 bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white font-bold text-sm rounded-xl shadow-lg shadow-sky-500/20 transition disabled:opacity-40 flex items-center justify-center gap-2"
                  >
                    {isAnalyzing
                      ? <><Loader2 className="w-4 h-4 animate-spin" /> Analyzing Tray...</>
                      : <><Eye className="w-4 h-4" /> Analyze Tray Photo</>}
                  </button>
                </div>
              )}

              {/* API Error Banner */}
              {apiError && (
                <div className="p-3 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-300">
                  <strong className="text-red-200">⚠ Backend Error:</strong> {apiError}
                  <br />Make sure the backend is running: <code className="text-red-400">python -m uvicorn backend.server:app --reload</code>
                </div>
              )}
              {/* Status Banner */}
              <div className="p-3 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-xs text-indigo-300 leading-relaxed">
                <strong className="text-indigo-200">Live Mode:</strong> Connected to your detection backend at <code>localhost:8000</code>. Real YOLO results returned.
              </div>
            </div>
          </div>

          {/* Right: Preview + Result Canvas */}
          <div className="lg:col-span-7">
            <div className="glass-card p-6 rounded-2xl border border-slate-800 flex flex-col min-h-[460px]">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-base font-bold text-white flex items-center gap-2">
                  <Eye className="w-4 h-4 text-emerald-400" /> Detection Canvas
                </h2>
                {demoResult && (
                  <span className="px-2.5 py-1 rounded-md text-xs font-semibold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                    Analysis Complete
                  </span>
                )}
              </div>

              <div className="flex-1 flex items-center justify-center bg-black/40 rounded-xl border border-slate-800 overflow-hidden relative min-h-[340px]">
                {isAnalyzing && (
                  <div className="absolute inset-0 z-10 flex flex-col items-center justify-center bg-black/60 backdrop-blur-sm gap-3">
                    <Loader2 className="w-10 h-10 animate-spin text-sky-400" />
                    <p className="text-sm font-semibold text-slate-200">Running Neural Inspection...</p>
                    <p className="text-xs text-slate-400">Analyzing tray layout & nozzle geometry</p>
                  </div>
                )}

                {/* Canvas Display */}
                {(demoResult?.imageUrl || displayImage) && !isAnalyzing ? (
                  <div className="relative w-full h-full flex items-center justify-center p-2">
                    <img
                      src={demoResult?.imageUrl || displayImage || ''}
                      alt="Annotated Tray Detection"
                      className="max-h-[400px] max-w-full object-contain rounded-lg shadow-2xl"
                    />
                    {demoResult && (
                      <div className="absolute top-3 left-3 px-2.5 py-1 rounded-full bg-emerald-500/90 text-white text-[10px] font-bold backdrop-blur-sm shadow-md">
                        {totalNozzles} nozzles identified ({demoResult.model_version})
                      </div>
                    )}
                  </div>
                ) : !isAnalyzing ? (
                  <div className="text-center p-8 space-y-2">
                    <Monitor className="w-12 h-12 text-slate-700 mx-auto" />
                    <p className="text-sm font-medium text-slate-400">No Image Yet</p>
                    <p className="text-xs text-slate-600 max-w-xs">
                      Snap a photo with the live camera or upload a tray image to see the annotated output here.
                    </p>
                  </div>
                ) : null}
              </div>
            </div>
          </div>
        </div>

        {/* ── Inventory BOM Table ── */}
        <div className="glass-card p-6 rounded-2xl border border-slate-800 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <h2 className="text-base font-bold text-white flex items-center gap-2">
                <ClipboardList className="w-4 h-4 text-sky-400" /> Bill of Materials Inventory
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">Auto-generated breakdown by nozzle class</p>
            </div>
            {demoResult && (
              <span className="text-[11px] px-3 py-1 rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 font-medium">
                ⚠ Placeholder data — connect model for real counts
              </span>
            )}
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-slate-800 text-slate-400 uppercase tracking-wider font-semibold">
                  <th className="py-3 px-4">Nozzle Type</th>
                  <th className="py-3 px-4">Qty</th>
                  <th className="py-3 px-4">Tray Share</th>
                  <th className="py-3 px-4">QC Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 text-slate-300">
                {demoResult ? (
                  demoResult.breakdown.map((row, i) => (
                    <tr key={i} className="hover:bg-slate-800/20 transition">
                      <td className="py-3 px-4 font-medium text-white">{row.type}</td>
                      <td className="py-3 px-4 font-mono font-bold text-sky-400">{row.count}</td>
                      <td className="py-3 px-4 font-mono">{row.percentage}%</td>
                      <td className="py-3 px-4">
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                          <CheckCircle2 className="w-3 h-3" /> Passed QC
                        </span>
                      </td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={4} className="py-8 text-center text-slate-500 italic">
                      Capture or upload a tray image to populate the inventory.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>

      </main>
    </div>
  );
};
