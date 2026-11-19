import React, { useState } from 'react';
import { Scan, Settings, X, CheckCircle2, AlertCircle } from 'lucide-react';

interface NavbarProps {
  apiUrl: string;
  onUpdateApiUrl: (url: string) => void;
  isBackendConnected: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({ apiUrl, onUpdateApiUrl, isBackendConnected }) => {
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [tempUrl, setTempUrl] = useState(apiUrl);

  const handleSave = () => {
    let clean = tempUrl.trim();
    if (clean.endsWith('/')) clean = clean.slice(0, -1);
    onUpdateApiUrl(clean);
    setIsModalOpen(false);
  };

  return (
    <>
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
                  Vercel + ngrok
                </span>
              </div>
              <p className="text-xs text-slate-400">Live Camera Industrial Tray Inspector</p>
            </div>
          </div>

          {/* ngrok Status Pill */}
          <button
            onClick={() => { setTempUrl(apiUrl); setIsModalOpen(true); }}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border transition ${
              isBackendConnected
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/20'
                : 'bg-amber-500/10 text-amber-400 border-amber-500/20 hover:bg-amber-500/20'
            }`}
          >
            <span className={`h-2 w-2 rounded-full ${isBackendConnected ? 'bg-emerald-400 animate-pulse' : 'bg-amber-400'}`}></span>
            <span>{isBackendConnected ? 'ngrok Backend: Active' : 'Set ngrok URL'}</span>
            <Settings className="w-3.5 h-3.5 opacity-70 ml-1" />
          </button>
        </div>
      </header>

      {/* ngrok URL Config Modal */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-card max-w-md w-full p-6 rounded-2xl border border-slate-700 shadow-2xl space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-base font-bold text-white flex items-center gap-2">
                <Settings className="w-4 h-4 text-sky-400" /> Connect ngrok Backend
              </h3>
              <button onClick={() => setIsModalOpen(false)} className="text-slate-400 hover:text-white">
                <X className="w-4 h-4" />
              </button>
            </div>
            <p className="text-xs text-slate-400 leading-relaxed">
              Enter your public <strong>ngrok tunnel URL</strong> running on your GPU laptop (e.g. <code className="text-sky-300">https://xxxx.ngrok-free.app</code>).
            </p>
            <div>
              <label className="text-[11px] font-semibold text-slate-300 uppercase tracking-wider">FastAPI Endpoint URL</label>
              <input
                type="text"
                value={tempUrl}
                onChange={(e) => setTempUrl(e.target.value)}
                placeholder="https://your-ngrok-url.ngrok-free.app"
                className="w-full mt-1.5 px-3 py-2 bg-slate-900 border border-slate-700 rounded-lg text-xs font-mono text-white focus:outline-none focus:border-sky-500"
              />
            </div>
            <div className="flex items-center justify-between pt-2">
              <span className="text-[11px] text-slate-400 flex items-center gap-1">
                {isBackendConnected ? (
                  <><CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" /> Tunnel reachable</>
                ) : (
                  <><AlertCircle className="w-3.5 h-3.5 text-amber-400" /> Offline / Connecting</>
                )}
              </span>
              <div className="flex gap-2">
                <button
                  onClick={() => setIsModalOpen(false)}
                  className="px-3 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:bg-slate-800"
                >
                  Cancel
                </button>
                <button
                  onClick={handleSave}
                  className="px-4 py-1.5 rounded-lg text-xs font-semibold bg-sky-500 hover:bg-sky-400 text-white shadow-md shadow-sky-500/20"
                >
                  Save & Apply
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};
