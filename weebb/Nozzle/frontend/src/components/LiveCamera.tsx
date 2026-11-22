import React, { useRef, useState, useEffect, useCallback } from 'react';
import { Camera, Play, VideoOff, Sparkles, AlertCircle, Pause } from 'lucide-react';

interface LiveCameraProps {
  onFrameCapture: (base64Image: string) => Promise<void>;
  isLoading: boolean;
}

export const LiveCamera: React.FC<LiveCameraProps> = ({ onFrameCapture, isLoading }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const [stream, setStream] = useState<MediaStream | null>(null);
  const [cameraActive, setCameraActive] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [continuousScan, setContinuousScan] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [fps, setFps] = useState(0);

  // Refs for the auto-loop so we never stale-close over state
  const loopActiveRef = useRef(false);
  const onFrameCaptureRef = useRef(onFrameCapture);
  useEffect(() => { onFrameCaptureRef.current = onFrameCapture; }, [onFrameCapture]);
  const fpsCountRef = useRef(0);
  const fpsTimerRef = useRef<any>(null);

  // Fetch available video input devices (external webcam, built-in webcam)
  const refreshDevices = async () => {
    try {
      const devList = await navigator.mediaDevices.enumerateDevices();
      const videoDevs = devList.filter((d) => d.kind === 'videoinput');
      setDevices(videoDevs);
      if (videoDevs.length > 0 && !selectedDeviceId) {
        // Prefer external webcam if labeled as such, else select the last device (often external)
        const external = videoDevs.find(
          (d) =>
            d.label.toLowerCase().includes('external') ||
            d.label.toLowerCase().includes('usb') ||
            d.label.toLowerCase().includes('c920') ||
            d.label.toLowerCase().includes('webcam')
        );
        setSelectedDeviceId(external ? external.deviceId : videoDevs[videoDevs.length - 1].deviceId);
      }
    } catch (e) {
      console.warn('Could not enumerate devices:', e);
    }
  };

  // Start Camera Stream — also auto-enables continuous scan
  const startCamera = async (devId?: string) => {
    setCameraError(null);
    try {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }

      const targetId = devId || selectedDeviceId;
      const constraints: MediaStreamConstraints = {
        video: targetId
          ? { deviceId: { exact: targetId }, width: { ideal: 1280 }, height: { ideal: 720 } }
          : { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      };

      const newStream = await navigator.mediaDevices.getUserMedia(constraints);

      if (videoRef.current) {
        videoRef.current.srcObject = newStream;
        await videoRef.current.play().catch(() => {});
      }
      setStream(newStream);
      setCameraActive(true);
      setContinuousScan(true); // ← auto-start live scan immediately

      // Re-enumerate to get labels now that permission is granted
      await refreshDevices();
    } catch (err: any) {
      console.error('Camera start error:', err);
      // Fallback: try basic video request without any constraints
      try {
        const fallbackStream = await navigator.mediaDevices.getUserMedia({ video: true });
        if (videoRef.current) {
          videoRef.current.srcObject = fallbackStream;
          await videoRef.current.play().catch(() => {});
        }
        setStream(fallbackStream);
        setCameraActive(true);
        setContinuousScan(true); // ← auto-start live scan on fallback too
        await refreshDevices();
      } catch (fallbackErr: any) {
        setCameraError(
          fallbackErr.message ||
            'Cannot access webcam. Check browser permissions and ensure no other app is using the camera.'
        );
        setCameraActive(false);
      }
    }
  };

  // Stop Camera Stream
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setCameraActive(false);
    setContinuousScan(false);
  };

  // Switch Device from Dropdown
  const handleDeviceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const newId = e.target.value;
    setSelectedDeviceId(newId);
    if (cameraActive) {
      startCamera(newId);
    }
  };

  useEffect(() => {
    refreshDevices();
    return () => {
      if (stream) stream.getTracks().forEach((t) => t.stop());
    };
  }, []);

  // Capture Frame as Base64 JPEG
  const grabFrame = useCallback((): string | null => {
    if (!videoRef.current || !canvasRef.current) return null;
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (video.videoWidth === 0 || video.videoHeight === 0) return null;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.85);
  }, []);

  // Single Frame Snapshot Action
  const handleSnapshot = async () => {
    const frame = grabFrame();
    if (frame) {
      await onFrameCapture(frame);
    }
  };

  // ── Self-chaining continuous scan loop ────────────────────────────────────
  // Uses recursion instead of setInterval so the next frame only fires AFTER
  // the backend responds — no queue build-up, no dropped frames.
  const runLoop = useCallback(async () => {
    if (!loopActiveRef.current) return;

    const frame = grabFrame();
    if (frame) {
      try {
        await onFrameCaptureRef.current(frame);
        // Count completed frames for FPS display
        fpsCountRef.current += 1;
      } catch (_) {
        // If the backend errors, wait a bit before retrying
        await new Promise((r) => setTimeout(r, 500));
      }
    } else {
      // Video not ready yet — wait a tick
      await new Promise((r) => setTimeout(r, 100));
    }

    // Schedule next frame immediately if still active
    if (loopActiveRef.current) {
      requestAnimationFrame(() => runLoop());
    }
  }, [grabFrame]);

  // Start/stop the auto-loop when continuousScan toggles
  useEffect(() => {
    if (continuousScan && cameraActive) {
      loopActiveRef.current = true;

      // FPS counter: count frames delivered per second
      fpsCountRef.current = 0;
      fpsTimerRef.current = setInterval(() => {
        setFps(fpsCountRef.current);
        fpsCountRef.current = 0;
      }, 1000);

      runLoop();
    } else {
      loopActiveRef.current = false;
      if (fpsTimerRef.current) {
        clearInterval(fpsTimerRef.current);
        fpsTimerRef.current = null;
      }
      setFps(0);
    }

    return () => {
      loopActiveRef.current = false;
      if (fpsTimerRef.current) {
        clearInterval(fpsTimerRef.current);
        fpsTimerRef.current = null;
      }
    };
  }, [continuousScan, cameraActive, runLoop]);

  return (
    <div className="space-y-4">
      {/* Device Selector (External Webcam Dropdown) */}
      {devices.length > 0 && (
        <div className="flex items-center gap-2">
          <label className="text-xs font-semibold text-slate-400 whitespace-nowrap">Camera:</label>
          <select
            value={selectedDeviceId}
            onChange={handleDeviceChange}
            className="w-full bg-slate-800/80 border border-slate-700 text-slate-200 text-xs rounded-lg px-2.5 py-1.5 focus:outline-none focus:border-sky-500"
          >
            {devices.map((d, i) => (
              <option key={d.deviceId || i} value={d.deviceId}>
                {d.label || `Camera ${i + 1}`}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Video Container */}
      <div className="relative aspect-video bg-black/60 rounded-xl overflow-hidden border border-slate-800 flex items-center justify-center">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className={`w-full h-full object-cover ${cameraActive ? 'block' : 'hidden'}`}
        />
        <canvas ref={canvasRef} className="hidden" />

        {/* Inactive State Display */}
        {!cameraActive && (
          <div className="text-center p-6 space-y-3">
            <div className="h-12 w-12 rounded-full bg-slate-800 flex items-center justify-center mx-auto text-slate-400">
              <Camera className="w-6 h-6" />
            </div>
            <div>
              <p className="text-sm font-semibold text-slate-200">Camera is Inactive</p>
              <p className="text-xs text-slate-500 mt-1">
                Attach your external webcam and click start to inspect live.
              </p>
            </div>
            {cameraError && (
              <div className="text-xs text-rose-400 bg-rose-500/10 py-1.5 px-3 rounded-lg border border-rose-500/20 flex items-center gap-2 max-w-sm mx-auto text-left">
                <AlertCircle className="w-4 h-4 shrink-0" />
                <span>{cameraError}</span>
              </div>
            )}
            <button
              onClick={() => startCamera()}
              className="px-4 py-2 bg-sky-500 hover:bg-sky-400 text-white rounded-xl text-xs font-semibold shadow-lg shadow-sky-500/20 transition flex items-center gap-1.5 mx-auto"
            >
              <Play className="w-3.5 h-3.5" /> Start Live Camera
            </button>
          </div>
        )}

        {/* Live Indicator Overlay */}
        {cameraActive && (
          <div className="absolute top-3 left-3 flex items-center gap-2 bg-black/60 backdrop-blur-md px-2.5 py-1 rounded-full border border-white/10 text-[11px] font-medium text-white">
            <span className={`h-2 w-2 rounded-full ${continuousScan ? 'bg-red-500 animate-pulse' : 'bg-amber-400'}`}></span>
            <span>{continuousScan ? `LIVE${fps > 0 ? ` · ${fps} FPS` : ''}` : 'PAUSED'}</span>
          </div>
        )}
      </div>

      {/* Camera Controls Bar */}
      {cameraActive && (
        <div className="grid grid-cols-3 gap-2">
          {/* Snapshot Button — only when paused */}
          <button
            onClick={handleSnapshot}
            disabled={isLoading || continuousScan}
            className="py-2.5 px-3 bg-sky-500 hover:bg-sky-400 disabled:opacity-40 text-white font-semibold text-xs rounded-xl shadow-md shadow-sky-500/20 transition flex items-center justify-center gap-1.5"
          >
            <Camera className="w-4 h-4" /> Snap & Detect
          </button>

          {/* Pause / Resume Live Scan */}
          <button
            onClick={() => setContinuousScan(!continuousScan)}
            className={`py-2.5 px-3 font-semibold text-xs rounded-xl border transition flex items-center justify-center gap-1.5 ${
              continuousScan
                ? 'bg-amber-500/20 text-amber-300 border-amber-500/40 hover:bg-amber-500/30'
                : 'bg-purple-600 text-white border-purple-500 shadow-md shadow-purple-500/25 hover:bg-purple-500'
            }`}
          >
            {continuousScan
              ? <><Pause className="w-4 h-4" /> Pause</>
              : <><Sparkles className="w-4 h-4" /> Resume Live</>}
          </button>

          {/* Stop Camera */}
          <button
            onClick={stopCamera}
            className="py-2.5 px-3 bg-slate-800 hover:bg-rose-500/20 hover:text-rose-400 text-slate-400 font-semibold text-xs rounded-xl border border-slate-700 transition flex items-center justify-center gap-1.5"
          >
            <VideoOff className="w-4 h-4" /> Stop
          </button>
        </div>
      )}
    </div>
  );
};
