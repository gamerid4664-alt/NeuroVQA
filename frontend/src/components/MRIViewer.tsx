import { useRef, useState, useCallback, useEffect } from "react";
import { ZoomIn, ZoomOut, RotateCcw, Sun, Contrast, Crosshair, Move } from "lucide-react";

interface MRIViewerProps {
  src: string | null;
  label?: string;
  showTools?: boolean;
  onRegionSelect?: (region: { x: number; y: number; w: number; h: number }) => void;
}

export default function MRIViewer({ src, label = "MRI Scan", showTools = true, onRegionSelect }: MRIViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);
  const [showCrosshair, setShowCrosshair] = useState(false);
  const [crosshairPos, setCrosshairPos] = useState({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [tool, setTool] = useState<"pan" | "crosshair">("pan");
  const imgRef = useRef<HTMLImageElement | null>(null);

  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
    ctx.save();
    ctx.translate(canvas.width / 2 + pan.x, canvas.height / 2 + pan.y);
    ctx.scale(zoom, zoom);
    ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2, img.naturalWidth, img.naturalHeight);
    ctx.restore();
    ctx.filter = "none";

    if (showCrosshair) {
      ctx.strokeStyle = "rgba(79, 70, 229, 0.7)";
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(crosshairPos.x, 0);
      ctx.lineTo(crosshairPos.x, canvas.height);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, crosshairPos.y);
      ctx.lineTo(canvas.width, crosshairPos.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }, [zoom, pan, brightness, contrast, showCrosshair, crosshairPos]);

  useEffect(() => {
    if (!src) return;
    const img = new Image();
    img.src = src;
    img.onload = () => {
      imgRef.current = img;
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = canvas.offsetWidth || 320;
        canvas.height = canvas.offsetHeight || 320;
      }
      drawCanvas();
    };
  }, [src]);

  useEffect(() => { drawCanvas(); }, [drawCanvas]);

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (tool === "crosshair") setCrosshairPos({ x, y });
    if (isPanning) {
      setPan({ x: pan.x + (e.clientX - panStart.x), y: pan.y + (e.clientY - panStart.y) });
      setPanStart({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseDown = (e: React.MouseEvent) => {
    if (tool === "pan") { setIsPanning(true); setPanStart({ x: e.clientX, y: e.clientY }); }
    if (tool === "crosshair" && onRegionSelect) {
      onRegionSelect({ x: crosshairPos.x, y: crosshairPos.y, w: 20, h: 20 });
    }
  };
  const handleMouseUp = () => setIsPanning(false);

  const reset = () => { setZoom(1); setPan({ x: 0, y: 0 }); setBrightness(100); setContrast(100); };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-2">
        <span className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider">{label}</span>
        {showTools && (
          <div className="flex items-center gap-1">
            <button onClick={() => setZoom(z => Math.min(z + 0.25, 4))} className="p-1.5 rounded hover:bg-[#1E2230] text-[#94A3B8] hover:text-white transition-colors" title="Zoom In">
              <ZoomIn size={12} />
            </button>
            <button onClick={() => setZoom(z => Math.max(z - 0.25, 0.5))} className="p-1.5 rounded hover:bg-[#1E2230] text-[#94A3B8] hover:text-white transition-colors" title="Zoom Out">
              <ZoomOut size={12} />
            </button>
            <button onClick={() => setTool(t => t === "crosshair" ? "pan" : "crosshair")}
              className={`p-1.5 rounded transition-colors ${tool === "crosshair" ? "bg-[#342CB3] text-white" : "hover:bg-[#1E2230] text-[#94A3B8] hover:text-white"}`} title="Crosshair">
              <Crosshair size={12} />
            </button>
            <button onClick={() => setShowCrosshair(s => !s)}
              className={`p-1.5 rounded transition-colors ${showCrosshair ? "bg-[#342CB3] text-white" : "hover:bg-[#1E2230] text-[#94A3B8] hover:text-white"}`} title="Toggle Crosshair">
              <Move size={12} />
            </button>
            <button onClick={reset} className="p-1.5 rounded hover:bg-[#1E2230] text-[#94A3B8] hover:text-white transition-colors" title="Reset">
              <RotateCcw size={12} />
            </button>
          </div>
        )}
      </div>

      <div ref={containerRef} className="relative flex-1 bg-[#0E1117] rounded-lg overflow-hidden border border-[#2A2F42] min-h-[240px]"
        onMouseMove={handleMouseMove} onMouseDown={handleMouseDown} onMouseUp={handleMouseUp} onMouseLeave={handleMouseUp}>
        {src ? (
          <canvas ref={canvasRef} className="w-full h-full cursor-crosshair" style={{ cursor: tool === "pan" ? "grab" : "crosshair" }} />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <div className="w-16 h-16 rounded-full bg-[#1E2230] border-2 border-dashed border-[#2A2F42] flex items-center justify-center mb-3">
              <span className="text-2xl">🧠</span>
            </div>
            <p className="text-[#64748B] text-xs text-center">Upload an MRI scan<br />to begin analysis</p>
          </div>
        )}
      </div>

      {showTools && (
        <div className="mt-2 grid grid-cols-2 gap-2">
          <div>
            <div className="flex items-center gap-1 mb-1">
              <Sun size={10} className="text-[#94A3B8]" />
              <span className="text-[10px] text-[#94A3B8]">Brightness {brightness}%</span>
            </div>
            <input type="range" min={50} max={200} value={brightness} onChange={e => setBrightness(Number(e.target.value))}
              className="w-full h-1 rounded accent-[#4F46E5]" />
          </div>
          <div>
            <div className="flex items-center gap-1 mb-1">
              <Contrast size={10} className="text-[#94A3B8]" />
              <span className="text-[10px] text-[#94A3B8]">Contrast {contrast}%</span>
            </div>
            <input type="range" min={50} max={200} value={contrast} onChange={e => setContrast(Number(e.target.value))}
              className="w-full h-1 rounded accent-[#4F46E5]" />
          </div>
        </div>
      )}

      {showTools && (
        <div className="mt-1 flex items-center justify-between text-[10px] text-[#64748B]">
          <span>Zoom: {zoom.toFixed(1)}x</span>
          {showCrosshair && <span>({Math.round(crosshairPos.x)}, {Math.round(crosshairPos.y)})</span>}
        </div>
      )}
    </div>
  );
}
