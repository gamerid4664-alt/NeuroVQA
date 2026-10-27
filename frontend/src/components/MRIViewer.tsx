import { useRef, useState, useCallback, useEffect } from "react";
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Sun,
  Contrast,
  Crosshair,
  Move,
  Square,
  Ruler,
  MessageSquare,
  Trash2,
  Check,
  X,
  Sparkles,
} from "lucide-react";

export interface ROIAnnotation {
  id: string;
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
  note: string;
  color: string;
}

export interface CaliperMeasurement {
  id: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  distanceMm: number;
}

interface MRIViewerProps {
  src: string | null;
  label?: string;
  showTools?: boolean;
  onRegionSelect?: (region: { x: number; y: number; w: number; h: number }) => void;
  onAskAboutRegion?: (prompt: string) => void;
}

const ROI_COLORS = ["#818CF8", "#34D399", "#FBBF24", "#FB7185", "#38BDF8"];

export default function MRIViewer({
  src,
  label = "MRI Scan",
  showTools = true,
  onRegionSelect,
  onAskAboutRegion,
}: MRIViewerProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);

  // View state
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [brightness, setBrightness] = useState(100);
  const [contrast, setContrast] = useState(100);

  // Active Tool: "pan" | "crosshair" | "roi" | "measure"
  const [tool, setTool] = useState<"pan" | "crosshair" | "roi" | "measure">("pan");
  const [crosshairPos, setCrosshairPos] = useState({ x: 0, y: 0 });

  // Pan interaction
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });

  // ROI state
  const [rois, setRois] = useState<ROIAnnotation[]>([]);
  const [drawingRoi, setDrawingRoi] = useState<{ startX: number; startY: number; currX: number; currY: number } | null>(null);
  const [editingRoi, setEditingRoi] = useState<ROIAnnotation | null>(null);

  // Measurement (Caliper) state
  const [calipers, setCalipers] = useState<CaliperMeasurement[]>([]);
  const [drawingCaliper, setDrawingCaliper] = useState<{ startX: number; startY: number; currX: number; currY: number } | null>(null);

  // Helper to convert canvas click coordinates into Image coordinates
  const canvasToImageCoords = useCallback((clientX: number, clientY: number) => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img) return { x: 0, y: 0 };
    const rect = canvas.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;

    // Inverse of: cx = canvas.width/2 + pan.x + (imgX - img.naturalWidth/2) * zoom
    const imgX = (cx - (canvas.width / 2 + pan.x)) / zoom + img.naturalWidth / 2;
    const imgY = (cy - (canvas.height / 2 + pan.y)) / zoom + img.naturalHeight / 2;
    return { x: imgX, y: imgY, canvasX: cx, canvasY: cy };
  }, [pan, zoom]);

  // Main Canvas Render
  const drawCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.complete) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply brightness and contrast filter to the MRI image
    ctx.filter = `brightness(${brightness}%) contrast(${contrast}%)`;
    ctx.save();
    ctx.translate(canvas.width / 2 + pan.x, canvas.height / 2 + pan.y);
    ctx.scale(zoom, zoom);
    ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2, img.naturalWidth, img.naturalHeight);
    ctx.filter = "none";

    // ── DRAW ROIS IN IMAGE SPACE ──
    const originX = -img.naturalWidth / 2;
    const originY = -img.naturalHeight / 2;

    rois.forEach((roi) => {
      ctx.strokeStyle = roi.color || "#818CF8";
      ctx.lineWidth = 2 / zoom;
      ctx.setLineDash([4 / zoom, 4 / zoom]);
      ctx.strokeRect(originX + roi.x, originY + roi.y, roi.w, roi.h);
      ctx.fillStyle = (roi.color || "#818CF8") + "22";
      ctx.fillRect(originX + roi.x, originY + roi.y, roi.w, roi.h);

      // Label badge
      ctx.setLineDash([]);
      ctx.fillStyle = roi.color || "#818CF8";
      ctx.font = `${Math.max(10, 12 / zoom)}px sans-serif`;
      ctx.fillText(roi.label, originX + roi.x, originY + roi.y - 4 / zoom);
    });

    // ── DRAW IN-PROGRESS ROI ──
    if (drawingRoi) {
      const rx = Math.min(drawingRoi.startX, drawingRoi.currX);
      const ry = Math.min(drawingRoi.startY, drawingRoi.currY);
      const rw = Math.abs(drawingRoi.currX - drawingRoi.startX);
      const rh = Math.abs(drawingRoi.currY - drawingRoi.startY);

      ctx.strokeStyle = "#FBBF24";
      ctx.lineWidth = 2 / zoom;
      ctx.setLineDash([4 / zoom, 2 / zoom]);
      ctx.strokeRect(originX + rx, originY + ry, rw, rh);
      ctx.fillStyle = "rgba(245, 158, 11, 0.15)";
      ctx.fillRect(originX + rx, originY + ry, rw, rh);
      ctx.setLineDash([]);
    }

    // ── DRAW CALIPERS IN IMAGE SPACE ──
    calipers.forEach((cal) => {
      ctx.strokeStyle = "#34D399";
      ctx.lineWidth = 2 / zoom;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(originX + cal.x1, originY + cal.y1);
      ctx.lineTo(originX + cal.x2, originY + cal.y2);
      ctx.stroke();

      // End tick marks
      const angle = Math.atan2(cal.y2 - cal.y1, cal.x2 - cal.x1);
      const tickLen = 6 / zoom;
      [
        { x: cal.x1, y: cal.y1 },
        { x: cal.x2, y: cal.y2 },
      ].forEach((pt) => {
        ctx.beginPath();
        ctx.moveTo(
          originX + pt.x - tickLen * Math.sin(angle),
          originY + pt.y + tickLen * Math.cos(angle)
        );
        ctx.lineTo(
          originX + pt.x + tickLen * Math.sin(angle),
          originY + pt.y - tickLen * Math.cos(angle)
        );
        ctx.stroke();
      });

      // Caliper measurement text
      const midX = (cal.x1 + cal.x2) / 2;
      const midY = (cal.y1 + cal.y2) / 2;
      ctx.fillStyle = "#34D399";
      ctx.font = `bold ${Math.max(10, 11 / zoom)}px sans-serif`;
      ctx.fillText(`${cal.distanceMm.toFixed(1)} mm`, originX + midX + 4 / zoom, originY + midY - 4 / zoom);
    });

    // ── DRAW IN-PROGRESS CALIPER ──
    if (drawingCaliper) {
      ctx.strokeStyle = "#38BDF8";
      ctx.lineWidth = 2 / zoom;
      ctx.setLineDash([3 / zoom, 3 / zoom]);
      ctx.beginPath();
      ctx.moveTo(originX + drawingCaliper.startX, originY + drawingCaliper.startY);
      ctx.lineTo(originX + drawingCaliper.currX, originY + drawingCaliper.currY);
      ctx.stroke();
      ctx.setLineDash([]);

      const dx = drawingCaliper.currX - drawingCaliper.startX;
      const dy = drawingCaliper.currY - drawingCaliper.startY;
      const dMm = Math.sqrt(dx * dx + dy * dy) * 0.5; // ~0.5 mm per pixel scale
      ctx.fillStyle = "#38BDF8";
      ctx.font = `bold ${Math.max(10, 11 / zoom)}px sans-serif`;
      ctx.fillText(`${dMm.toFixed(1)} mm`, originX + drawingCaliper.currX + 6 / zoom, originY + drawingCaliper.currY - 6 / zoom);
    }

    ctx.restore();

    // ── CROSSHAIR OVERLAY (IN SCREEN CANVAS SPACE) ──
    if (tool === "crosshair") {
      ctx.strokeStyle = "rgba(129, 140, 248, 0.75)";
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
  }, [
    zoom,
    pan,
    brightness,
    contrast,
    tool,
    crosshairPos,
    rois,
    drawingRoi,
    calipers,
    drawingCaliper,
  ]);

  useEffect(() => {
    if (!src) return;
    const img = new Image();
    img.src = src;
    img.onload = () => {
      imgRef.current = img;
      const canvas = canvasRef.current;
      if (canvas) {
        canvas.width = canvas.offsetWidth || 340;
        canvas.height = canvas.offsetHeight || 340;
      }
      drawCanvas();
    };
  }, [src, drawCanvas]);

  useEffect(() => {
    drawCanvas();
  }, [drawCanvas]);

  // Mouse Handlers
  const handleMouseDown = (e: React.MouseEvent) => {
    if (!src) return;
    const coords = canvasToImageCoords(e.clientX, e.clientY);

    if (tool === "pan") {
      setIsPanning(true);
      setPanStart({ x: e.clientX, y: e.clientY });
    } else if (tool === "roi") {
      setDrawingRoi({
        startX: coords.x,
        startY: coords.y,
        currX: coords.x,
        currY: coords.y,
      });
    } else if (tool === "measure") {
      setDrawingCaliper({
        startX: coords.x,
        startY: coords.y,
        currX: coords.x,
        currY: coords.y,
      });
    } else if (tool === "crosshair" && onRegionSelect) {
      onRegionSelect({ x: Math.round(coords.x), y: Math.round(coords.y), w: 24, h: 24 });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!src) return;
    const rect = canvasRef.current?.getBoundingClientRect();
    if (!rect) return;
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;

    if (tool === "crosshair") {
      setCrosshairPos({ x: cx, y: cy });
    }

    if (isPanning && tool === "pan") {
      setPan((p) => ({
        x: p.x + (e.clientX - panStart.x),
        y: p.y + (e.clientY - panStart.y),
      }));
      setPanStart({ x: e.clientX, y: e.clientY });
    } else if (drawingRoi && tool === "roi") {
      const coords = canvasToImageCoords(e.clientX, e.clientY);
      setDrawingRoi((r) => (r ? { ...r, currX: coords.x, currY: coords.y } : null));
    } else if (drawingCaliper && tool === "measure") {
      const coords = canvasToImageCoords(e.clientX, e.clientY);
      setDrawingCaliper((c) => (c ? { ...c, currX: coords.x, currY: coords.y } : null));
    }
  };

  const handleMouseUp = () => {
    if (isPanning) {
      setIsPanning(false);
    }

    if (drawingRoi) {
      const rx = Math.min(drawingRoi.startX, drawingRoi.currX);
      const ry = Math.min(drawingRoi.startY, drawingRoi.currY);
      const rw = Math.abs(drawingRoi.currX - drawingRoi.startX);
      const rh = Math.abs(drawingRoi.currY - drawingRoi.startY);

      // Only save if meaningful size
      if (rw > 8 && rh > 8) {
        const newRoi: ROIAnnotation = {
          id: `roi_${Date.now()}`,
          label: `ROI #${rois.length + 1}`,
          x: Math.round(rx),
          y: Math.round(ry),
          w: Math.round(rw),
          h: Math.round(rh),
          note: "",
          color: ROI_COLORS[rois.length % ROI_COLORS.length],
        };
        setRois((prev) => [...prev, newRoi]);
        setEditingRoi(newRoi);
        if (onRegionSelect) {
          onRegionSelect({ x: newRoi.x, y: newRoi.y, w: newRoi.w, h: newRoi.h });
        }
      }
      setDrawingRoi(null);
    }

    if (drawingCaliper) {
      const dx = drawingCaliper.currX - drawingCaliper.startX;
      const dy = drawingCaliper.currY - drawingCaliper.startY;
      const distPx = Math.sqrt(dx * dx + dy * dy);
      if (distPx > 6) {
        const newCaliper: CaliperMeasurement = {
          id: `cal_${Date.now()}`,
          x1: Math.round(drawingCaliper.startX),
          y1: Math.round(drawingCaliper.startY),
          x2: Math.round(drawingCaliper.currX),
          y2: Math.round(drawingCaliper.currY),
          distanceMm: distPx * 0.5,
        };
        setCalipers((prev) => [...prev, newCaliper]);
      }
      setDrawingCaliper(null);
    }
  };

  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
    setBrightness(100);
    setContrast(100);
  };

  // Windowing Presets
  const applyPreset = (preset: "brain" | "subdural" | "stroke" | "reset") => {
    if (preset === "brain") {
      setBrightness(100);
      setContrast(115);
    } else if (preset === "subdural") {
      setBrightness(125);
      setContrast(145);
    } else if (preset === "stroke") {
      setBrightness(90);
      setContrast(140);
    } else {
      resetView();
    }
  };

  // Ask AI about this region trigger
  const handleAskAI = (roi: ROIAnnotation) => {
    const prompt = `Evaluate the focal region at [X: ${roi.x}, Y: ${roi.y}, W: ${roi.w}, H: ${roi.h}] (${roi.label}). Doctor clinical note: ${
      roi.note ? `"${roi.note}"` : "Suspected lesion or focal signal abnormality"
    }. Is this finding indicative of acute pathology?`;
    if (onAskAboutRegion) {
      onAskAboutRegion(prompt);
    }
    setEditingRoi(null);
  };

  const updateRoiNote = (id: string, note: string) => {
    setRois((prev) => prev.map((r) => (r.id === id ? { ...r, note } : r)));
    if (editingRoi?.id === id) {
      setEditingRoi((prev) => (prev ? { ...prev, note } : null));
    }
  };

  const deleteRoi = (id: string) => {
    setRois((prev) => prev.filter((r) => r.id !== id));
    if (editingRoi?.id === id) setEditingRoi(null);
  };

  return (
    <div className="flex flex-col h-full select-none">
      {/* ── TOOLBAR HEADER ── */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          <span className="text-xs font-semibold text-[#94A3B8] uppercase tracking-wider">
            {label}
          </span>
          {rois.length > 0 && (
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#342CB3]/30 border border-[#342CB3] text-[#818CF8]">
              {rois.length} ROI{rois.length > 1 ? "s" : ""}
            </span>
          )}
          {calipers.length > 0 && (
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-[#10B981]/20 border border-[#10B981] text-[#34D399]">
              {calipers.length} Caliper{calipers.length > 1 ? "s" : ""}
            </span>
          )}
        </div>

        {showTools && (
          <div className="flex items-center gap-1 bg-[#161922] p-1 rounded-xl border border-[#2A2F42]">
            {/* Pan Tool */}
            <button
              onClick={() => setTool("pan")}
              className={`p-1.5 rounded-lg transition-colors ${
                tool === "pan" ? "bg-[#342CB3] text-white" : "text-[#94A3B8] hover:text-white hover:bg-[#1E2230]"
              }`}
              title="Pan / Navigate"
            >
              <Move size={13} />
            </button>

            {/* Crosshair Tool */}
            <button
              onClick={() => setTool("crosshair")}
              className={`p-1.5 rounded-lg transition-colors ${
                tool === "crosshair" ? "bg-[#342CB3] text-white" : "text-[#94A3B8] hover:text-white hover:bg-[#1E2230]"
              }`}
              title="Anatomical Crosshair Navigation"
            >
              <Crosshair size={13} />
            </button>

            {/* Bounding Box ROI Tool (FR-17) */}
            <button
              onClick={() => setTool("roi")}
              className={`p-1.5 rounded-lg transition-colors ${
                tool === "roi" ? "bg-[#342CB3] text-white" : "text-[#94A3B8] hover:text-white hover:bg-[#1E2230]"
              }`}
              title="Mark Region of Interest (ROI) with Clinical Notes"
            >
              <Square size={13} />
            </button>

            {/* Caliper Measurement Tool (FR-17) */}
            <button
              onClick={() => setTool("measure")}
              className={`p-1.5 rounded-lg transition-colors ${
                tool === "measure" ? "bg-[#342CB3] text-white" : "text-[#94A3B8] hover:text-white hover:bg-[#1E2230]"
              }`}
              title="Caliper Distance Measurement (mm)"
            >
              <Ruler size={13} />
            </button>

            <div className="w-[1px] h-4 bg-[#2A2F42] mx-0.5" />

            {/* Zoom In */}
            <button
              onClick={() => setZoom((z) => Math.min(z + 0.25, 4))}
              className="p-1.5 rounded-lg text-[#94A3B8] hover:text-white hover:bg-[#1E2230] transition-colors"
              title="Zoom In"
            >
              <ZoomIn size={13} />
            </button>

            {/* Zoom Out */}
            <button
              onClick={() => setZoom((z) => Math.max(z - 0.25, 0.5))}
              className="p-1.5 rounded-lg text-[#94A3B8] hover:text-white hover:bg-[#1E2230] transition-colors"
              title="Zoom Out"
            >
              <ZoomOut size={13} />
            </button>

            {/* Reset */}
            <button
              onClick={resetView}
              className="p-1.5 rounded-lg text-[#94A3B8] hover:text-white hover:bg-[#1E2230] transition-colors"
              title="Reset View and Windowing"
            >
              <RotateCcw size={13} />
            </button>
          </div>
        )}
      </div>

      {/* ── CANVAS VIEWPORT ── */}
      <div
        ref={containerRef}
        className="relative flex-1 bg-[#0E1117] rounded-xl overflow-hidden border border-[#2A2F42] min-h-[260px] flex items-center justify-center"
        onMouseMove={handleMouseMove}
        onMouseDown={handleMouseDown}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        style={{
          cursor:
            tool === "pan"
              ? isPanning
                ? "grabbing"
                : "grab"
              : tool === "roi" || tool === "measure"
              ? "crosshair"
              : "default",
        }}
      >
        {src ? (
          <canvas ref={canvasRef} className="w-full h-full block" />
        ) : (
          <div className="absolute inset-0 flex flex-col items-center justify-center p-6 text-center">
            <div className="w-14 h-14 rounded-2xl bg-[#1E2230] border border-[#2A2F42] flex items-center justify-center mb-3">
              <span className="text-2xl">🧠</span>
            </div>
            <p className="text-sm font-semibold text-white mb-1">No MRI Scan Loaded</p>
            <p className="text-[#64748B] text-xs max-w-xs">
              Upload an axial MRI slice or select a clinical case to launch decision support.
            </p>
          </div>
        )}

        {/* Floating Tool Instructions Pill */}
        {src && showTools && (
          <div className="absolute bottom-2 left-2 bg-[#161922]/90 backdrop-blur border border-[#2A2F42] rounded-lg px-2.5 py-1 text-[10px] text-[#94A3B8] flex items-center gap-2">
            <span>
              Mode: <strong className="text-white capitalize">{tool}</strong>
            </span>
            <span>•</span>
            <span>Zoom: {zoom.toFixed(1)}x</span>
            {tool === "crosshair" && (
              <>
                <span>•</span>
                <span>
                  Pos: ({Math.round(crosshairPos.x)}, {Math.round(crosshairPos.y)})
                </span>
              </>
            )}
          </div>
        )}
      </div>

      {/* ── ATTACHED ROI NOTE / ASK AI MODAL OVERLAY ── */}
      {editingRoi && (
        <div className="mt-2 p-3 bg-[#161922] border border-[#342CB3]/50 rounded-xl shadow-lg">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-full"
                style={{ backgroundColor: editingRoi.color }}
              />
              <span className="text-xs font-bold text-white">{editingRoi.label}</span>
              <span className="text-[10px] text-[#64748B]">
                ({editingRoi.w}×{editingRoi.h}px at X:{editingRoi.x}, Y:{editingRoi.y})
              </span>
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={() => deleteRoi(editingRoi.id)}
                className="p-1 text-[#94A3B8] hover:text-[#FB7185] transition-colors"
                title="Delete ROI"
              >
                <Trash2 size={12} />
              </button>
              <button
                onClick={() => setEditingRoi(null)}
                className="p-1 text-[#94A3B8] hover:text-white transition-colors"
              >
                <X size={12} />
              </button>
            </div>
          </div>

          <div className="mb-2">
            <label className="text-[10px] text-[#94A3B8] block mb-1">
              Attach Clinical Note to this Region:
            </label>
            <textarea
              value={editingRoi.note}
              onChange={(e) => updateRoiNote(editingRoi.id, e.target.value)}
              placeholder="e.g. Suspected focal high-intensity lesion in right frontoparietal cortex..."
              rows={2}
              className="w-full bg-[#0E1117] border border-[#2A2F42] rounded-lg p-2 text-xs text-white placeholder-[#64748B] resize-none focus:outline-none focus:border-[#4F46E5]"
            />
          </div>

          <div className="flex items-center justify-between">
            <button
              onClick={() => setEditingRoi(null)}
              className="px-2.5 py-1 text-[11px] text-[#94A3B8] hover:text-white transition-colors flex items-center gap-1"
            >
              <Check size={11} /> Done
            </button>

            {/* "Ask AI about this region" (Design §5.7) */}
            <button
              onClick={() => handleAskAI(editingRoi)}
              className="px-3 py-1 bg-gradient-to-r from-[#4F46E5] to-[#342CB3] hover:from-[#6366F1] hover:to-[#4338CA] text-white text-[11px] font-semibold rounded-lg shadow flex items-center gap-1.5 transition-all"
            >
              <Sparkles size={11} />
              <span>Ask AI About This Region</span>
            </button>
          </div>
        </div>
      )}

      {/* ── SAVED ROIS & CALIPERS TABS ── */}
      {showTools && (rois.length > 0 || calipers.length > 0) && !editingRoi && (
        <div className="mt-2 flex flex-wrap gap-1.5 max-h-20 overflow-y-auto">
          {rois.map((roi) => (
            <button
              key={roi.id}
              onClick={() => setEditingRoi(roi)}
              className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-[#161922] border border-[#2A2F42] hover:border-[#818CF8] text-[10px] text-[#94A3B8] hover:text-white transition-all"
            >
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: roi.color }} />
              <span>{roi.label}</span>
              {roi.note && <MessageSquare size={10} className="text-[#818CF8] ml-0.5" />}
            </button>
          ))}
          {calipers.map((cal, idx) => (
            <span
              key={cal.id}
              className="flex items-center gap-1 px-2 py-0.5 rounded-lg bg-[#161922] border border-[#2A2F42] text-[10px] text-[#34D399]"
            >
              <Ruler size={10} />
              <span>D{idx + 1}: {cal.distanceMm.toFixed(1)} mm</span>
              <button
                onClick={() => setCalipers((c) => c.filter((x) => x.id !== cal.id))}
                className="text-[#64748B] hover:text-[#FB7185] ml-0.5"
              >
                ×
              </button>
            </span>
          ))}
        </div>
      )}

      {/* ── WINDOWING PRESETS & BRIGHTNESS/CONTRAST CONTROLS ── */}
      {showTools && (
        <div className="mt-2 space-y-2 border-t border-[#2A2F42]/60 pt-2">
          {/* PACS Window Presets */}
          <div className="flex items-center justify-between text-[10px]">
            <span className="text-[#64748B] font-semibold uppercase">Window Presets:</span>
            <div className="flex items-center gap-1">
              <button
                onClick={() => applyPreset("brain")}
                className="px-2 py-0.5 rounded bg-[#161922] hover:bg-[#1E2230] border border-[#2A2F42] text-[#94A3B8] hover:text-white transition-colors"
              >
                Brain
              </button>
              <button
                onClick={() => applyPreset("stroke")}
                className="px-2 py-0.5 rounded bg-[#161922] hover:bg-[#1E2230] border border-[#2A2F42] text-[#94A3B8] hover:text-white transition-colors"
              >
                Stroke
              </button>
              <button
                onClick={() => applyPreset("subdural")}
                className="px-2 py-0.5 rounded bg-[#161922] hover:bg-[#1E2230] border border-[#2A2F42] text-[#94A3B8] hover:text-white transition-colors"
              >
                Subdural
              </button>
              <button
                onClick={() => applyPreset("reset")}
                className="px-2 py-0.5 rounded bg-[#161922] hover:bg-[#1E2230] border border-[#2A2F42] text-[#94A3B8] hover:text-white transition-colors"
              >
                Default
              </button>
            </div>
          </div>

          {/* Sliders */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-[#94A3B8] flex items-center gap-1">
                  <Sun size={10} /> Brightness
                </span>
                <span className="text-[10px] text-[#64748B]">{brightness}%</span>
              </div>
              <input
                type="range"
                min={50}
                max={200}
                value={brightness}
                onChange={(e) => setBrightness(Number(e.target.value))}
                className="w-full h-1 rounded accent-[#4F46E5] bg-[#1E2230]"
              />
            </div>
            <div>
              <div className="flex items-center justify-between mb-1">
                <span className="text-[10px] text-[#94A3B8] flex items-center gap-1">
                  <Contrast size={10} /> Contrast
                </span>
                <span className="text-[10px] text-[#64748B]">{contrast}%</span>
              </div>
              <input
                type="range"
                min={50}
                max={200}
                value={contrast}
                onChange={(e) => setContrast(Number(e.target.value))}
                className="w-full h-1 rounded accent-[#4F46E5] bg-[#1E2230]"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
