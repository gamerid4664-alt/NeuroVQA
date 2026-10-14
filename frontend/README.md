# NeuroVQA — Frontend Client

The frontend client for **NeuroVQA**: an offline, MRI-grounded clinical decision-support assistant for brain MRI Visual Question Answering.

---

## Key Features

- **Diagnostic Workspace (`/app`):**
  - Clinical intake for axial MRI slices (2D and 2.5D stacks).
  - Side-by-side viewer for raw scan and Grad-CAM $14 \times 14$ attention heatmaps.
  - Interactive medical image controls: Pan, Zoom, Windowing (Brightness/Contrast), Crosshairs.
  - VQA diagnostic card displaying top-1 finding, cross-attention confidence gauge, and top-3 ranked predictions.
  - Clinician feedback module (Correct, Partial, Incorrect, Uncertain) and one-click structured PDF report generator.
- **Real-Time Clinical Chat (`ChatPanel`):**
  - Streaming token-by-token WebSocket communication with local Ollama (`llama3.2:1b`).
  - Strict grounding: explanations are constrained exclusively to verified VQA output and focused Grad-CAM attention regions.
  - Medical decision-support disclaimer visibly maintained.
- **Analytics Dashboard (`/dashboard`):**
  - KPI telemetry: Total Cases, Pathologies Detected, Normal Findings, Inconclusive Scans, Average Confidence %, Inference Latency.
  - Recharts visual graphs for confidence distribution and pathology categorization.
  - Recent diagnostic activity log.
- **Diagnostic Case History (`/history`):**
  - Case search and filtering by pathology or status.
  - One-click case reload into the workspace for secondary review.
  - Historical report download.
- **Authentication & Security (`/login`, `/signup`):**
  - JWT session management with auto-logout on token expiration.

---

## Tech Stack

- **Framework:** React 19 + TypeScript + Vite
- **Styling:** Tailwind CSS v4 (Custom Dark-Slate Clinical Theme `#0B0D11`, `#161922`, `#1E2230`, `#342CB3`, `#10B981`, `#E11D48`)
- **Animations:** Framer Motion
- **Icons:** Lucide React
- **Data Visualization:** Recharts
- **State Management:** Zustand with local persistence
- **Routing:** React Router v7

---

## Development & Build

```bash
# Install dependencies
npm install

# Start development server (http://localhost:5173)
npm run dev

# Build production bundle (served statically via FastAPI backend)
npm run build

# Preview production build
npm run preview
```
