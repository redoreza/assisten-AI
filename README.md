# AI Avatar Companion — Web-Based, Laptop Dev → VPS Deploy

## 0. Konteks Developer

- **OS:** Windows 11
- **Hardware:** 16GB RAM, NVIDIA GPU 4-6GB VRAM
- **Strategi:** Develop di laptop (localhost), deploy ke VPS murah nanti
- **Filosofi:** Server tipis, browser tebal, AI lewat API gratis tier

## 1. Tujuan Proyek

Membangun AI avatar virtual web-based dengan karakter fiksi yang bisa:
- **Mode Companion:** diajak ngobrol natural pakai voice (mic → suara karakter)
- **Mode Customer Service:** jawab pertanyaan berdasarkan knowledge base (RAG)
- Avatar dirender di browser dengan lip-sync real-time
- Akses dari mana saja via web URL

## 2. Tech Stack (Final)

### Frontend (jalan di browser user)
| Komponen | Library | Catatan |
|---|---|---|
| Framework | **Next.js 15** + TypeScript | App Router |
| UI Components | shadcn/ui + Tailwind CSS | |
| Avatar 3D | **Three.js** + `@pixiv/three-vrm` | Untuk model VRM dari VRoid |
| Lip-sync | **TalkingHead.js** (MIT) | Fonem → blendshape |
| Audio rec | MediaRecorder API | Native browser |
| Audio play | Web Audio API | Native browser |
| WebSocket | native WebSocket | |
| State mgmt | Zustand | Ringan |

### Backend (jalan di laptop saat dev, di VPS saat production)
| Komponen | Library | Catatan |
|---|---|---|
| Framework | **FastAPI** (Python 3.11) | Async, WebSocket native |
| HTTP client | `httpx` | Async |
| Orchestration | LangChain (light usage) | Untuk RAG saja |
| Vector DB | **ChromaDB** | Embedded, no server |
| Database | **SQLite** + `aiosqlite` | File-based |
| Audio utils | `pydub` + FFmpeg | Format conversion |
| Logging | `loguru` | |
| Config | `pydantic-settings` + `.env` | |

### External APIs (semua free tier)
| Layanan | Fungsi | Free tier |
|---|---|---|
| **Groq** | LLM (Llama 3.3 70B) + STT (Whisper-large-v3-turbo) | Sangat besar, super cepat |
| **Edge TTS** | Text-to-speech | Tanpa batas, gratis selamanya |
| **Voyage AI** | Embeddings (untuk RAG) | 50M token/bulan |
| **Google Gemini** (backup) | LLM fallback | 1500 req/hari Flash |

## 3. Kenapa Stack Ini Cocok untuk Laptop Anda

- **RAM 16GB cukup banget** — backend cuma butuh 300-500MB, frontend dev server ~500MB
- **GPU 4-6GB tidak terpakai berat** — semua AI heavy lifting di cloud (Groq)
- **Tidak perlu local LLM** — Groq API gratis dan jauh lebih cepat
- **Windows-friendly** — tidak perlu WSL atau Docker untuk dev

## 4. Struktur Folder

```
assisten-AI/
├── README.md
├── .env.example
├── .gitignore
├── frontend/                       # Next.js (port 3000)
│   ├── app/
│   │   ├── page.tsx                # Chat + avatar UI
│   │   └── admin/page.tsx          # Upload knowledge base
│   ├── components/
│   │   ├── Avatar3D.tsx
│   │   ├── ChatInterface.tsx
│   │   ├── CameraView.tsx
│   │   ├── VoiceRecorder.tsx
│   │   └── VadListener.tsx
│   └── lib/
│       ├── websocket.ts
│       ├── audioQueue.ts
│       ├── store.ts
│       └── types.ts
├── backend/                        # FastAPI (port 8000)
│   └── app/
│       ├── main.py
│       ├── config.py
│       ├── api/
│       │   ├── websocket.py
│       │   ├── chat.py
│       │   └── face.py
│       ├── services/
│       │   ├── stt_groq.py
│       │   ├── llm_groq.py
│       │   ├── tts_edge.py
│       │   └── face_recognition.py
│       ├── core/
│       │   ├── orchestrator.py
│       │   ├── persona.py
│       │   └── viseme.py
│       └── recognition/
│           └── face_database.py
├── data/                           # gitignored
│   ├── personas/
│   ├── knowledge_base/
│   ├── chroma/
│   └── sqlite/
└── scripts/
```

## 5. Alur Data End-to-End

### Mode Voice Chat
```
[USER BROWSER]
1. User klik mic → MediaRecorder rekam audio (webm/opus)
2. VAD client-side deteksi akhir ucapan → kirim via WS:
   { type: "audio_chunk", data: base64, format: "webm" }

[BACKEND - FastAPI]
3. Terima audio → Groq Whisper STT
4. Transcript → Groq Llama 3.3 70B (streaming)
5. Per kalimat selesai → Edge TTS → audio mp3 + word boundaries
6. Kirim ke client per kalimat:
   { type: "ai_text", text: "..." }
   { type: "audio", data: base64_mp3 }
   { type: "viseme", events: [...] }

[USER BROWSER]
7. AudioQueue putar mp3 (gapless)
8. Sync viseme → VRM blendshape (lip-sync)
```

## 6. WebSocket Protocol

**Client → Server:**
```typescript
type ClientMsg =
  | { type: "audio_chunk"; data: string; format: "webm" }
  | { type: "text"; message: string }
  | { type: "set_persona"; persona_id: string }
  | { type: "set_mode"; mode: "companion" | "customer_service" }
  | { type: "face_present"; match_name: string | null; image_base64: string }
  | { type: "face_lost" }
```

**Server → Client:**
```typescript
type ServerMsg =
  | { type: "transcript"; text: string }
  | { type: "ai_text"; text: string; is_final: boolean }
  | { type: "audio"; data: string; format: "mp3"; sequence: number }
  | { type: "viseme"; events: VisemeEvent[]; audio_seq: number }
  | { type: "error"; message: string }
  | { type: "done" }
```

## 7. Persona Config Format

```json
{
  "id": "pointer",
  "name": "Pointer",
  "avatar_file": "/avatars/pointer.vrm",
  "language": "id",
  "personality": "Ramah, informatif, suportif",
  "background": "Asisten virtual kampus Polinela.",
  "speaking_style": "Casual, panggil user dengan 'kamu'",
  "voice": {
    "provider": "edge",
    "voice_id": "id-ID-GadisNeural",
    "rate": "+5%",
    "pitch": "+2Hz"
  },
  "system_prompt_template": "Kamu adalah {name}. {personality} Background: {background}. Cara bicara: {speaking_style}. Jawab dalam bahasa {language}. JANGAN keluar dari karakter."
}
```

## 8. Konfigurasi `.env`

Salin `.env.example` ke `.env` lalu isi:

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxx
VOYAGE_API_KEY=pa-xxxxxxxxxxxxx
TAVILY_API_KEY=tvly-xxxxxxxxxxxxx  # opsional, web search

LLM_MODEL=llama-3.3-70b-versatile
STT_MODEL=whisper-large-v3-turbo
TTS_VOICE_DEFAULT=id-ID-GadisNeural

HOST=127.0.0.1
PORT=8000
CORS_ORIGINS=http://localhost:3000
DEFAULT_PERSONA=pointer
DEFAULT_MODE=companion
```

## 9. Setup (Windows)

### Prerequisites (install manual sekali saja)

| Tool | Link | Catatan |
|------|------|---------|
| Python 3.11 atau 3.12 | https://python.org | Centang "Add to PATH" saat install |
| Node.js 20+ LTS | https://nodejs.org | Pilih LTS |
| Git | https://git-scm.com | |
| FFmpeg | `winget install ffmpeg` | Untuk konversi audio webm→wav |
| Visual C++ Build Tools | `winget install Microsoft.VisualStudio.2022.BuildTools` | **Wajib** untuk InsightFace. Pilih workload "Desktop development with C++". Tanpa ini `uv sync` gagal. |

### Setup otomatis

Setelah semua prerequisites terinstall, jalankan dari PowerShell:

```powershell
git clone https://github.com/redoreza/assisten-AI.git
cd assisten-AI
.\scripts\setup_windows.ps1
```

Script otomatis melakukan:
- Verifikasi semua prerequisites dan beri pesan jelas kalau ada yang kurang
- Install `uv` jika belum ada
- `uv sync` — install semua Python dependencies dari `uv.lock`
- `npm install` — install semua Node.js dependencies dari `package-lock.json`
- Buat folder `data/sqlite`, `data/chroma`, `data/knowledge_base`, `data/personas`
- Salin `.env.example` → `.env`

### Setelah setup

1. Buka `.env` dan isi API keys:
   - `GROQ_API_KEY` — **wajib**, gratis di https://console.groq.com
   - `TAVILY_API_KEY` — opsional, web search, gratis di https://app.tavily.com

2. Pastikan file persona ada: `data/personas/pointer.json` (lihat format di §7)

3. Jalankan aplikasi — double-click `start-pointer.bat`
   atau manual di dua terminal terpisah:
   ```powershell
   # Terminal 1 — backend
   cd backend
   uv run uvicorn app.main:app --reload --port 8000

   # Terminal 2 — frontend
   cd frontend
   npm run dev
   ```

4. Buka http://localhost:3000

> **Catatan:** Pertama kali jalan, InsightFace akan download model `buffalo_l` (~300 MB) otomatis. Hanya sekali.

## 10. Milestone Development

### Phase 1–3 + R1 (Selesai) ✅
- Backend Foundation (FastAPI, persona, LLM, STT, TTS)
- Voice Pipeline (WebSocket, orchestrator, viseme)
- Frontend UI (Chat, VAD, AudioQueue)
- Face Recognition (InsightFace, enrollment, auto-greet, adaptive learning)

### Phase 4 — Avatar 3D (TODO)
- [ ] `Avatar3D.tsx` (Three.js + three-vrm)
- [ ] `lib/lipSync.ts` (viseme → VRM blendshape)
- [ ] VRM asset

### Phase 5 — Memory & RAG (TODO)
- [ ] `core/rag.py` + ChromaDB
- [ ] `services/embedding_voyage.py`
- [ ] `scripts/ingest_kb.py`

### Phase 6 — Deploy (TODO)
- [ ] Docker setup
- [ ] VPS deployment

## 11. Tantangan & Solusi

**Latency target: < 2 detik first audio**
- STT (Groq Whisper turbo): ~300–500ms
- LLM first token: ~200–400ms
- TTS first sentence: ~500–800ms

Strategi: streaming di semua step, TTS dimulai per kalimat.

**Demo tanpa deploy:** Cloudflare Tunnel
```powershell
winget install --id Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8000
```

## 12. Avatar Asset (VRM)

1. **Bikin sendiri:** VRoid Studio (gratis di Steam) → export VRM
2. **Download:** [VRoid Hub](https://hub.vroid.com) atau [Booth](https://booth.pm)

Taruh file di `frontend/public/avatars/` dan referensikan di persona JSON.

## 13. Referensi

- [Groq API](https://console.groq.com/docs)
- [Edge TTS](https://github.com/rany2/edge-tts)
- [three-vrm](https://github.com/pixiv/three-vrm)
- [Voyage AI](https://docs.voyageai.com)
- [VRoid Studio](https://vroid.com/en/studio)
- [FastAPI WebSocket](https://fastapi.tiangolo.com/advanced/websockets/)

## 14. Estimasi Biaya

| Item | Biaya |
|------|-------|
| Development (localhost) | Rp 0 |
| VPS Contabo/Vultr | ~Rp 90.000–180.000/bulan |
| Groq API | Gratis (hobby traffic) |
| Edge TTS | Gratis selamanya |
| Voyage AI | Gratis s/d 50M token/bulan |
