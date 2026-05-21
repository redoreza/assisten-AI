'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { recognizeFace, type FaceMatch } from '@/lib/faceApi'
import { useChatStore } from '@/lib/store'

const RECOGNIZE_INTERVAL_MS = 500
/** Downscale before sending — 480px is what SCRFD effectively works on at det_size 320. */
const CAPTURE_MAX_WIDTH = 480
const JPEG_QUALITY = 0.6
/** Cap on extrapolation factor to avoid bbox overshoot when face stops moving. */
const INTERP_MAX_K = 0.5

interface CameraDevice {
  deviceId: string
  label: string
}

/** Heuristic: prefer USB / external / known webcam labels over laptop built-in. */
function autoPickIndex(cams: CameraDevice[]): number {
  const i = cams.findIndex((c) => /usb|external|medical|c920|c922|c270|webcam pro/i.test(c.label))
  return i >= 0 ? i : 0
}

export function CameraView() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const captureRef = useRef<HTMLCanvasElement>(null)

  const [devices, setDevices] = useState<CameraDevice[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('')
  const [streamActive, setStreamActive] = useState(false)
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [statusMsg, setStatusMsg] = useState<string>('Initializing…')
  const [faces, setFaces] = useState<FaceMatch[]>([])
  const [lastLatencyMs, setLastLatencyMs] = useState<number | null>(null)
  const [recognizing, setRecognizing] = useState(false)

  const streamRef = useRef<MediaStream | null>(null)
  const recognizeTimerRef = useRef<number | null>(null)
  const inFlightRef = useRef(false)
  const hasInitedRef = useRef(false)

  // Bbox interpolation state — track last two server responses + timestamps.
  // Between server updates, the rAF loop extrapolates forward using linear
  // velocity for smooth motion at 60 fps even when inference runs at ~2 fps.
  // Stored in refs (not state) to avoid re-renders on every update.
  const lastFacesRef = useRef<FaceMatch[]>([])
  const prevFacesRef = useRef<FaceMatch[]>([])
  const lastFacesTimeRef = useRef<number>(0)
  const prevFacesTimeRef = useRef<number>(0)
  const rafRef = useRef<number | null>(null)

  // WS face-event throttling — only emit face_present once per second when stable
  const lastFaceEventAtRef = useRef<number>(0)
  const lastFaceLostSentRef = useRef<boolean>(true) // start "lost" so first detection emits

  const sendFacePresent = useChatStore((s) => s.sendFacePresent)
  const sendFaceLost = useChatStore((s) => s.sendFaceLost)
  const setFacePresent = useChatStore((s) => s.setFacePresent)
  const wsReady = useChatStore((s) => s.ready)
  const awaitingName = useChatStore((s) => s.awaitingName)

  const stopStream = useCallback(() => {
    if (recognizeTimerRef.current !== null) {
      window.clearInterval(recognizeTimerRef.current)
      recognizeTimerRef.current = null
    }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setStreamActive(false)
    setFaces([])
  }, [])

  /** Open the stream for a specific device (or default camera if deviceId empty). */
  const openStream = useCallback(
    async (deviceId: string) => {
      stopStream()
      setErrorMsg(null)
      setStatusMsg('Membuka kamera…')
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            deviceId: deviceId ? { exact: deviceId } : undefined,
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        })
        streamRef.current = stream
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
        }
        setStreamActive(true)
        setStatusMsg('Kamera aktif')
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e)
        console.error('[CameraView] getUserMedia failed', e)
        setErrorMsg(`Tidak bisa membuka kamera: ${msg}`)
        setStreamActive(false)
        setStatusMsg('Gagal membuka kamera')
      }
    },
    [stopStream]
  )

  /** Init: request permission FIRST so enumerateDevices() returns proper labels.
   *  This pattern is mandatory for Firefox + matches the working face_web.html. */
  const init = useCallback(async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setErrorMsg('Browser tidak support navigator.mediaDevices (butuh HTTPS atau localhost)')
      setStatusMsg('mediaDevices tidak tersedia')
      return
    }
    setStatusMsg('Meminta izin kamera…')
    try {
      const tmp = await navigator.mediaDevices.getUserMedia({ video: true })
      tmp.getTracks().forEach((t) => t.stop())
    } catch (e) {
      console.error('[CameraView] permission request failed', e)
      const msg = e instanceof Error ? e.message : String(e)
      setErrorMsg(`Izin kamera ditolak: ${msg}`)
      setStatusMsg('Izin ditolak')
      return
    }

    setStatusMsg('Mendaftar kamera…')
    let cams: CameraDevice[] = []
    try {
      const all = await navigator.mediaDevices.enumerateDevices()
      cams = all
        .filter((d) => d.kind === 'videoinput')
        .map((d, idx) => ({
          deviceId: d.deviceId,
          label: d.label || `Kamera ${idx + 1}`,
        }))
    } catch (e) {
      console.error('[CameraView] enumerateDevices failed', e)
      const msg = e instanceof Error ? e.message : String(e)
      setErrorMsg(`Tidak bisa menampilkan daftar kamera: ${msg}`)
      return
    }

    if (cams.length === 0) {
      setErrorMsg('Tidak ada kamera terdeteksi. Pastikan USB camera tercolok dan tidak dipakai aplikasi lain.')
      setStatusMsg('Tidak ada kamera')
      return
    }

    setDevices(cams)
    const idx = autoPickIndex(cams)
    const pickId = cams[idx].deviceId
    setSelectedDeviceId(pickId)
    await openStream(pickId)
  }, [openStream])

  // Run init once on mount
  useEffect(() => {
    if (hasInitedRef.current) return
    hasInitedRef.current = true
    void init()
    const onDeviceChange = () => {
      // Re-enumerate when user plugs/unplugs cameras; don't auto-switch though.
      navigator.mediaDevices.enumerateDevices().then((all) => {
        const cams = all
          .filter((d) => d.kind === 'videoinput')
          .map((d, idx) => ({
            deviceId: d.deviceId,
            label: d.label || `Kamera ${idx + 1}`,
          }))
        setDevices(cams)
      })
    }
    navigator.mediaDevices?.addEventListener('devicechange', onDeviceChange)
    return () => {
      navigator.mediaDevices?.removeEventListener('devicechange', onDeviceChange)
      stopStream()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // When user picks a different camera from the dropdown, re-open stream
  const onPickCamera = useCallback(
    (deviceId: string) => {
      setSelectedDeviceId(deviceId)
      void openStream(deviceId)
    },
    [openStream]
  )

  const captureFrame = useCallback((): {
    base64: string
    videoWidth: number
    videoHeight: number
    captureWidth: number
    captureHeight: number
  } | null => {
    const video = videoRef.current
    const canvas = captureRef.current
    if (!video || !canvas || video.videoWidth === 0) return null
    const vw = video.videoWidth
    const vh = video.videoHeight
    const scale = Math.min(1, CAPTURE_MAX_WIDTH / vw)
    const cw = Math.round(vw * scale)
    const ch = Math.round(vh * scale)
    canvas.width = cw
    canvas.height = ch
    const ctx = canvas.getContext('2d')
    if (!ctx) return null
    ctx.drawImage(video, 0, 0, cw, ch)
    const dataUrl = canvas.toDataURL('image/jpeg', JPEG_QUALITY)
    return {
      base64: dataUrl,
      videoWidth: vw,
      videoHeight: vh,
      captureWidth: cw,
      captureHeight: ch,
    }
  }, [])

  const recognizeTick = useCallback(async () => {
    if (inFlightRef.current) return
    const frame = captureFrame()
    if (!frame) return
    inFlightRef.current = true
    setRecognizing(true)
    const t0 = performance.now()
    try {
      const res = await recognizeFace(frame.base64)
      const sx = frame.videoWidth / frame.captureWidth
      const sy = frame.videoHeight / frame.captureHeight
      const scaled = res.faces.map((f) => ({
        ...f,
        bbox: {
          x: Math.round(f.bbox.x * sx),
          y: Math.round(f.bbox.y * sy),
          width: Math.round(f.bbox.width * sx),
          height: Math.round(f.bbox.height * sy),
        },
      }))
      // Stash history for rAF interpolation. Refs first (no re-render),
      // then setFaces only for the bottom status bar text.
      prevFacesRef.current = lastFacesRef.current
      prevFacesTimeRef.current = lastFacesTimeRef.current
      lastFacesRef.current = scaled
      lastFacesTimeRef.current = performance.now()
      setFaces(scaled)
      setLastLatencyMs(Math.round(performance.now() - t0))

      // Mirror face presence into the store so VadListener (and any other
      // consumer) can gate behavior on whether someone is in front of camera.
      setFacePresent(scaled.length > 0)

      // ── Emit face state to backend for greet / enrollment flow ──────────
      if (wsReady) {
        const now = performance.now()
        if (scaled.length === 0) {
          if (!lastFaceLostSentRef.current) {
            sendFaceLost()
            lastFaceLostSentRef.current = true
          }
        } else {
          // Pick the largest face (most prominent / closest to camera)
          const largest = scaled.reduce((a, b) =>
            a.bbox.width * a.bbox.height >= b.bbox.width * b.bbox.height ? a : b
          )
          // Throttle to 1Hz when stable, but always send immediately on transition
          const stateChanged = lastFaceLostSentRef.current
          if (stateChanged || now - lastFaceEventAtRef.current >= 1000) {
            sendFacePresent({
              match_name: largest.match_name,
              match_person_id: largest.match_person_id,
              similarity: largest.similarity,
              image_base64: frame.base64,
            })
            lastFaceEventAtRef.current = now
            lastFaceLostSentRef.current = false
          }
        }
      }
    } catch (e: unknown) {
      console.error('[CameraView] recognize failed', e)
    } finally {
      inFlightRef.current = false
      setRecognizing(false)
    }
  }, [captureFrame, sendFacePresent, sendFaceLost, setFacePresent, wsReady])

  // Drive recognize loop when stream is active
  useEffect(() => {
    if (!streamActive) return
    void recognizeTick()
    recognizeTimerRef.current = window.setInterval(() => {
      void recognizeTick()
    }, RECOGNIZE_INTERVAL_MS)
    return () => {
      if (recognizeTimerRef.current !== null) {
        window.clearInterval(recognizeTimerRef.current)
        recognizeTimerRef.current = null
      }
    }
  }, [streamActive, recognizeTick])

  // rAF render loop — interpolates bbox positions between server updates.
  // Pattern adapted from detection-engine/face_web.html lines 519-570.
  // Server runs at ~2 fps; this draws at 60 fps so motion looks smooth.
  useEffect(() => {
    const overlay = overlayRef.current
    const video = videoRef.current
    if (!overlay || !video) return

    const matchByCenter = (target: FaceMatch, pool: FaceMatch[]): FaceMatch | null => {
      if (pool.length === 0) return null
      const tx = target.bbox.x + target.bbox.width / 2
      const ty = target.bbox.y + target.bbox.height / 2
      let best: FaceMatch | null = null
      let bestD = Infinity
      for (const c of pool) {
        const dx = c.bbox.x + c.bbox.width / 2 - tx
        const dy = c.bbox.y + c.bbox.height / 2 - ty
        const d = dx * dx + dy * dy
        if (d < bestD) {
          bestD = d
          best = c
        }
      }
      return best
    }

    const render = () => {
      const w = video.clientWidth
      const h = video.clientHeight
      if (overlay.width !== w || overlay.height !== h) {
        overlay.width = w
        overlay.height = h
      }
      const ctx = overlay.getContext('2d')
      if (!ctx) {
        rafRef.current = requestAnimationFrame(render)
        return
      }
      ctx.clearRect(0, 0, w, h)
      if (video.videoWidth === 0 || lastFacesRef.current.length === 0) {
        rafRef.current = requestAnimationFrame(render)
        return
      }

      const elAspect = w / h
      const vidAspect = video.videoWidth / video.videoHeight
      let renderedW: number
      let renderedH: number
      let offX: number
      let offY: number
      if (vidAspect > elAspect) {
        renderedW = w
        renderedH = w / vidAspect
        offX = 0
        offY = (h - renderedH) / 2
      } else {
        renderedH = h
        renderedW = h * vidAspect
        offX = (w - renderedW) / 2
        offY = 0
      }
      const sx = renderedW / video.videoWidth
      const sy = renderedH / video.videoHeight

      // Interpolation factor between prev and last bbox positions
      const dtUpdates = lastFacesTimeRef.current - prevFacesTimeRef.current
      const sinceLast = performance.now() - lastFacesTimeRef.current
      const k = dtUpdates > 0 ? Math.min(INTERP_MAX_K, sinceLast / dtUpdates) : 0

      for (const last of lastFacesRef.current) {
        const prev = matchByCenter(last, prevFacesRef.current)
        let x = last.bbox.x
        let y = last.bbox.y
        let bw = last.bbox.width
        let bh = last.bbox.height
        if (prev && k > 0) {
          x = last.bbox.x + (last.bbox.x - prev.bbox.x) * k
          y = last.bbox.y + (last.bbox.y - prev.bbox.y) * k
          bw = last.bbox.width + (last.bbox.width - prev.bbox.width) * k
          bh = last.bbox.height + (last.bbox.height - prev.bbox.height) * k
        }
        const screenX = offX + x * sx
        const screenY = offY + y * sy
        const screenW = bw * sx
        const screenH = bh * sy
        const known = last.match_name !== null
        ctx.lineWidth = 3
        ctx.strokeStyle = known ? '#10b981' : '#f59e0b'
        ctx.strokeRect(screenX, screenY, screenW, screenH)
        const label = known
          ? `${last.match_name} (${(last.similarity * 100).toFixed(0)}%)`
          : 'Belum dikenal'
        ctx.font = '600 14px system-ui, sans-serif'
        const metrics = ctx.measureText(label)
        const padX = 8
        const padY = 4
        const labelW = metrics.width + padX * 2
        const labelH = 22
        const labelY = screenY - labelH < offY + 4 ? screenY + screenH + 4 : screenY - labelH
        ctx.fillStyle = known ? '#10b981' : '#f59e0b'
        ctx.fillRect(screenX, labelY, labelW, labelH)
        ctx.fillStyle = '#0f172a'
        ctx.fillText(label, screenX + padX, labelY + labelH - padY - 2)
      }
      rafRef.current = requestAnimationFrame(render)
    }
    rafRef.current = requestAnimationFrame(render)
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
  }, [])

  const knownFaces = faces.filter((f) => f.match_name !== null)
  const unknownCount = faces.length - knownFaces.length

  return (
    <div className="relative w-full h-full bg-slate-900 flex flex-col">
      <div className="flex items-center gap-2 px-3 py-2 bg-slate-900/90 border-b border-slate-700">
        <label className="text-xs text-slate-300">Kamera:</label>
        <select
          value={selectedDeviceId}
          onChange={(e) => onPickCamera(e.target.value)}
          disabled={devices.length === 0}
          className="flex-1 max-w-xs rounded-md bg-slate-800 text-slate-100 text-xs px-2 py-1 border border-slate-700 focus:outline-none focus:ring-1 focus:ring-indigo-400 disabled:opacity-50"
        >
          {devices.length === 0 && <option value="">(belum ada kamera)</option>}
          {devices.map((d) => (
            <option key={d.deviceId} value={d.deviceId}>
              {d.label}
            </option>
          ))}
        </select>
        <button
          type="button"
          onClick={() => void init()}
          title="Refresh daftar kamera"
          className="text-xs text-slate-300 bg-slate-800 border border-slate-700 rounded px-2 py-1 hover:bg-slate-700"
        >
          ↻
        </button>
        <div className="text-xs text-slate-400 ml-auto flex items-center gap-2">
          <span
            className={`inline-block h-2 w-2 rounded-full ${
              streamActive ? 'bg-emerald-500' : 'bg-slate-500'
            } ${recognizing ? 'animate-pulse' : ''}`}
          />
          {lastLatencyMs !== null && <span>{lastLatencyMs}ms</span>}
        </div>
      </div>

      <div className="relative flex-1 min-h-0 bg-black">
        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-contain"
          playsInline
          muted
          autoPlay
        />
        <canvas ref={overlayRef} className="absolute inset-0 pointer-events-none" />
        <canvas ref={captureRef} className="hidden" />

        {errorMsg && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/85 px-4">
            <div className="max-w-md text-center bg-red-900/60 border border-red-700 rounded-lg p-4">
              <p className="text-sm text-red-100">{errorMsg}</p>
              <button
                onClick={() => void init()}
                className="mt-3 text-xs bg-red-800 hover:bg-red-700 text-white rounded px-3 py-1"
              >
                Coba lagi
              </button>
            </div>
          </div>
        )}

        {awaitingName && !errorMsg && (
          <div className="absolute top-3 left-1/2 -translate-x-1/2 px-3 py-1.5 rounded-full bg-amber-500/90 backdrop-blur-sm text-xs font-medium text-amber-50 border border-amber-300 shadow-lg animate-pulse">
            Sebutkan nama kamu (ketik atau tahan mic)
          </div>
        )}

        {!errorMsg && !streamActive && (
          <div className="absolute inset-0 flex items-center justify-center bg-slate-900/60 pointer-events-none">
            <p className="text-sm text-slate-300 animate-pulse">{statusMsg}</p>
          </div>
        )}
      </div>

      <div className="px-3 py-2 bg-slate-900 border-t border-slate-700 text-xs text-slate-300 flex items-center gap-3">
        {faces.length === 0 ? (
          <span className="text-slate-500">{streamActive ? 'Belum ada wajah terdeteksi' : statusMsg}</span>
        ) : (
          <>
            <span>
              <span className="text-emerald-400 font-semibold">{knownFaces.length}</span> dikenal
              {knownFaces.length > 0 && (
                <span className="text-slate-400">
                  {' '}({knownFaces.map((f) => f.match_name).join(', ')})
                </span>
              )}
            </span>
            {unknownCount > 0 && (
              <span>
                <span className="text-amber-400 font-semibold">{unknownCount}</span> baru
              </span>
            )}
          </>
        )}
      </div>
    </div>
  )
}
