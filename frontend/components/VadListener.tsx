'use client'

import { MicVAD } from '@ricky0123/vad-web'
import { useCallback, useEffect, useRef, useState } from 'react'

import { useChatStore } from '@/lib/store'
import { encodeWav } from '@/lib/wav'

/** Silero VAD emits audio at 16 kHz mono Float32. */
const VAD_SAMPLE_RATE = 16000
/** Wait this long after losing face before pausing VAD — survives brief turns. */
const FACE_GRACE_MS = 5000

type VadState =
  | 'idle'
  | 'loading'
  | 'listening'
  | 'paused'
  | 'speaking'
  | 'sending'
  | 'error'

export function VadListener() {
  const [state, setState] = useState<VadState>('idle')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)

  const vadRef = useRef<MicVAD | null>(null)

  const sendAudio = useChatStore((s) => s.sendAudio)
  const wsReady = useChatStore((s) => s.ready)
  const facePresent = useChatStore((s) => s.facePresent)
  const assistantSpeaking = useChatStore((s) => s.assistantSpeaking)
  const assistantBusy = useChatStore((s) => s.assistantBusy)
  const awaitingName = useChatStore((s) => s.awaitingName)

  const lastFaceSeenRef = useRef<number>(0)
  useEffect(() => {
    if (facePresent) lastFaceSeenRef.current = performance.now()
  }, [facePresent])

  const shouldListen = useCallback((): boolean => {
    if (!wsReady) return false
    // Keep the mic off both while Pointer is speaking AND while the backend is
    // still composing the reply. Without the assistantBusy check the VAD could
    // re-fire during the think window and Pointer would answer twice.
    if (assistantSpeaking || assistantBusy) return false
    if (facePresent) return true
    if (
      lastFaceSeenRef.current > 0 &&
      performance.now() - lastFaceSeenRef.current < FACE_GRACE_MS
    ) {
      return true
    }
    if (awaitingName) return true
    return false
  }, [wsReady, assistantSpeaking, assistantBusy, facePresent, awaitingName])

  useEffect(() => {
    let disposed = false
    let localVad: MicVAD | null = null
    setState('loading')
    setErrorMsg(null)
    console.info('[VAD] init starting')

    ;(async () => {
      try {
        const vad = await MicVAD.new({
          baseAssetPath: '/',
          onnxWASMBasePath: '/',
          // ── VAD tuning ──────────────────────────────────────────────────
          // Stock settings end a segment after only ~256ms of silence, so a
          // normal sentence with a mid-thought pause splits into two segments.
          // With Fase A the second segment is dropped rather than answered —
          // but that loses half the sentence. Widening the silence hangover
          // keeps a paused sentence as ONE utterance.
          model: 'v5',
          redemptionMs: 640, // silence to wait out before a segment is "done"
          minSpeechMs: 200, // ignore shorter blips (clicks, coughs, echo)
          preSpeechPadMs: 256, // lead-in kept so STT doesn't clip the onset
          positiveSpeechThreshold: 0.6, // less twitchy on background noise
          negativeSpeechThreshold: 0.4,
          onSpeechStart: () => {
            if (disposed) return
            setState('speaking')
          },
          onSpeechEnd: async (audio) => {
            if (disposed) return
            setState('sending')
            const wavBytes = encodeWav(audio, VAD_SAMPLE_RATE)
            const blob = new Blob([wavBytes], { type: 'audio/wav' })
            try {
              await sendAudio(blob, 'wav')
            } catch (e) {
              console.error('[VAD] sendAudio failed', e)
            } finally {
              if (!disposed) setState('listening')
            }
          },
        })
        if (disposed) {
          // Cleanup ran before init finished; throw away the new instance.
          void vad.destroy().catch(() => {})
          return
        }
        localVad = vad
        vadRef.current = vad
        setState('paused') // gating effect below will start() if conditions met
        console.info('[VAD] init complete')
      } catch (e: unknown) {
        if (disposed) return
        const msg = e instanceof Error ? e.message : String(e)
        console.error('[VAD] init failed', e)
        setErrorMsg(`VAD init failed: ${msg}`)
        setState('error')
      }
    })()

    return () => {
      disposed = true
      if (localVad) {
        void localVad.destroy().catch(() => {})
      }
      vadRef.current = null
    }
  }, [sendAudio])

  useEffect(() => {
    const vad = vadRef.current
    if (!vad) return
    if (state === 'error' || state === 'loading' || state === 'idle') return
    const listening = shouldListen()
    if (listening) {
      vad.start()
      setState((s) => (s === 'paused' ? 'listening' : s))
    } else {
      vad.pause()
      setState((s) => (s === 'speaking' || s === 'listening' ? 'paused' : s))
    }
    // Re-tick every 1s — grace-period clock isn't reactive
    const interval = window.setInterval(() => {
      const v = vadRef.current
      if (!v) return
      if (shouldListen()) v.start()
      else v.pause()
    }, 1000)
    return () => window.clearInterval(interval)
  }, [state, shouldListen])

  const label = ((): string => {
    switch (state) {
      case 'loading':
        return 'Memuat model suara…'
      case 'listening':
        return awaitingName ? 'Mendengar — sebutkan namamu' : 'Mendengar'
      case 'speaking':
        return 'Bicara…'
      case 'sending':
        return 'Mengirim…'
      case 'paused':
        if (assistantSpeaking) return 'Pointer sedang bicara'
        if (!facePresent) return 'Tunggu wajah di kamera'
        if (!wsReady) return 'Menyambungkan…'
        return 'Tidak aktif'
      case 'error':
        return errorMsg ?? 'Error'
      default:
        return ''
    }
  })()

  const dotColor = (() => {
    if (state === 'speaking') return 'bg-red-500'
    if (state === 'sending') return 'bg-amber-500'
    if (state === 'listening') return 'bg-emerald-500'
    if (state === 'error') return 'bg-red-700'
    return 'bg-slate-500'
  })()

  return (
    <div className="flex items-center gap-2 px-4 py-2 rounded-full bg-slate-100 border border-slate-300">
      <span
        className={`inline-block h-2.5 w-2.5 rounded-full ${dotColor} ${
          state === 'listening' || state === 'speaking' ? 'animate-pulse' : ''
        }`}
      />
      <span className="text-sm text-slate-700">{label}</span>
    </div>
  )
}
