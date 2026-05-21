'use client'

import { useCallback, useEffect, useRef, useState } from 'react'

import { useChatStore } from '@/lib/store'

type RecorderState = 'idle' | 'requesting' | 'ready' | 'recording' | 'sending' | 'error'

const PREFERRED_MIME = 'audio/webm;codecs=opus'

function pickMimeType(): string {
  if (typeof MediaRecorder === 'undefined') return ''
  if (MediaRecorder.isTypeSupported(PREFERRED_MIME)) return PREFERRED_MIME
  if (MediaRecorder.isTypeSupported('audio/webm')) return 'audio/webm'
  if (MediaRecorder.isTypeSupported('audio/mp4')) return 'audio/mp4'
  return ''
}

export function VoiceRecorder() {
  const [state, setState] = useState<RecorderState>('idle')
  const [errorMsg, setErrorMsg] = useState<string | null>(null)
  const [recordingMs, setRecordingMs] = useState<number>(0)

  const streamRef = useRef<MediaStream | null>(null)
  const recorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const tickRef = useRef<number | null>(null)
  const startedAtRef = useRef<number>(0)

  const sendAudio = useChatStore((s) => s.sendAudio)
  const wsReady = useChatStore((s) => s.ready)

  const stopStream = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    recorderRef.current = null
    if (tickRef.current !== null) {
      window.clearInterval(tickRef.current)
      tickRef.current = null
    }
  }, [])

  useEffect(() => () => stopStream(), [stopStream])

  const start = useCallback(async () => {
    if (state === 'recording' || state === 'sending') return
    setErrorMsg(null)
    setState('requesting')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const mime = pickMimeType()
      const rec = mime
        ? new MediaRecorder(stream, { mimeType: mime })
        : new MediaRecorder(stream)
      recorderRef.current = rec
      chunksRef.current = []
      rec.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data)
      }
      rec.onstop = async () => {
        try {
          const blob = new Blob(chunksRef.current, {
            type: rec.mimeType || 'audio/webm',
          })
          chunksRef.current = []
          setState('sending')
          const format = (rec.mimeType || 'audio/webm').includes('mp4') ? 'mp3' : 'webm'
          await sendAudio(blob, format as 'webm' | 'mp3')
          setState('ready')
        } catch (e) {
          console.error('send audio failed', e)
          setErrorMsg('Gagal mengirim audio')
          setState('error')
        } finally {
          stopStream()
          setState((s) => (s === 'sending' ? 'idle' : s))
        }
      }
      rec.start()
      startedAtRef.current = performance.now()
      setRecordingMs(0)
      tickRef.current = window.setInterval(() => {
        setRecordingMs(Math.round(performance.now() - startedAtRef.current))
      }, 100)
      setState('recording')
    } catch (e) {
      console.error('mic access failed', e)
      setErrorMsg(
        'Izin mikrofon ditolak atau perangkat tidak tersedia. Periksa pengaturan browser.'
      )
      setState('error')
      stopStream()
    }
  }, [sendAudio, state, stopStream])

  const stop = useCallback(() => {
    if (state !== 'recording') return
    if (tickRef.current !== null) {
      window.clearInterval(tickRef.current)
      tickRef.current = null
    }
    try {
      recorderRef.current?.stop()
    } catch (e) {
      console.error('stop failed', e)
      stopStream()
      setState('idle')
    }
  }, [state, stopStream])

  const buttonLabel =
    state === 'recording'
      ? `Lepas untuk kirim (${(recordingMs / 1000).toFixed(1)}s)`
      : state === 'sending'
        ? 'Mengirim...'
        : state === 'requesting'
          ? 'Izin mikrofon...'
          : 'Tahan untuk bicara'

  const disabled =
    !wsReady || state === 'requesting' || state === 'sending'

  return (
    <div className="flex flex-col items-center gap-2">
      <button
        type="button"
        onPointerDown={start}
        onPointerUp={stop}
        onPointerLeave={stop}
        onPointerCancel={stop}
        disabled={disabled}
        className={[
          'select-none rounded-full px-8 py-4 text-base font-semibold transition-all',
          'shadow-lg outline-offset-4 focus-visible:outline-2',
          state === 'recording'
            ? 'bg-red-500 text-white scale-105 shadow-red-500/40'
            : 'bg-slate-900 text-white hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed',
        ].join(' ')}
        aria-pressed={state === 'recording'}
      >
        <span className="inline-flex items-center gap-2">
          <span
            aria-hidden
            className={[
              'inline-block h-3 w-3 rounded-full',
              state === 'recording' ? 'bg-white animate-pulse' : 'bg-slate-400',
            ].join(' ')}
          />
          {buttonLabel}
        </span>
      </button>
      {errorMsg && (
        <p role="alert" className="text-sm text-red-600 max-w-xs text-center">
          {errorMsg}
        </p>
      )}
      {!wsReady && (
        <p className="text-xs text-slate-500">Menghubungkan ke backend...</p>
      )}
    </div>
  )
}
