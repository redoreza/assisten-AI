'use client'

import { useEffect, useRef, useState } from 'react'

import { subscribeAudioEnd, subscribeAudioStart, useChatStore } from '@/lib/store'

/**
 * Audio-reactive orb — replaces 3D avatar with a modern voice-assistant style
 * visualizer. The central orb pulses with the amplitude of whatever audio
 * chunk is currently playing; idle state breathes gently so the surface
 * never feels static.
 */
export function OrbVisualizer() {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [speaking, setSpeaking] = useState(false)
  const personaId = useChatStore((s) => s.personaId)
  const displayName =
    personaId.charAt(0).toUpperCase() + personaId.slice(1)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let audioCtx: AudioContext | null = null
    let analyser: AnalyserNode | null = null
    let freqData: Uint8Array<ArrayBuffer> | null = null
    let activeChunks = 0
    let disposed = false
    let raf = 0

    const dpr = () => window.devicePixelRatio || 1

    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      canvas.width = Math.max(1, Math.floor(rect.width * dpr()))
      canvas.height = Math.max(1, Math.floor(rect.height * dpr()))
    }
    resize()
    const ro = new ResizeObserver(resize)
    ro.observe(canvas)

    const ensureAudioGraph = (audioEl: HTMLAudioElement) => {
      if (!audioCtx) {
        const AC =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext })
            .webkitAudioContext
        audioCtx = new AC()
        analyser = audioCtx.createAnalyser()
        analyser.fftSize = 256
        analyser.smoothingTimeConstant = 0.75
        freqData = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount))
        analyser.connect(audioCtx.destination)
      }
      try {
        const src = audioCtx.createMediaElementSource(audioEl)
        src.connect(analyser!)
      } catch (e) {
        // Each HTMLAudioElement can only be wrapped once; the queue creates a
        // fresh element per chunk so this normally succeeds. Log just in case.
        console.warn('[OrbVisualizer] createMediaElementSource failed', e)
      }
      if (audioCtx.state === 'suspended') {
        void audioCtx.resume()
      }
    }

    const unsubStart = subscribeAudioStart((audioEl) => {
      ensureAudioGraph(audioEl)
      activeChunks += 1
      setSpeaking(true)
    })

    const unsubEnd = subscribeAudioEnd(() => {
      activeChunks = Math.max(0, activeChunks - 1)
      if (activeChunks === 0) setSpeaking(false)
    })

    const draw = (now: number) => {
      if (disposed) return
      const w = canvas.width
      const h = canvas.height
      const cx = w / 2
      const cy = h / 2
      const minDim = Math.min(w, h)
      const baseRadius = minDim * 0.18

      let amp = 0
      if (analyser && freqData && activeChunks > 0) {
        analyser.getByteFrequencyData(freqData)
        const bins = Math.floor(freqData.length * 0.6)
        let sum = 0
        for (let i = 0; i < bins; i++) sum += freqData[i]
        amp = sum / bins / 255
      }

      const t = now / 1000
      const idle = (Math.sin(t * 1.2) + 1) * 0.5 * 0.18
      const effect = Math.max(amp, idle)

      ctx.clearRect(0, 0, w, h)
      const bgGrad = ctx.createRadialGradient(cx, cy, minDim * 0.1, cx, cy, minDim * 0.7)
      bgGrad.addColorStop(0, 'rgba(30, 41, 59, 1)')
      bgGrad.addColorStop(1, 'rgba(15, 23, 42, 1)')
      ctx.fillStyle = bgGrad
      ctx.fillRect(0, 0, w, h)

      for (let i = 4; i > 0; i--) {
        const r = baseRadius + effect * minDim * 0.35 + i * minDim * 0.025
        const alpha = (0.16 - i * 0.03) * (0.35 + effect)
        const g = ctx.createRadialGradient(cx, cy, baseRadius * 0.9, cx, cy, r)
        g.addColorStop(0, `rgba(129, 140, 248, ${alpha + 0.05})`)
        g.addColorStop(0.6, `rgba(99, 102, 241, ${alpha * 0.5})`)
        g.addColorStop(1, 'rgba(99, 102, 241, 0)')
        ctx.fillStyle = g
        ctx.beginPath()
        ctx.arc(cx, cy, r, 0, Math.PI * 2)
        ctx.fill()
      }

      if (activeChunks > 0) {
        const ringR = baseRadius * (1.25 + effect * 0.6)
        ctx.beginPath()
        ctx.arc(cx, cy, ringR, 0, Math.PI * 2)
        ctx.lineWidth = Math.max(1.5, minDim * 0.004)
        ctx.strokeStyle = `rgba(165, 180, 252, ${0.55 + effect * 0.3})`
        ctx.stroke()
      }

      const orbR = baseRadius * (1 + effect * 0.25)
      const orbGrad = ctx.createRadialGradient(
        cx - orbR * 0.35,
        cy - orbR * 0.4,
        orbR * 0.05,
        cx,
        cy,
        orbR
      )
      orbGrad.addColorStop(0, '#e0e7ff')
      orbGrad.addColorStop(0.3, '#a5b4fc')
      orbGrad.addColorStop(0.7, '#6366f1')
      orbGrad.addColorStop(1, '#3730a3')
      ctx.fillStyle = orbGrad
      ctx.beginPath()
      ctx.arc(cx, cy, orbR, 0, Math.PI * 2)
      ctx.fill()

      const hlGrad = ctx.createRadialGradient(
        cx - orbR * 0.35,
        cy - orbR * 0.4,
        0,
        cx - orbR * 0.35,
        cy - orbR * 0.4,
        orbR * 0.5
      )
      hlGrad.addColorStop(0, 'rgba(255, 255, 255, 0.55)')
      hlGrad.addColorStop(1, 'rgba(255, 255, 255, 0)')
      ctx.fillStyle = hlGrad
      ctx.beginPath()
      ctx.arc(cx - orbR * 0.35, cy - orbR * 0.4, orbR * 0.5, 0, Math.PI * 2)
      ctx.fill()

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)

    return () => {
      disposed = true
      cancelAnimationFrame(raf)
      ro.disconnect()
      unsubStart()
      unsubEnd()
      if (audioCtx && audioCtx.state !== 'closed') void audioCtx.close()
    }
  }, [])

  return (
    <div className="relative w-full h-full overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full" />
      <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-3 py-1 rounded-full bg-slate-900/60 backdrop-blur-sm text-xs text-slate-200 border border-white/10">
        {speaking ? `${displayName} sedang bicara...` : 'Siap mendengarkan'}
      </div>
    </div>
  )
}
