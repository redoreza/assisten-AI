'use client'

import dynamic from 'next/dynamic'
import { useEffect } from 'react'

import { ChatInterface } from '@/components/ChatInterface'
import { useChatStore } from '@/lib/store'

// CameraView uses getUserMedia + canvas — browser-only. Lazy with SSR disabled.
const CameraView = dynamic(
  () => import('@/components/CameraView').then((m) => m.CameraView),
  { ssr: false, loading: () => <CameraSkeleton /> }
)

// VadListener loads Silero ONNX (~2MB) + audio worklet — browser-only.
const VadListener = dynamic(
  () => import('@/components/VadListener').then((m) => m.VadListener),
  { ssr: false, loading: () => <VadSkeleton /> }
)

function VadSkeleton() {
  return (
    <div className="text-xs text-slate-400 animate-pulse">Memuat mic VAD…</div>
  )
}

function CameraSkeleton() {
  return (
    <div className="w-full h-full flex items-center justify-center bg-slate-900">
      <p className="text-sm text-slate-300 animate-pulse">Memuat kamera...</p>
    </div>
  )
}

export default function Home() {
  const connect = useChatStore((s) => s.connect)
  const disconnect = useChatStore((s) => s.disconnect)

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return (
    // Viewport-bounded: nothing scrolls past the bottom of the window.
    <div className="flex flex-col h-screen overflow-hidden bg-slate-50">
      <header className="shrink-0 px-4 py-2.5 border-b border-slate-200 bg-white">
        <h1 className="text-base font-semibold text-slate-900">Pointer</h1>
        <p className="text-[11px] text-slate-500">Asisten Kampus</p>
      </header>

      <div className="flex flex-col lg:flex-row flex-1 min-h-0">
        {/* Camera: compact 16:9 on desktop, narrower height on mobile */}
        <section className="shrink-0 lg:shrink h-[32vh] lg:h-auto lg:w-[420px] xl:w-[480px] border-b lg:border-b-0 lg:border-r border-slate-200">
          <CameraView />
        </section>

        {/* Chat side: takes remaining space, internal flex shrinks the message list */}
        <section className="flex flex-col flex-1 min-h-0 min-w-0">
          <ChatInterface />
          <div className="shrink-0 py-3 flex justify-center border-t border-slate-200 bg-white">
            <VadListener />
          </div>
        </section>
      </div>
    </div>
  )
}
