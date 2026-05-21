'use client'

import { useEffect, useRef, useState } from 'react'

import { useChatStore } from '@/lib/store'

/** How many recent turns (user+assistant pairs are 2 messages each) to keep
 *  visible. Older messages are hidden behind a toggle so the page doesn't
 *  scroll forever. */
const VISIBLE_MESSAGES = 4

export function ChatInterface() {
  const messages = useChatStore((s) => s.messages)
  const status = useChatStore((s) => s.status)
  const ready = useChatStore((s) => s.ready)
  const personaId = useChatStore((s) => s.personaId)
  const timing = useChatStore((s) => s.latestTiming)
  const sendText = useChatStore((s) => s.sendText)
  const clearHistory = useChatStore((s) => s.clearHistory)
  const assistantBusy = useChatStore((s) => s.assistantBusy)

  // Avoid hydration mismatch: Zustand state can drift between the SSR snapshot
  // and the client (HMR keeps the store alive while the server re-renders from
  // fresh initial state). We treat anything derived from the store as "not
  // ready yet" until the component has mounted on the client.
  const [hydrated, setHydrated] = useState(false)
  useEffect(() => {
    setHydrated(true)
  }, [])
  const isReady = hydrated && ready
  // Block typed input while a turn is in flight — one turn at a time, and the
  // disabled state makes that visible instead of silently dropping the message.
  const busy = hydrated && assistantBusy

  const [draft, setDraft] = useState('')
  const [showAll, setShowAll] = useState(false)
  const listRef = useRef<HTMLDivElement>(null)

  const visible = showAll
    ? messages
    : messages.slice(Math.max(0, messages.length - VISIBLE_MESSAGES))
  const hiddenCount = messages.length - visible.length

  useEffect(() => {
    // Scroll to bottom on new messages so the latest is in view
    listRef.current?.scrollTo({
      top: listRef.current.scrollHeight,
      behavior: 'smooth',
    })
  }, [visible.length])

  const submit = (e: React.FormEvent) => {
    e.preventDefault()
    const text = draft.trim()
    if (!text || !isReady || busy) return
    sendText(text)
    setDraft('')
  }

  const displayName =
    personaId.charAt(0).toUpperCase() + personaId.slice(1)

  return (
    <section className="flex flex-col flex-1 min-h-0 w-full max-w-2xl mx-auto px-4">
      <header className="shrink-0 flex items-center justify-between py-2 border-b border-slate-200">
        <div className="flex items-center gap-2 min-w-0">
          <StatusDot status={hydrated ? status : 'idle'} />
          <h2 className="text-sm font-semibold truncate">{displayName}</h2>
          {hydrated && timing?.first_audio_ms != null && (
            <span className="text-[11px] text-slate-500 truncate">
              {timing.first_audio_ms}ms / {timing.total_ms}ms
            </span>
          )}
        </div>
        <div className="flex items-center gap-3 shrink-0">
          {hiddenCount > 0 && (
            <button
              type="button"
              onClick={() => setShowAll((v) => !v)}
              className="text-[11px] text-slate-500 hover:text-slate-900 underline-offset-4 hover:underline"
            >
              {showAll ? 'Sembunyikan' : `+${hiddenCount} pesan`}
            </button>
          )}
          <button
            type="button"
            onClick={clearHistory}
            className="text-[11px] text-slate-500 hover:text-slate-900 underline-offset-4 hover:underline"
          >
            Reset
          </button>
        </div>
      </header>

      {/* Message list — bounded height, scrolls internally instead of growing the page */}
      <div
        ref={listRef}
        className="flex-1 min-h-0 overflow-y-auto py-3 space-y-2"
        aria-live="polite"
      >
        {visible.length === 0 && (
          <p className="text-sm text-slate-400 italic text-center mt-4">
            Hadap kamera lalu bicara — atau ketik di bawah.
          </p>
        )}
        {visible.map((m) => (
          <div
            key={m.id}
            className={[
              'flex flex-col gap-1',
              m.role === 'user' ? 'items-end' : 'items-start',
            ].join(' ')}
          >
            <div
              className={[
                'max-w-[85%] rounded-2xl px-3.5 py-2 text-sm whitespace-pre-wrap break-words',
                m.role === 'user'
                  ? 'bg-slate-900 text-white rounded-br-sm'
                  : 'bg-slate-100 text-slate-900 rounded-bl-sm',
                m.pending ? 'opacity-60 italic' : '',
              ].join(' ')}
            >
              {m.text || (m.pending ? '...' : '')}
            </div>
            {m.sources && m.sources.length > 0 && (
              <SourcesPanel query={m.searchQuery} sources={m.sources} />
            )}
          </div>
        ))}
      </div>

      <form
        onSubmit={submit}
        className="shrink-0 py-2 border-t border-slate-200 flex gap-2"
      >
        <input
          type="text"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder={
            !isReady
              ? 'Menunggu koneksi…'
              : busy
                ? `${displayName} sedang menjawab…`
                : `Ketik untuk ${displayName}…`
          }
          disabled={!isReady || busy}
          className="flex-1 rounded-md border border-slate-300 px-3 py-1.5 text-sm
                     focus:outline-none focus:ring-2 focus:ring-slate-400
                     disabled:bg-slate-50 disabled:text-slate-400"
        />
        <button
          type="submit"
          disabled={!isReady || !draft.trim() || busy}
          className="rounded-md bg-slate-900 text-white px-3.5 py-1.5 text-sm font-medium
                     hover:bg-slate-800 disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Kirim
        </button>
      </form>
    </section>
  )
}

function SourcesPanel({
  query,
  sources,
}: {
  query?: string
  sources: { title: string; url: string; snippet: string }[]
}) {
  const [open, setOpen] = useState(false)
  if (sources.length === 0) return null
  return (
    <div className="max-w-[85%] text-[11px] text-slate-500 ml-1">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1 hover:text-slate-900"
      >
        <span aria-hidden>🔍</span>
        <span className="underline-offset-2 hover:underline">
          {open ? 'Sembunyikan' : `${sources.length} sumber`}
          {query ? ` · "${query}"` : ''}
        </span>
      </button>
      {open && (
        <ul className="mt-1 space-y-1.5 pl-1 border-l-2 border-slate-200">
          {sources.map((s, i) => (
            <li key={i} className="pl-2">
              <a
                href={s.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-slate-700 hover:text-slate-900 underline-offset-2 hover:underline font-medium block truncate"
                title={s.url}
              >
                {s.title || s.url}
              </a>
              <p className="text-slate-500 line-clamp-2">{s.snippet}</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}

function StatusDot({ status }: { status: string }) {
  const color =
    status === 'open'
      ? 'bg-emerald-500'
      : status === 'connecting'
        ? 'bg-amber-500 animate-pulse'
        : status === 'error'
          ? 'bg-red-500'
          : 'bg-slate-400'
  return (
    <span
      aria-label={`Status: ${status}`}
      className={['inline-block h-2 w-2 rounded-full', color].join(' ')}
    />
  )
}
