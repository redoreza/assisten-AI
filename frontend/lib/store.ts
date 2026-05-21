import { create } from 'zustand'

import { AudioQueue } from './audioQueue'
import type { ChatMessage, ConnectionStatus, ServerMsg, VisemeEvent } from './types'
import { blobToBase64, connectWs, type WsHandle } from './websocket'

const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? 'ws://127.0.0.1:8000/ws'

/** Force-clear `assistantBusy` if no terminal `done`/`error` ever arrives, so a
 *  dropped turn can't disable the mic permanently. */
const BUSY_TIMEOUT_MS = 30_000

interface Timing {
  first_token_ms?: number | null
  first_audio_ms?: number | null
  total_ms?: number
}

interface ChatState {
  status: ConnectionStatus
  personaId: string
  messages: ChatMessage[]
  latestTiming: Timing | null
  ready: boolean

  /** True after server emits face_awaiting_name; UI uses this to hint
   *  that the next user input will be interpreted as a name. */
  awaitingName: boolean
  /** True while any face is currently visible in the camera. Used by VAD
   *  listener to gate the always-on mic. */
  facePresent: boolean
  /** Updated whenever the audio queue starts playing a chunk and reset
   *  shortly after the last chunk ends — VAD listener pauses on this to
   *  avoid hearing Pointer's own TTS through the speakers. */
  assistantSpeaking: boolean
  /** True from the moment a turn is sent (text/audio) until the server emits
   *  `done` or `error`. While true the VAD mic is gated off so one utterance
   *  cannot trigger a second backend turn (which made Pointer answer twice). */
  assistantBusy: boolean

  connect: () => void
  disconnect: () => void
  sendText: (message: string) => void
  sendAudio: (audioBlob: Blob, format: 'webm' | 'mp3' | 'wav') => Promise<void>
  setPersona: (personaId: string) => void
  clearHistory: () => void
  sendFacePresent: (params: {
    match_name: string | null
    match_person_id: number | null
    similarity: number
    image_base64: string
  }) => void
  sendFaceLost: () => void
  setFacePresent: (present: boolean) => void
}

let ws: WsHandle | null = null
let audioQ: AudioQueue | null = null
// Bumped on every connect()/disconnect() so callbacks from a stale socket
// (StrictMode double-mount, reconnect) are ignored instead of clobbering the
// live connection.
let wsGeneration = 0
// Timer handle for the assistantBusy safety timeout.
let busyTimer: number | null = null
const assistantMsgRef = { current: null as string | null, finalApplied: false }
// Deferred drop of assistantSpeaking — if a new chunk starts within 300 ms of
// the previous one ending we cancel the drop so VAD doesn't flicker.
let speakingResetTimer: number | null = null
// Sources for the next assistant message — they arrive BEFORE the streaming
// text starts, so we buffer them and attach to the next created bubble.
let pendingSources: {
  query: string
  sources: { title: string; url: string; snippet: string }[]
} | null = null

// Module-level viseme buffer keyed by audio_seq — not Zustand state
// (subscribers don't need re-render when this mutates).
const visemeBuffer = new Map<number, VisemeEvent[]>()

// Module-level pub/sub for audio playback start events. Avatar3D subscribes.
type AudioStartListener = (audio: HTMLAudioElement, visemes: VisemeEvent[]) => void
const audioStartListeners = new Set<AudioStartListener>()
type AudioEndListener = (sequence: number) => void
const audioEndListeners = new Set<AudioEndListener>()

export function subscribeAudioStart(cb: AudioStartListener): () => void {
  audioStartListeners.add(cb)
  return () => audioStartListeners.delete(cb)
}

export function subscribeAudioEnd(cb: AudioEndListener): () => void {
  audioEndListeners.add(cb)
  return () => audioEndListeners.delete(cb)
}

function ensureAudioQueue(): AudioQueue {
  if (!audioQ) audioQ = new AudioQueue()
  return audioQ
}

/** Toggle the "turn in flight" flag, arming a safety timeout when set so the
 *  mic is never gated off forever if a `done`/`error` event goes missing. */
function setBusy(
  set: (partial: Partial<ChatState>) => void,
  value: boolean
): void {
  if (busyTimer !== null) {
    window.clearTimeout(busyTimer)
    busyTimer = null
  }
  if (value) {
    busyTimer = window.setTimeout(() => {
      busyTimer = null
      set({ assistantBusy: false })
    }, BUSY_TIMEOUT_MS)
  }
  set({ assistantBusy: value })
}

function newId(): string {
  return `m_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`
}

export const useChatStore = create<ChatState>((set, get) => ({
  status: 'idle',
  personaId: 'pointer',
  messages: [],
  latestTiming: null,
  ready: false,
  awaitingName: false,
  facePresent: false,
  assistantSpeaking: false,
  assistantBusy: false,

  connect: () => {
    if (ws) {
      ws.close()
      ws = null
    }
    // Each connection gets a generation id; callbacks check it so a late event
    // from a previous socket can't mutate state for the current one.
    const gen = ++wsGeneration
    setBusy(set, false)
    set({ status: 'connecting', ready: false })
    ws = connectWs(WS_URL, {
      onOpen: () => {
        if (gen === wsGeneration) set({ status: 'open' })
      },
      onMessage: (m: ServerMsg) => {
        if (gen === wsGeneration) handleServerMsg(m, set, get)
      },
      onError: () => {
        if (gen === wsGeneration) set({ status: 'error' })
      },
      onClose: () => {
        if (gen !== wsGeneration) return // stale socket — ignore
        set({ status: 'closed', ready: false })
        ws = null
      },
    })
  },

  disconnect: () => {
    // Invalidate the current socket's callbacks before closing it.
    wsGeneration++
    ws?.close()
    ws = null
    audioQ?.stop()
    visemeBuffer.clear()
    setBusy(set, false)
    set({ status: 'closed', ready: false })
  },

  sendText: (message: string) => {
    if (!ws) return
    if (get().assistantBusy) {
      console.warn('sendText ignored — a turn is already in progress')
      return
    }
    assistantMsgRef.current = null
    assistantMsgRef.finalApplied = false
    set((s) => ({
      messages: [...s.messages, { id: newId(), role: 'user', text: message }],
      latestTiming: null,
    }))
    setBusy(set, true)
    ws.send({ type: 'text', message })
  },

  sendAudio: async (audioBlob: Blob, format: 'webm' | 'mp3' | 'wav') => {
    if (!ws) return
    // Drop the second of two near-simultaneous triggers — the VAD can re-fire
    // on a pause mid-utterance, which previously made Pointer answer twice.
    if (get().assistantBusy) {
      console.warn('sendAudio ignored — a turn is already in progress')
      return
    }
    const data = await blobToBase64(audioBlob)
    // Re-check: another onSpeechEnd may have won the race during the encode.
    if (get().assistantBusy) {
      console.warn('sendAudio ignored after encode — turn already in progress')
      return
    }
    assistantMsgRef.current = null
    assistantMsgRef.finalApplied = false
    const placeholderId = newId()
    set((s) => ({
      messages: [
        ...s.messages,
        { id: placeholderId, role: 'user', text: '...', pending: true },
      ],
      latestTiming: null,
    }))
    setBusy(set, true)
    ws.send({ type: 'audio_chunk', data, format })
    ws.send({ type: 'audio_end' })
  },

  setPersona: (personaId: string) => {
    set({ personaId })
    ws?.send({ type: 'set_persona', persona_id: personaId })
  },

  sendFacePresent: (params) => {
    if (!ws) return
    ws.send({
      type: 'face_present',
      match_name: params.match_name,
      match_person_id: params.match_person_id,
      similarity: params.similarity,
      image_base64: params.image_base64,
    })
  },

  sendFaceLost: () => {
    if (!ws) return
    ws.send({ type: 'face_lost' })
  },

  setFacePresent: (present: boolean) => {
    set({ facePresent: present })
  },

  clearHistory: () => {
    set({ messages: [], latestTiming: null })
    visemeBuffer.clear()
    setBusy(set, false)
    ws?.send({ type: 'clear_history' })
    audioQ?.stop()
  },
}))

function handleServerMsg(
  m: ServerMsg,
  set: (
    partial: Partial<ChatState> | ((s: ChatState) => Partial<ChatState>)
  ) => void,
  get: () => ChatState
): void {
  switch (m.type) {
    case 'ready':
      set({ personaId: m.persona_id, ready: true })
      break

    case 'ack':
      break

    case 'transcript':
      set((s) => {
        const msgs = [...s.messages]
        const last = msgs.at(-1)
        if (last && last.role === 'user' && last.pending) {
          msgs[msgs.length - 1] = { ...last, text: m.text, pending: false }
        } else {
          msgs.push({ id: newId(), role: 'user', text: m.text })
        }
        return { messages: msgs }
      })
      break

    case 'ai_text': {
      if (m.is_final) {
        set((s) => {
          if (!assistantMsgRef.current) {
            const sources = pendingSources?.sources
            const searchQuery = pendingSources?.query
            pendingSources = null
            return {
              messages: [
                ...s.messages,
                {
                  id: newId(),
                  role: 'assistant',
                  text: m.text,
                  ...(sources ? { sources, searchQuery } : {}),
                },
              ],
            }
          }
          const id = assistantMsgRef.current
          assistantMsgRef.finalApplied = true
          return {
            messages: s.messages.map((msg) =>
              msg.id === id ? { ...msg, text: m.text, pending: false } : msg
            ),
          }
        })
        break
      }
      if (!assistantMsgRef.current) {
        const id = newId()
        assistantMsgRef.current = id
        // Attach any buffered sources to this new bubble
        const sources = pendingSources?.sources
        const searchQuery = pendingSources?.query
        pendingSources = null
        set((s) => ({
          messages: [
            ...s.messages,
            {
              id,
              role: 'assistant',
              text: m.text,
              pending: true,
              ...(sources ? { sources, searchQuery } : {}),
            },
          ],
        }))
      } else if (!assistantMsgRef.finalApplied) {
        const id = assistantMsgRef.current
        set((s) => ({
          messages: s.messages.map((msg) =>
            msg.id === id
              ? { ...msg, text: `${msg.text} ${m.text}`.trim() }
              : msg
          ),
        }))
      }
      break
    }

    case 'viseme':
      visemeBuffer.set(m.audio_seq, m.events)
      break

    case 'audio': {
      const seq = m.sequence
      ensureAudioQueue().enqueueBase64Mp3(m.data, seq, {
        onStart: (sequence, audioEl) => {
          // Cancel any pending "stop speaking" timer — a new chunk just started
          if (speakingResetTimer !== null) {
            window.clearTimeout(speakingResetTimer)
            speakingResetTimer = null
          }
          set({ assistantSpeaking: true })
          const visemes = visemeBuffer.get(sequence) ?? []
          for (const cb of audioStartListeners) {
            try {
              cb(audioEl, visemes)
            } catch (e) {
              console.error('audioStart listener threw', e)
            }
          }
        },
        onEnd: (sequence) => {
          visemeBuffer.delete(sequence)
          // Defer the reset by 300 ms so back-to-back chunks (typical for
          // sentence-streaming TTS) don't flicker assistantSpeaking off/on.
          // onStart of the next chunk will cancel this timer.
          if (speakingResetTimer !== null) {
            window.clearTimeout(speakingResetTimer)
          }
          speakingResetTimer = window.setTimeout(() => {
            speakingResetTimer = null
            set({ assistantSpeaking: false })
          }, 300)
          for (const cb of audioEndListeners) {
            try {
              cb(sequence)
            } catch (e) {
              console.error('audioEnd listener threw', e)
            }
          }
        },
      })
      break
    }

    case 'timing':
      set({
        latestTiming: {
          first_token_ms: m.first_token_ms,
          first_audio_ms: m.first_audio_ms,
          total_ms: m.total_ms,
        },
      })
      break

    case 'done':
      setBusy(set, false)
      if (assistantMsgRef.current && !assistantMsgRef.finalApplied) {
        const id = assistantMsgRef.current
        set((s) => ({
          messages: s.messages.map((msg) =>
            msg.id === id ? { ...msg, pending: false } : msg
          ),
        }))
      }
      break

    case 'error':
      setBusy(set, false)
      set((s) => ({
        messages: [
          ...s.messages,
          {
            id: newId(),
            role: 'assistant',
            text: `⚠ ${m.message}`,
          },
        ],
      }))
      break

    case 'tool_use':
      // Buffer query so the next assistant bubble shows what was searched.
      // Sources arrive separately via `search_sources`.
      pendingSources = { query: m.query ?? '', sources: [] }
      break

    case 'search_sources':
      pendingSources = { query: m.query, sources: m.sources }
      break

    case 'face_awaiting_name':
      set({ awaitingName: true })
      break

    case 'face_enrolled':
      set((s) => ({
        awaitingName: false,
        messages: [
          ...s.messages,
          {
            id: newId(),
            role: 'assistant',
            text: `✓ Tersimpan sebagai "${m.name}"`,
          },
        ],
      }))
      break
  }
}
