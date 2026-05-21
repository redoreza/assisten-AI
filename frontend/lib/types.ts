/**
 * Mirror of the backend WebSocket protocol (see README.md §6 and
 * backend/app/api/websocket.py + core/orchestrator.py).
 */

export interface VisemeEvent {
  phoneme: string
  time: number
  duration: number
}

export type ClientMsg =
  | { type: 'audio_chunk'; data: string; format: 'webm' | 'mp3' | 'wav' | 'ogg' }
  | { type: 'audio_end' }
  | { type: 'text'; message: string }
  | { type: 'set_persona'; persona_id: string }
  | { type: 'set_mode'; mode: 'companion' | 'customer_service' }
  | { type: 'clear_history' }
  | {
      type: 'face_present'
      match_name: string | null
      match_person_id: number | null
      similarity: number
      image_base64: string
    }
  | { type: 'face_lost' }

export type ServerMsg =
  | { type: 'ready'; persona_id: string; mode: string }
  | { type: 'ack'; field: string; value: string }
  | { type: 'transcript'; text: string; latency_ms: number }
  | { type: 'ai_text'; text: string; is_final: boolean; sentence_idx: number }
  | {
      type: 'audio'
      data: string
      format: 'mp3'
      sequence: number
      sentence_idx: number
      tts_ms: number
    }
  | { type: 'viseme'; events: VisemeEvent[]; audio_seq: number }
  | {
      type: 'timing'
      first_token_ms: number | null
      first_audio_ms: number | null
      total_ms: number
      sentences: number
    }
  | { type: 'done'; reason?: string }
  | { type: 'error'; message: string }
  | { type: 'face_awaiting_name' }
  | { type: 'face_enrolled'; person_id: number; name: string }
  | { type: 'tool_use'; tools: string[]; query?: string }
  | {
      type: 'search_sources'
      query: string
      sources: { title: string; url: string; snippet: string }[]
    }

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  text: string
  pending?: boolean
  timing?: {
    first_token_ms?: number | null
    first_audio_ms?: number | null
    total_ms?: number
  }
  /** Sources Pointer used to compose this reply (from web search). */
  sources?: { title: string; url: string; snippet: string }[]
  /** Query Pointer searched with (when search was triggered). */
  searchQuery?: string
}

export type ConnectionStatus = 'idle' | 'connecting' | 'open' | 'closed' | 'error'
