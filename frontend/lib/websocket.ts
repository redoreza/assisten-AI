import type { ClientMsg, ServerMsg } from './types'

export interface WsHandle {
  send: (msg: ClientMsg) => void
  close: () => void
  readyState: () => number
}

export interface WsCallbacks {
  onOpen?: () => void
  onMessage: (msg: ServerMsg) => void
  onError?: (err: Event) => void
  onClose?: (ev: CloseEvent) => void
}

export function connectWs(url: string, cb: WsCallbacks): WsHandle {
  const ws = new WebSocket(url)

  ws.addEventListener('open', () => cb.onOpen?.())
  ws.addEventListener('message', (ev) => {
    try {
      const parsed = JSON.parse(ev.data) as ServerMsg
      cb.onMessage(parsed)
    } catch (e) {
      console.error('Invalid JSON from server', ev.data, e)
    }
  })
  ws.addEventListener('error', (err) => cb.onError?.(err))
  ws.addEventListener('close', (ev) => cb.onClose?.(ev))

  return {
    send(msg) {
      if (ws.readyState !== WebSocket.OPEN) {
        console.warn('WS not open, dropping', msg.type)
        return
      }
      ws.send(JSON.stringify(msg))
    },
    close() {
      if (
        ws.readyState === WebSocket.OPEN ||
        ws.readyState === WebSocket.CONNECTING
      ) {
        ws.close()
      }
    },
    readyState: () => ws.readyState,
  }
}

/** Convert a Blob (recording) to base64 — what the backend WS expects in audio_chunk. */
export async function blobToBase64(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer()
  let binary = ''
  const bytes = new Uint8Array(buf)
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}
