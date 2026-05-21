/** Encode a Float32 PCM audio buffer (range -1..1) into a 16-bit mono WAV byte
 *  array. Silero VAD emits 16 kHz audio in Float32, but our backend STT
 *  (Groq Whisper) accepts WAV happily, so we wrap with a minimal header. */

export function encodeWav(
  samples: Float32Array,
  sampleRate: number
): Uint8Array<ArrayBuffer> {
  const numSamples = samples.length
  const bytesPerSample = 2
  const blockAlign = bytesPerSample // mono
  const byteRate = sampleRate * blockAlign
  const dataSize = numSamples * bytesPerSample
  const buffer = new ArrayBuffer(44 + dataSize)
  const view = new DataView(buffer)

  // RIFF header
  writeString(view, 0, 'RIFF')
  view.setUint32(4, 36 + dataSize, true) // file size - 8
  writeString(view, 8, 'WAVE')

  // fmt chunk
  writeString(view, 12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM = 1
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, byteRate, true)
  view.setUint16(32, blockAlign, true)
  view.setUint16(34, 16, true) // bits per sample

  // data chunk
  writeString(view, 36, 'data')
  view.setUint32(40, dataSize, true)

  // PCM samples (Float32 -1..1 → Int16 -32768..32767)
  let offset = 44
  for (let i = 0; i < numSamples; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
    offset += 2
  }
  return new Uint8Array(buffer) as Uint8Array<ArrayBuffer>
}

function writeString(view: DataView, offset: number, s: string): void {
  for (let i = 0; i < s.length; i++) {
    view.setUint8(offset + i, s.charCodeAt(i))
  }
}

/** Same WAV bytes encoded as base64 — handy for sending in JSON over WS. */
export function encodeWavBase64(samples: Float32Array, sampleRate: number): string {
  const bytes = encodeWav(samples, sampleRate)
  let bin = ''
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    bin += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }
  return btoa(bin)
}
