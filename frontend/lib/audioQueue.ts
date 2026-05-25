/**
 * Gapless MP3 playback queue.
 *
 * Browser MP3 decoding via <audio> element is the simplest portable option
 * (Web Audio API can decode too, but adds extra latency for short clips).
 * We chain clips by listening to `ended` and starting the next one immediately.
 *
 * Order is preserved by enqueueing in the order audio_event arrives. The backend
 * already emits audio events in sentence order, so no reordering needed here.
 */

interface QueueItem {
  url: string
  sequence: number
  onStart?: (sequence: number, audioEl: HTMLAudioElement) => void
  onEnd?: (sequence: number) => void
}

export class AudioQueue {
  private queue: QueueItem[] = []
  private current: HTMLAudioElement | null = null
  private currentItem: QueueItem | null = null
  private playing = false
  private allObjectUrls: string[] = []
  // Tracks elements stopped via stop() so their async error/ended events are
  // suppressed — setting src='' triggers an error event asynchronously, which
  // would cause a double onEnd call and a spurious console.error.
  private _stoppedAudio = new WeakSet<HTMLAudioElement>()

  enqueueBase64Mp3(
    b64: string,
    sequence: number,
    callbacks?: {
      onStart?: (sequence: number, audioEl: HTMLAudioElement) => void
      onEnd?: (sequence: number) => void
    }
  ): void {
    const bin = atob(b64)
    const bytes = new Uint8Array(bin.length)
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
    const blob = new Blob([bytes], { type: 'audio/mpeg' })
    const url = URL.createObjectURL(blob)
    this.allObjectUrls.push(url)
    this.queue.push({ url, sequence, ...callbacks })
    if (!this.playing) {
      void this.playNext()
    }
  }

  private async playNext(): Promise<void> {
    const item = this.queue.shift()
    if (!item) {
      this.playing = false
      return
    }
    this.playing = true
    const audio = new Audio(item.url)
    audio.preload = 'auto'
    this.current = audio
    this.currentItem = item

    audio.addEventListener(
      'ended',
      () => {
        if (this._stoppedAudio.has(audio)) return
        item.onEnd?.(item.sequence)
        URL.revokeObjectURL(item.url)
        this.allObjectUrls = this.allObjectUrls.filter((u) => u !== item.url)
        void this.playNext()
      },
      { once: true }
    )
    audio.addEventListener(
      'error',
      () => {
        if (this._stoppedAudio.has(audio)) return
        const code = audio.error?.code ?? 'unknown'
        console.error(`Audio playback error for sequence ${item.sequence} (MediaError code=${code})`)
        URL.revokeObjectURL(item.url)
        this.allObjectUrls = this.allObjectUrls.filter((u) => u !== item.url)
        item.onEnd?.(item.sequence)
        void this.playNext()
      },
      { once: true }
    )

    try {
      await audio.play()
      item.onStart?.(item.sequence, audio)
    } catch (e) {
      // Suppress the async error event that fires after play() rejection
      // so the queue isn't double-advanced and no spurious console.error appears.
      this._stoppedAudio.add(audio)
      console.warn(`audio.play() failed for sequence ${item.sequence}:`, e)
      URL.revokeObjectURL(item.url)
      this.allObjectUrls = this.allObjectUrls.filter((u) => u !== item.url)
      void this.playNext()
    }
  }

  stop(): void {
    if (this.current) {
      this._stoppedAudio.add(this.current)  // suppress the async error event
      this.current.pause()
      this.current.src = ''
      this.current = null
    }
    // Fire onEnd for the interrupted item so store.ts resets assistantSpeaking.
    if (this.currentItem) {
      this.currentItem.onEnd?.(this.currentItem.sequence)
      this.currentItem = null
    }
    this.queue = []
    this.playing = false
    for (const url of this.allObjectUrls) URL.revokeObjectURL(url)
    this.allObjectUrls = []
  }

  get isPlaying(): boolean {
    return this.playing
  }

  get pendingCount(): number {
    return this.queue.length
  }
}
