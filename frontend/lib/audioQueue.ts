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
  private playing = false
  private allObjectUrls: string[] = []

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

    audio.addEventListener(
      'ended',
      () => {
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
        console.error('Audio playback error for sequence', item.sequence)
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
      console.warn('audio.play() rejected — needs a user gesture first', e)
      URL.revokeObjectURL(item.url)
      this.playing = false
    }
  }

  stop(): void {
    if (this.current) {
      this.current.pause()
      this.current.src = ''
      this.current = null
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
