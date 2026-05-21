/** REST client for /api/face/* — mirrors backend types from app/services/face_recognition.py. */

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://127.0.0.1:8000'

export interface FaceBox {
  x: number
  y: number
  width: number
  height: number
}

export interface FaceMatch {
  bbox: FaceBox
  det_score: number
  match_name: string | null
  match_person_id: number | null
  similarity: number
}

export interface RecognizeResponse {
  faces: FaceMatch[]
  count: number
}

export interface Person {
  person_id: number
  name: string
  embedding_count: number
}

export interface EnrollResponse {
  person_id: number
  name: string
  images_provided: number
  embeddings_added: number
}

async function postJson<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!r.ok) {
    const text = await r.text().catch(() => '')
    throw new Error(`${path} → ${r.status} ${r.statusText}: ${text}`)
  }
  return (await r.json()) as T
}

export async function recognizeFace(
  imageBase64: string,
  threshold?: number
): Promise<RecognizeResponse> {
  return postJson<RecognizeResponse>('/api/face/recognize', {
    image_base64: imageBase64,
    threshold: threshold ?? null,
  })
}

export async function enrollFace(
  name: string,
  imagesBase64: string[]
): Promise<EnrollResponse> {
  return postJson<EnrollResponse>('/api/face/enroll', {
    name,
    images_base64: imagesBase64,
  })
}

export async function listPersons(): Promise<{ persons: Person[]; count: number }> {
  const r = await fetch(`${API_BASE}/api/face/persons`)
  if (!r.ok) throw new Error(`/api/face/persons → ${r.status}`)
  return r.json() as Promise<{ persons: Person[]; count: number }>
}

export async function deletePerson(personId: number): Promise<void> {
  const r = await fetch(`${API_BASE}/api/face/persons/${personId}`, {
    method: 'DELETE',
  })
  if (!r.ok) throw new Error(`delete person ${personId} → ${r.status}`)
}

export async function warmupFace(): Promise<{ ready: boolean; db_stats: unknown }> {
  return postJson('/api/face/warmup', {})
}
