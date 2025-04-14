export interface Citation {
  company: string
  ticker: string
  filing: string
  section: string
  page?: number
}

export interface SecResponse {
  answer: string
  citations: Citation[]
  suggested_queries?: string[]
}

export interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  citations?: Citation[]
  suggestedQueries?: string[]
  error?: boolean
  isLoading?: boolean
  loadingDots?: number
  loadingStage?: string
}
