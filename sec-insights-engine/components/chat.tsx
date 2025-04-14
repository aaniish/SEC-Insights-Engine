"use client"

import { useState, useEffect, type FormEvent, type ChangeEvent, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2, Send } from "lucide-react"
import { ChatMessage } from "@/components/chat-message"
import { CompanySelector } from "@/components/company-selector"
import type { Citation, SecResponse } from "@/lib/types"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"

// Define message structure for frontend state
interface Message {
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

interface ChatProps {
  selectedCompanies: string[]
  onCompaniesChange?: (companies: string[]) => void
}

export function Chat({ selectedCompanies, onCompaniesChange }: ChatProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [loadingStage, setLoadingStage] = useState<string>("")
  const [loadingDots, setLoadingDots] = useState<number>(0)
  const [initialSuggestedQueries] = useState<string[]>([
    "What are Apple's main risk factors in their latest 10-K?",
    "How has Microsoft's revenue changed over the last 3 quarters?",
    "What are the key growth initiatives mentioned in Amazon's latest MD&A section?",
    "What is Tesla's strategy for managing supply chain risks?",
  ])

  const messagesEndRef = useRef<HTMLDivElement>(null)
  const scrollTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Improved scroll handling with debounce to avoid ResizeObserver issues
  const scrollToBottom = () => {
    if (scrollTimeoutRef.current) {
      clearTimeout(scrollTimeoutRef.current)
    }

    scrollTimeoutRef.current = setTimeout(() => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: "smooth" })
      }
    }, 100)
  }

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Clean up timeout on unmount
  useEffect(() => {
    return () => {
      if (scrollTimeoutRef.current) {
        clearTimeout(scrollTimeoutRef.current)
      }
    }
  }, [])

  // Loading animation for dots
  useEffect(() => {
    if (!isLoading) return

    const timer = setInterval(() => {
      setLoadingDots((prev) => (prev + 1) % 4)
    }, 500)

    return () => clearInterval(timer)
  }, [isLoading])

  // Loading stages for complex queries
  useEffect(() => {
    if (!isLoading) {
      setLoadingStage("")
      return
    }

    const stages = [
      "Retrieving relevant SEC filings",
      "Analyzing financial statements",
      "Extracting insights from MD&A sections",
      "Evaluating risk factors and disclosures",
      "Synthesizing information from multiple sources",
      "Preparing comprehensive response",
    ]

    let stageIndex = 0
    const stageTimer = setInterval(() => {
      setLoadingStage(stages[stageIndex])
      stageIndex = (stageIndex + 1) % stages.length

      // Update the loading message if it exists
      setMessages((prevMessages) => {
        const loadingMsgIndex = prevMessages.findIndex((msg) => msg.isLoading)
        if (loadingMsgIndex === -1) return prevMessages

        const updatedMessages = [...prevMessages]
        updatedMessages[loadingMsgIndex] = {
          ...updatedMessages[loadingMsgIndex],
          loadingStage: stages[stageIndex],
          loadingDots: loadingDots,
        }
        return updatedMessages
      })
    }, 5000) // Change stage every 5 seconds

    return () => clearInterval(stageTimer)
  }, [isLoading, loadingDots])

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    setInput(e.target.value)
  }

  const handleSuggestedQueryClick = (query: string) => {
    setInput(query)
    // Optional: Immediately submit the query
    // handleSubmit(undefined, query);
  }

  const handleSubmit = async (e?: FormEvent<HTMLFormElement>, queryOverride?: string) => {
    if (e) e.preventDefault()
    const currentQuery = queryOverride || input.trim()
    if (!currentQuery) return

    // Set overall loading state
    setIsLoading(true)
    setLoadingStage("Preparing to search SEC filings...")

    // Add user message immediately
    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content:
        currentQuery + (selectedCompanies.length > 0 ? ` (Companies selected: ${selectedCompanies.join(", ")})` : ""),
    }
    setMessages((prev: Message[]) => [...prev, userMessage])

    // Add a placeholder loading message for the assistant
    const loadingMsgId = crypto.randomUUID()
    const loadingMessage: Message = {
      id: loadingMsgId,
      role: "assistant",
      content: "",
      isLoading: true,
      loadingDots: loadingDots,
      loadingStage: loadingStage,
    }
    setMessages((prev: Message[]) => [...prev, loadingMessage])
    setInput("") // Clear input immediately

    // Prepare history for API (limit length if needed)
    const chatHistory = messages
      .filter((msg) => !msg.isLoading) // Filter out any loading messages
      .map((msg: Message) => ({
        role: msg.role,
        content: msg.content.replace(/\(Companies selected:.*?\)/, '').trim(), // Remove company selection text for API
      }))

    try {
      // Detect if this is likely a complex query that needs the agent
      const isComplexQuery =
        currentQuery.toLowerCase().includes("compare") ||
        currentQuery.toLowerCase().includes("summarize") ||
        currentQuery.toLowerCase().includes("analyze")

      // Set a longer timeout for complex queries
      const timeoutMs = isComplexQuery ? 240000 : 120000 // 4 min or 2 min
      
      // Create the request with timeout
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), timeoutMs)
      
      // Number of retries and backoff configuration
      let retries = 0
      const maxRetries = 2
      const backoffMs = 1000 // Start with 1 second backoff
      
      let response = null
      let error = null
      
      // Retry loop
      while (retries <= maxRetries) {
        try {
          response = await fetch("/api/query", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              query: currentQuery,
              companies: selectedCompanies,
              chat_history: chatHistory,
              stream: isComplexQuery // Opt-in to streaming for complex queries
            }),
            signal: controller.signal,
          })
          
          if (response.ok) break // If successful, break out of retry loop
          
          // If we got a 504 or other error, we'll retry
          error = new Error(`HTTP error! status: ${response.status}`)
          throw error
          
        } catch (err: any) {
          error = err
          
          // Check if it's a network error that might benefit from a retry
          if (err.code === 'ECONNRESET' || err.name === 'AbortError' || 
              (response && (response.status === 504 || response.status === 503))) {
            retries++
            if (retries <= maxRetries) {
              console.log(`Retry ${retries}/${maxRetries} after ECONNRESET or timeout error`)
              // Wait using exponential backoff
              await new Promise(resolve => setTimeout(resolve, backoffMs * retries))
              continue
            }
          }
          
          // For other errors or if we've exhausted retries, throw to be caught by outer catch block
          throw err
        }
      }
      
      // Clear the timeout to prevent memory leaks
      clearTimeout(timeoutId)
      
      if (!response || !response.ok) {
        throw error || new Error("Failed to get a valid response after retries")
      }

      const data: SecResponse = await response.json()

      // Remove the loading message and add the real response
      setMessages((prev: Message[]) =>
        prev
          .filter((msg) => msg.id !== loadingMsgId)
          .concat([
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: data.answer,
              citations: data.citations,
              suggestedQueries: data.suggested_queries,
            },
          ])
      )
    } catch (err: any) {
      console.error("API Error:", err)

      // Check for timeout errors
      const errorMessage =
        err.name === "AbortError"
          ? "Request timed out. This query may be too complex or require too much processing. Try a simpler query or fewer companies."
          : err.message || "An unexpected error occurred retrieving the response."

      // Remove the loading message and add an error message
      setMessages((prev: Message[]) =>
        prev
          .filter((msg) => msg.id !== loadingMsgId)
          .concat([
            {
              id: crypto.randomUUID(),
              role: "assistant",
              content: `Error: ${errorMessage}`,
              error: true,
            },
          ])
      )

      toast.error(errorMessage)
    } finally {
      setIsLoading(false)
    }
  }

  // Helper function to get a friendly company name for display
  const getLabelForCompany = (ticker: string): string => {
    const companyMap: {[key: string]: string} = {
      'AAPL': 'Apple',
      'MSFT': 'Microsoft',
      'GOOGL': 'Google',
      'META': 'Meta',
      'AMZN': 'Amazon',
      'TSLA': 'Tesla'
    };
    return companyMap[ticker] || ticker;
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto space-y-4 pb-4 pt-4 pr-4 md:pr-0">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center px-4">
            <h2 className="text-2xl font-bold mb-2 font-sans">Welcome to SEC Insights Engine</h2>
            <p className="text-muted-foreground mb-6 max-w-md font-serif">
              Ask questions about public company SEC filings to get insights from 10-Ks and 10-Qs.
            </p>
            {selectedCompanies.length > 0 ? (
              <div className="mb-6 text-sm inline-block bg-muted p-3 rounded-md font-serif">
                You've selected <span className="font-medium">{getLabelForCompany(selectedCompanies[0])}</span>. Your queries will focus on this company's filings.
              </div>
            ) : (
              <div className="mb-6 text-sm inline-block bg-muted p-3 rounded-md font-serif">
                No company selected. For more specific results, select a company in the sidebar.
              </div>
            )}
            <div className="grid gap-2 max-w-md mx-auto w-full">
              {initialSuggestedQueries.map((query, i) => (
                <Button
                  key={i}
                  variant="outline"
                  size="sm"
                  className="justify-start text-left h-auto py-2 px-3 whitespace-normal font-serif"
                  onClick={() => handleSuggestedQueryClick(query)}
                >
                  {query}
                </Button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <ChatMessage
              key={message.id}
              message={message}
              onSuggestedQueryClick={handleSuggestedQueryClick}
            />
          ))
        )}
        <div ref={messagesEndRef} /> {/* Element to scroll to */}
      </div>
      <div className="p-4 border-t">
        <form onSubmit={handleSubmit} className="flex space-x-2">
          <Input
            className="flex-1 font-serif"
            placeholder="Ask about SEC filings..."
            value={input}
            onChange={handleInputChange}
            disabled={isLoading}
          />
          <Button type="submit" disabled={isLoading || !input.trim()}>
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
          </Button>
        </form>
      </div>
    </div>
  )
}
