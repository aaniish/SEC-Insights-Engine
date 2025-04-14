"use client"; // Ensure this is a client component

import { User, Bot, AlertTriangle, Loader2 } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import ReactMarkdown from "react-markdown"
import type { ChatMessage as CustomChatMessage } from "@/lib/types"
import { useEffect, useState, memo } from "react"

interface ChatMessageProps {
  message: CustomChatMessage
  onSuggestedQueryClick?: (query: string) => void
}

// Memoize the component to prevent unnecessary re-renders
export const ChatMessage = memo(function ChatMessage({ message, onSuggestedQueryClick }: ChatMessageProps) {
  const isUser = message.role === "user"
  const isError = message.error === true
  const isLoading = message.isLoading === true
  
  const [loadingText, setLoadingText] = useState<string>("")
  const [loadingStage, setLoadingStage] = useState<number>(0)
  const [loadingDots, setLoadingDots] = useState<number>(0)

  // Update local loading dots
  useEffect(() => {
    if (!isLoading) return;
    
    const dotsTimer = setInterval(() => {
      setLoadingDots(prev => (prev + 1) % 4);
    }, 500);
    
    return () => clearInterval(dotsTimer);
  }, [isLoading]);

  // Loading animation effect
  useEffect(() => {
    if (!isLoading) return;
    
    // List of loading stages with progressively more specific text
    const stages = [
      "Retrieving relevant SEC documents",
      "Analyzing financial information",
      "Processing risk factors and MD&A sections",
      "Synthesizing insights from multiple filings",
      "Preparing comprehensive response"
    ];
    
    // Use message.loadingStage if available, otherwise cycle through stages
    if (message.loadingStage) {
      // Update dots animation every 500ms
      const dotsInterval = setInterval(() => {
        const dots = ".".repeat((message.loadingDots || 0) + 1 > 3 ? 3 : (message.loadingDots || 0) + 1);
        setLoadingText(`${message.loadingStage}${dots}`);
      }, 500);
      
      return () => {
        clearInterval(dotsInterval);
      };
    } else {
      // Cycle through stages every 8 seconds if no loading stage provided
      const stageInterval = setInterval(() => {
        setLoadingStage(prev => (prev + 1) % stages.length);
      }, 8000);
      
      // Update dots animation every 500ms
      const dotsInterval = setInterval(() => {
        const stage = stages[loadingStage];
        const dots = ".".repeat((loadingDots % 3) + 1);
        setLoadingText(`${stage}${dots}`);
      }, 500);
      
      return () => {
        clearInterval(stageInterval);
        clearInterval(dotsInterval);
      };
    }
  }, [isLoading, loadingStage, message.loadingStage, message.loadingDots, loadingDots]);

  return (
    <div
      className={cn(
        "flex gap-3 p-4 rounded-lg",
        isUser ? "bg-muted" : "bg-background border",
        isError ? "border-destructive" : "",
        isLoading ? "border border-muted bg-muted/50" : "",
      )}
    >
      <div
        className={cn(
          "h-8 w-8 rounded-full flex items-center justify-center flex-shrink-0",
          isError ? "bg-destructive/10" : 
          isLoading ? "bg-muted" : "bg-primary/10"
        )}
      >
        {isError ? (
          <AlertTriangle className="h-4 w-4 text-destructive" />
        ) : isLoading ? (
          <Loader2 className="h-4 w-4 text-primary animate-spin" />
        ) : isUser ? (
          <User className="h-4 w-4 text-primary" />
        ) : (
          <Bot className="h-4 w-4 text-primary" />
        )}
      </div>
      <div className="flex-1 space-y-2 overflow-hidden">
        <div className="font-medium text-sm font-sans">{isUser ? "You" : "SEC Insights"}</div>
        {isLoading ? (
          <div className="flex flex-col space-y-4">
            <div className="text-primary font-medium font-serif">
              <span>{loadingText || "Analyzing SEC filings..."}</span>
            </div>
            <div className="w-full bg-secondary rounded-full h-1 mb-4 overflow-hidden">
              <div 
                className="bg-primary h-1 rounded-full" 
                style={{
                  width: '100%',
                  animation: 'indeterminateProgress 1.5s ease-in-out infinite',
                  transformOrigin: '0% 50%'
                }}
              ></div>
            </div>
          </div>
        ) : (
          <div
            className={cn(
              "prose prose-sm max-w-none break-words font-serif",
              isError ? "text-destructive" : "text-foreground"
            )}
          >
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}

        {!isLoading && message.citations && message.citations.length > 0 && (
          <div className="pt-2">
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem value="item-1" className="border-b-0">
                <AccordionTrigger className="text-xs py-1 hover:no-underline font-medium text-primary font-sans">
                  Sources ({message.citations.length})
                </AccordionTrigger>
                <AccordionContent className="pt-1 pb-0">
                  <ul className="list-none space-y-2 pl-0 font-serif">
                    {message.citations.map((citation, index) => (
                      <li key={index} className="text-xs p-2 rounded-md bg-muted">
                        <div className="flex items-start">
                          <span className="font-semibold text-primary mr-1 font-sans">[{index + 1}]</span>
                          <div>
                            <p className="font-medium font-sans">
                              {citation.company} ({citation.ticker})
                            </p>
                            <p className="text-muted-foreground">Filing: {citation.filing}</p>
                            <p className="text-muted-foreground">Section: {citation.section}</p>
                          </div>
                        </div>
                      </li>
                    ))}
                  </ul>
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        )}

        {!isLoading && message.suggestedQueries && message.suggestedQueries.length > 0 && (
          <div className="flex flex-wrap gap-2 pt-2">
            {message.suggestedQueries.map((query, i) => (
              <Button
                key={i}
                variant="outline"
                size="sm"
                className="text-xs h-auto py-1 px-2 font-serif"
                onClick={() => onSuggestedQueryClick?.(query)}
                disabled={!onSuggestedQueryClick}
              >
                {query}
              </Button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
})
