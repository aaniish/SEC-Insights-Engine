"use client";

import { useChat } from "@ai-sdk/react";
import { DefaultChatTransport } from "ai";
import dynamic from "next/dynamic";
import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { SiteHeader } from "@/components/site-header";
import type { ChatMode } from "@/lib/ai/models";
import type { ChatMessage } from "@/lib/ai/types";
import { toTurns } from "@/lib/chat/parts";
import { CURATED_COMPANIES, type FeaturedQuestion } from "@/lib/featured";
import { MAX_COMPANIES, type PickedCompany } from "./company-picker";
import { Composer } from "./composer";
import { Home } from "./home";

// The answer view pulls in the markdown renderer, charts, and diff library; keep it
// out of the home page bundle and prefetch it while the visitor reads the page.
const loadTurn = () => import("./turn").then((m) => m.Turn);
const Turn = dynamic(loadTurn, { ssr: false });

/** API errors arrive as the raw response body; surface the message inside it. */
function readableError(error: Error): string {
  try {
    const parsed = JSON.parse(error.message);
    if (typeof parsed?.error === "string") return parsed.error;
  } catch {
    // Not JSON.
  }
  return error.message || "Something went wrong. Please try again.";
}

export function ChatApp() {
  const [companies, setCompanies] = useState<PickedCompany[]>([]);
  const [mode, setMode] = useState<ChatMode>("fast");
  const options = useRef({ tickers: [] as string[], mode: "fast" as ChatMode });
  options.current = { tickers: companies.map((c) => c.ticker), mode };

  const [transport] = useState(
    () => new DefaultChatTransport<ChatMessage>({ api: "/api/chat", body: () => options.current }),
  );
  const { messages, sendMessage, setMessages, status, stop, error, regenerate, clearError } =
    useChat<ChatMessage>({
      transport,
    });

  const busy = status === "submitted" || status === "streaming";
  const turns = toTurns(messages);
  const lastTurnId = turns.at(-1)?.question.id;

  // Bring each new question to the top of the viewport; the answer streams in below it.
  useEffect(() => {
    const prefetch = () => void loadTurn();
    if ("requestIdleCallback" in window) {
      const id = window.requestIdleCallback(prefetch);
      return () => window.cancelIdleCallback(id);
    }
    const timer = setTimeout(prefetch, 1500);
    return () => clearTimeout(timer);
  }, []);

  const turnCount = turns.length;
  useEffect(() => {
    if (!lastTurnId) return;
    if (turnCount === 1) window.scrollTo({ top: 0 });
    else
      document
        .querySelector(`[data-turn="${lastTurnId}"]`)
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, [lastTurnId, turnCount]);

  const ask = (text: string) => {
    if (busy) return;
    clearError();
    sendMessage({ text });
  };

  const toggleCompany = (company: PickedCompany) => {
    setCompanies((current) => {
      if (current.some((c) => c.ticker === company.ticker))
        return current.filter((c) => c.ticker !== company.ticker);
      if (current.length >= MAX_COMPANIES) {
        toast(`Compare up to ${MAX_COMPANIES} companies at a time.`);
        return current;
      }
      return [...current, company];
    });
  };

  const openFeatured = async (featured: FeaturedQuestion) => {
    setCompanies(
      featured.tickers.map((t) => ({
        ticker: t,
        name: CURATED_COMPANIES.find((c) => c.ticker === t)?.name ?? t,
      })),
    );
    options.current = { tickers: [...featured.tickers], mode };
    try {
      const res = await fetch(`/api/featured/${featured.id}`);
      if (res.ok) {
        const cached = (await res.json()) as { messages: ChatMessage[] };
        setMessages(cached.messages);
        return;
      }
    } catch {
      // Fall through to asking live.
    }
    ask(featured.question);
  };

  const reset = () => {
    stop();
    setMessages([]);
    clearError();
  };

  const composer = (variant: "hero" | "dock") => (
    <Composer
      variant={variant}
      onSubmit={ask}
      onStop={stop}
      busy={busy}
      companies={companies}
      onToggleCompany={toggleCompany}
      mode={mode}
      onModeChange={setMode}
    />
  );

  return (
    <div className="flex min-h-dvh flex-col">
      <SiteHeader onNewQuestion={messages.length > 0 ? reset : undefined} />
      {turns.length === 0 ? (
        <Home composer={composer("hero")} onFeatured={openFeatured} onPickCompany={toggleCompany} />
      ) : (
        <>
          <main className="mx-auto w-full max-w-6xl flex-1 space-y-16 px-4 pt-8 pb-40 sm:px-6">
            {turns.map((turn, i) => {
              const isLast = i === turns.length - 1;
              return (
                <Turn
                  key={turn.question.id}
                  question={turn.question}
                  answer={turn.answer}
                  working={isLast && busy}
                  error={isLast && error ? readableError(error) : undefined}
                  onRetry={isLast && error ? () => regenerate() : undefined}
                  onAsk={ask}
                />
              );
            })}
          </main>
          <div className="sticky bottom-0 z-20 bg-gradient-to-t from-paper via-paper/95 to-transparent pt-8 pb-4">
            <div className="mx-auto max-w-6xl px-4 sm:px-6">
              <div className="max-w-[calc(100%-19.5rem)] max-lg:max-w-none">{composer("dock")}</div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
