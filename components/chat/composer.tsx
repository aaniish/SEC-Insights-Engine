"use client";

import { ArrowUp, Square, X } from "lucide-react";
import { type FormEvent, useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import type { ChatMode } from "@/lib/ai/models";
import { cn } from "@/lib/utils";
import { CompanyPicker, type PickedCompany } from "./company-picker";

interface ComposerProps {
  onSubmit: (text: string) => void;
  onStop: () => void;
  busy: boolean;
  companies: PickedCompany[];
  onToggleCompany: (company: PickedCompany) => void;
  mode: ChatMode;
  onModeChange: (mode: ChatMode) => void;
  variant: "hero" | "dock";
}

const MODES: { value: ChatMode; label: string; hint: string }[] = [
  { value: "fast", label: "Fast", hint: "Quick, focused answers" },
  { value: "deep", label: "Deep", hint: "A stronger model that searches more thoroughly (5 per day)" },
];

export function Composer({
  onSubmit,
  onStop,
  busy,
  companies,
  onToggleCompany,
  mode,
  onModeChange,
  variant,
}: ComposerProps) {
  const [text, setText] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Focus the hero box on desktop only; on phones it would pop the keyboard over the page.
  useEffect(() => {
    if (variant === "hero" && window.matchMedia("(pointer: fine)").matches) textareaRef.current?.focus();
  }, [variant]);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    const question = text.trim();
    if (!question || busy) return;
    onSubmit(question);
    setText("");
  };

  return (
    <form
      onSubmit={submit}
      className={cn(
        "rounded-xl border bg-card shadow-[0_1px_0_rgba(0,0,0,0.02),0_8px_24px_-12px_rgba(20,24,29,0.18)] transition-colors focus-within:border-amber/70",
        variant === "hero" && "shadow-[0_1px_0_rgba(0,0,0,0.02),0_18px_40px_-20px_rgba(20,24,29,0.28)]",
      )}
    >
      {companies.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-3 pt-3">
          {companies.map((company) => (
            <span
              key={company.ticker}
              className="inline-flex items-center gap-1.5 rounded-md border bg-muted py-0.5 pr-1 pl-2 text-xs"
            >
              <span className="font-mono font-medium">{company.ticker}</span>
              <span className="max-w-[10rem] truncate text-graphite">{company.name}</span>
              <button
                type="button"
                onClick={() => onToggleCompany(company)}
                aria-label={`Remove ${company.ticker}`}
                className="rounded-sm p-0.5 text-graphite hover:bg-accent hover:text-foreground"
              >
                <X className="size-3" />
              </button>
            </span>
          ))}
        </div>
      )}
      <label htmlFor={`question-${variant}`} className="sr-only">
        Ask about a company’s SEC filings
      </label>
      <textarea
        id={`question-${variant}`}
        ref={textareaRef}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) submit(e);
        }}
        rows={1}
        maxLength={1000}
        placeholder={
          variant === "hero" ? "Ask about risks, margins, segments, or what changed…" : "Ask a follow-up…"
        }
        className={cn(
          "block max-h-48 w-full resize-none bg-transparent px-4 pt-3.5 pb-2 outline-none field-sizing-content placeholder:text-graphite/80",
          variant === "hero" ? "min-h-[4.5rem] text-lg" : "min-h-12 text-base",
        )}
      />
      <div className="flex items-center gap-1 px-2 pb-2">
        <CompanyPicker selected={companies} onToggle={onToggleCompany} />
        <fieldset className="ml-1 flex rounded-md border p-0.5" aria-label="Answer mode">
          {MODES.map((option) => (
            <Tooltip key={option.value}>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-pressed={mode === option.value}
                  onClick={() => onModeChange(option.value)}
                  className={cn(
                    "rounded-[5px] px-2.5 py-1 font-mono text-[0.7rem] text-graphite transition-colors",
                    mode === option.value && "bg-ink text-paper",
                  )}
                >
                  {option.label}
                </button>
              </TooltipTrigger>
              <TooltipContent>{option.hint}</TooltipContent>
            </Tooltip>
          ))}
        </fieldset>
        {busy ? (
          <Button
            type="button"
            size="icon"
            variant="outline"
            onClick={onStop}
            className="ml-auto"
            aria-label="Stop"
          >
            <Square className="size-3.5 fill-current" />
          </Button>
        ) : (
          <Button type="submit" size="icon" disabled={!text.trim()} className="ml-auto" aria-label="Ask">
            <ArrowUp className="size-4" />
          </Button>
        )}
      </div>
    </form>
  );
}
