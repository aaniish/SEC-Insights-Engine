"use client";

import { ArrowRight, Square, X } from "lucide-react";
import { type FormEvent, useEffect, useRef, useState } from "react";
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
  const hero = variant === "hero";

  // Focus the hero box on desktop only; on phones it would pop the keyboard over the page.
  useEffect(() => {
    if (hero && window.matchMedia("(pointer: fine)").matches) textareaRef.current?.focus();
  }, [hero]);

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
        "glass rounded-[1.75rem] p-2 transition-shadow focus-within:shadow-[inset_0_1px_0_var(--glass-highlight),0_0_0_1px_var(--ring),var(--glass-shadow)]",
        hero ? "rounded-[2rem] p-2.5" : "glass-dense",
      )}
    >
      {companies.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-2 pt-1 pb-1">
          {companies.map((company) => (
            <span
              key={company.ticker}
              className="inline-flex items-center gap-1.5 rounded-full bg-navy-soft py-1 pr-1 pl-3 text-xs text-navy-ink"
            >
              <span className="font-semibold tabular">{company.ticker}</span>
              <span className="max-w-[9rem] truncate opacity-80">{company.name}</span>
              <button
                type="button"
                onClick={() => onToggleCompany(company)}
                aria-label={`Remove ${company.ticker}`}
                className="grid size-5 place-items-center rounded-full hover:bg-navy-soft"
              >
                <X className="size-3" />
              </button>
            </span>
          ))}
        </div>
      )}

      <div className="flex items-end gap-2">
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
          placeholder={hero ? "Try ‘What changed in Tesla’s risk factors?’" : "Ask a follow-up…"}
          className={cn(
            "block max-h-48 min-w-0 flex-1 resize-none bg-transparent px-3 outline-none field-sizing-content placeholder:text-graphite/75",
            hero ? "py-3 text-lg sm:text-xl" : "py-2.5 text-base",
          )}
        />
        {busy ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop"
            className="grid size-11 shrink-0 place-items-center rounded-full bg-ink/90 text-paper"
          >
            <Square className="size-3.5 fill-current" />
          </button>
        ) : (
          <button
            type="submit"
            disabled={!text.trim()}
            aria-label="Ask"
            className={cn(
              "gloss grid shrink-0 place-items-center rounded-full transition-[transform,opacity] active:scale-95 disabled:cursor-default disabled:opacity-80",
              hero ? "size-12" : "size-11",
            )}
          >
            <ArrowRight className="size-5" />
          </button>
        )}
      </div>

      <div className="flex items-center gap-1 px-1 pt-1">
        <CompanyPicker selected={companies} onToggle={onToggleCompany} />
        <fieldset className="flex rounded-full bg-muted p-0.5" aria-label="Answer mode">
          {MODES.map((option) => (
            <Tooltip key={option.value}>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  aria-pressed={mode === option.value}
                  onClick={() => onModeChange(option.value)}
                  className={cn(
                    "rounded-full px-3 py-1 text-xs font-medium text-graphite transition-all",
                    mode === option.value && "bg-card text-ink shadow-[0_1px_3px_rgb(10_19_48/0.12)]",
                  )}
                >
                  {option.label}
                </button>
              </TooltipTrigger>
              <TooltipContent>{option.hint}</TooltipContent>
            </Tooltip>
          ))}
        </fieldset>
        {hero && <span className="ml-auto hidden pr-2 text-xs text-graphite sm:block">Enter to ask</span>}
      </div>
    </form>
  );
}
