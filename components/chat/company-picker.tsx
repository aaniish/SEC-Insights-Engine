"use client";

import { Building2, Check } from "lucide-react";
import { useEffect, useState } from "react";
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";
import { cn } from "@/lib/utils";

export interface PickedCompany {
  ticker: string;
  name: string;
}

interface SearchResult extends PickedCompany {
  cik: number;
  indexed: boolean;
}

export const MAX_COMPANIES = 3;

function useCompanySearch(query: string, open: boolean) {
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const res = await fetch(`/api/companies/search?q=${encodeURIComponent(query)}`, {
          signal: controller.signal,
        });
        if (res.ok) setResults((await res.json()).results);
      } catch {
        // Aborted by a newer keystroke.
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, 150);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [query, open]);

  return { results, loading };
}

export function CompanyPicker({
  selected,
  onToggle,
}: {
  selected: PickedCompany[];
  onToggle: (company: PickedCompany) => void;
}) {
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const { results, loading } = useCompanySearch(query, open);
  const isFull = selected.length >= MAX_COMPANIES;

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <button
          type="button"
          className="inline-flex h-8 items-center gap-1.5 rounded-full px-3 text-xs font-medium text-graphite transition-colors hover:bg-muted hover:text-ink aria-expanded:bg-muted aria-expanded:text-ink"
        >
          <Building2 className="size-3.5" />
          {selected.length === 0 ? "Pick companies" : "Companies"}
        </button>
      </PopoverTrigger>
      <PopoverContent
        align="start"
        sideOffset={10}
        className="glass glass-dense w-[min(24rem,calc(100vw-2rem))] overflow-hidden rounded-2xl border-0 p-0 ring-0"
      >
        <Command shouldFilter={false} className="bg-transparent">
          <CommandInput
            value={query}
            onValueChange={setQuery}
            placeholder="Search by ticker or name, e.g. CMG"
          />
          <CommandList>
            {!loading && <CommandEmpty>No SEC filer matches “{query}”.</CommandEmpty>}
            <CommandGroup heading={query ? "SEC filers" : "Ready to search"}>
              {results.map((company) => {
                const isSelected = selected.some((c) => c.ticker === company.ticker);
                return (
                  <CommandItem
                    key={company.cik}
                    value={company.ticker}
                    disabled={isFull && !isSelected}
                    onSelect={() => onToggle({ ticker: company.ticker, name: company.name })}
                    className="gap-3 rounded-xl"
                  >
                    <span className="w-14 shrink-0 text-xs font-semibold tabular">{company.ticker}</span>
                    <span className="min-w-0 flex-1 truncate">{company.name}</span>
                    {company.indexed ? (
                      <span className="shrink-0 text-[0.7rem] text-graphite">Indexed</span>
                    ) : (
                      <span className="shrink-0 text-[0.7rem] text-navy-ink">On demand</span>
                    )}
                    <Check
                      className={cn(
                        "size-3.5 shrink-0 text-navy-ink",
                        isSelected ? "opacity-100" : "opacity-0",
                      )}
                    />
                  </CommandItem>
                );
              })}
            </CommandGroup>
          </CommandList>
          <p className="border-t px-3 py-2.5 text-[0.72rem] leading-snug text-graphite">
            Up to {MAX_COMPANIES} companies. “On demand” filings are downloaded and indexed the first time you
            ask (about 20 seconds).
          </p>
        </Command>
      </PopoverContent>
    </Popover>
  );
}
