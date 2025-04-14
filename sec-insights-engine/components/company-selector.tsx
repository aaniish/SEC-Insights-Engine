"use client"

import { useState, useEffect, memo } from "react"
import { Check, ChevronsUpDown, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from "@/components/ui/command"
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"

interface CompanySelectorProps {
  onChange: (companies: string[]) => void
  localStorageKey?: string
  selectedCompaniesOverride?: string[]
}

interface CompanyOption {
  value: string
  label: string
}

const DEFAULT_STORAGE_KEY = "pinnedCompanies"

// Define backend API URL from environment variable or fallback for local dev
const backendApiUrl = process.env.NEXT_PUBLIC_BACKEND_URL || '/api';

// Memoize the component to prevent unnecessary re-renders
export const CompanySelector = memo(function CompanySelector({
  onChange,
  localStorageKey = DEFAULT_STORAGE_KEY,
  selectedCompaniesOverride,
}: CompanySelectorProps) {
  const [open, setOpen] = useState(false)
  const [availableCompanies, setAvailableCompanies] = useState<CompanyOption[]>([])
  const [selectedCompanies, setSelectedCompanies] = useState<string[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  // Use the override if provided
  useEffect(() => {
    if (selectedCompaniesOverride !== undefined) {
      setSelectedCompanies(selectedCompaniesOverride)
    }
  }, [selectedCompaniesOverride])

  useEffect(() => {
    const fetchCompanies = async () => {
      setIsLoading(true)
      setError(null)
      try {
        const response = await fetch(`${backendApiUrl}/companies`)
        if (!response.ok) {
          throw new Error(`Failed to fetch companies: ${response.statusText}`)
        }
        const data = await response.json()
        const formattedCompanies: CompanyOption[] = (data.companies || []).map(
          (comp: { ticker: string; name: string }): CompanyOption => ({
            value: comp.ticker,
            label: `${comp.name} (${comp.ticker})`,
          }),
        )
        setAvailableCompanies(formattedCompanies)
      } catch (err: any) {
        console.error("Error fetching companies:", err)
        setError(err.message || "Could not load company list.")
        setAvailableCompanies([])
      } finally {
        setIsLoading(false)
      }
    }

    fetchCompanies()
  }, [])

  // Only load from localStorage if no override is provided
  useEffect(() => {
    if (selectedCompaniesOverride === undefined) {
      try {
        const storedSelection = localStorage.getItem(localStorageKey)
        if (storedSelection) {
          const parsedSelection = JSON.parse(storedSelection)
          if (Array.isArray(parsedSelection)) {
            setSelectedCompanies(parsedSelection)
            onChange(parsedSelection)
          }
        }
      } catch (error) {
        console.error("Failed to load pinned companies from localStorage:", error)
      }
    }
  }, [localStorageKey, onChange, selectedCompaniesOverride])

  // Save to localStorage when selection changes
  useEffect(() => {
    try {
      localStorage.setItem(localStorageKey, JSON.stringify(selectedCompanies))
    } catch (error) {
      console.error("Failed to save pinned companies to localStorage:", error)
    }
  }, [selectedCompanies, localStorageKey])

  const handleSelect = (value: string) => {
    if (!availableCompanies.some((c: CompanyOption) => c.value === value)) return

    // Always set to just the selected company (replacing any existing selection)
    setSelectedCompanies([value])
    onChange([value])
    setOpen(false)
  }

  const removeCompany = (value: string) => {
    setSelectedCompanies([])
    onChange([])
  }

  const getLabelForValue = (value: string): string => {
    return availableCompanies.find((c: CompanyOption) => c.value === value)?.label || value
  }

  return (
    <div className="space-y-2">
      <Popover open={open} onOpenChange={setOpen}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full justify-between h-10"
            disabled={isLoading || !!error}
          >
            <span className="truncate">
              {selectedCompanies.length > 0
                ? `Selected: ${getLabelForValue(selectedCompanies[0])}`
                : "Select a company..."}
            </span>
            <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[--radix-popover-trigger-width] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search companies..." />
            <CommandList>
              {isLoading ? (
                <div className="p-2 space-y-1">
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                  <Skeleton className="h-8 w-full" />
                </div>
              ) : error ? (
                <div className="p-4 text-center text-sm text-destructive">{error}</div>
              ) : (
                <>
                  <CommandEmpty>No company found.</CommandEmpty>
                  <CommandGroup>
                    {availableCompanies.map((company: CompanyOption) => (
                      <CommandItem
                        key={company.value}
                        value={company.label}
                        onSelect={() => handleSelect(company.value)}
                      >
                        <Check
                          className={cn(
                            "mr-2 h-4 w-4",
                            selectedCompanies.includes(company.value) ? "opacity-100" : "opacity-0",
                          )}
                        />
                        {company.label}
                      </CommandItem>
                    ))}
                  </CommandGroup>
                </>
              )}
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {selectedCompanies.length > 0 && (
        <div className="flex flex-wrap gap-1 pt-1">
          {selectedCompanies.map((value: string) => (
            <Badge key={value} variant="secondary" className="text-xs group pl-2 pr-1">
              {getLabelForValue(value)}
              <button
                className="ml-1 rounded-full opacity-70 group-hover:opacity-100 hover:bg-muted p-0.5 focus:outline-none focus:ring-1 focus:ring-ring transition-colors"
                onClick={() => removeCompany(value)}
                aria-label={`Remove ${getLabelForValue(value)}`}
              >
                <X className="h-3 w-3" />
              </button>
            </Badge>
          ))}
          {selectedCompanies.length > 0 && (
            <Badge
              variant="outline"
              className="text-xs cursor-pointer hover:bg-muted"
              onClick={() => {
                setSelectedCompanies([])
                onChange([])
              }}
            >
              Clear all
            </Badge>
          )}
        </div>
      )}
    </div>
  )
})
