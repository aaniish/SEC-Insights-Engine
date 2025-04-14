"use client"
import { CompanySelector } from "@/components/company-selector"
import { ThemeToggle } from "@/components/theme-toggle"
import { memo } from "react"
import { SunIcon as Sunburst } from "lucide-react"

interface SidebarProps {
  onCompaniesChange?: (companies: string[]) => void
  selectedCompanies?: string[]
}

// Memoize the sidebar to prevent unnecessary re-renders
export const Sidebar = memo(function Sidebar({ onCompaniesChange, selectedCompanies = [] }: SidebarProps) {
  const handleCompaniesChange = (companies: string[]) => {
    if (onCompaniesChange) {
      onCompaniesChange(companies)
    }
  }

  return (
    <div className="group border-r bg-background h-full w-[300px] flex-col flex overflow-hidden">
      <div className="p-4 border-b">
        <div className="flex items-center">
          <Sunburst className="h-6 w-6 mr-3 text-primary" />
          <h1 className="text-xl font-semibold font-sans">SEC Insights Engine</h1>
        </div>
      </div>
      <div className="p-4 overflow-y-auto flex-1">
        <div className="space-y-4">
          <h3 className="text-sm font-medium text-muted-foreground font-sans">Select Company</h3>
          <CompanySelector onChange={handleCompaniesChange} selectedCompaniesOverride={selectedCompanies} />
          <p className="text-xs text-muted-foreground mt-2">
            Selecting a specific company will focus your queries on their SEC filings.
          </p>
        </div>
      </div>
      <div className="mt-auto p-4 border-t">
        <div className="flex items-center justify-between">
          <div className="text-xs text-muted-foreground">
            <p>SEC Insights Engine</p>
            <p className="mt-1">By: Anish Thiriveedhi</p>
          </div>
          <ThemeToggle />
        </div>
      </div>
    </div>
  )
}) 