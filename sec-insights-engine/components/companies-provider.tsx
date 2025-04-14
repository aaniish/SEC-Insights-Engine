"use client"

import { createContext, useContext, useState, type ReactNode } from "react"

interface CompaniesContextType {
  selectedCompanies: string[]
  setSelectedCompanies: (companies: string[]) => void
}

const CompaniesContext = createContext<CompaniesContextType | undefined>(undefined)

export function CompaniesProvider({ children }: { children: ReactNode }) {
  const [selectedCompanies, setSelectedCompanies] = useState<string[]>([])

  return (
    <CompaniesContext.Provider value={{ selectedCompanies, setSelectedCompanies }}>
      {children}
    </CompaniesContext.Provider>
  )
}

export function useCompanies() {
  const context = useContext(CompaniesContext)
  if (context === undefined) {
    throw new Error("useCompanies must be used within a CompaniesProvider")
  }
  return context
} 