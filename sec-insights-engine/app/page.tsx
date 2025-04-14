"use client"

import { Chat } from "@/components/chat"
import { Sidebar } from "@/components/sidebar"
import { useCompanies } from "@/components/companies-provider"
import { useEffect, useState } from "react"

export default function Home() {
  const { selectedCompanies, setSelectedCompanies } = useCompanies()
  const [isClient, setIsClient] = useState(false)

  // Prevent hydration mismatch by only rendering after client-side hydration
  useEffect(() => {
    setIsClient(true)
  }, [])

  if (!isClient) {
    return null // Return nothing during SSR to prevent hydration mismatch
  }

  return (
    <div className="flex h-screen bg-background">
      <Sidebar onCompaniesChange={setSelectedCompanies} selectedCompanies={selectedCompanies} />
      <div className="flex flex-col flex-1 h-full">
        <main className="flex-1 overflow-hidden">
          <div className="container h-full py-6">
            <div className="h-full rounded-lg border bg-card">
              <div className="h-full px-4 py-6 lg:px-8">
                <Chat selectedCompanies={selectedCompanies} onCompaniesChange={setSelectedCompanies} />
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}
