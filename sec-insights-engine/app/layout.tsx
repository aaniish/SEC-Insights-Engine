import type { Metadata } from "next"
import "@/app/globals.css"
import { Space_Grotesk, Newsreader } from "next/font/google"
import { ThemeProvider } from "@/components/theme-provider"
import { CompaniesProvider } from "@/components/companies-provider"
import { Toaster } from "@/components/ui/sonner"

// Sans-serif font similar to Styrene for headings
const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans",
})

// Serif font similar to Tiempos for body text
const newsreader = Newsreader({
  subsets: ["latin"],
  variable: "--font-serif",
})

export const metadata: Metadata = {
  title: "SEC Insights Engine",
  description: "AI-powered insights from SEC filings",
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${spaceGrotesk.variable} ${newsreader.variable} font-serif`}>
        <ThemeProvider attribute="class" defaultTheme="system" enableSystem disableTransitionOnChange>
          <CompaniesProvider>
            {children}
            <Toaster richColors position="top-right" />
          </CompaniesProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}


import './globals.css'