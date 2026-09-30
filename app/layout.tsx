import type { Metadata, Viewport } from "next";
import { Archivo, IBM_Plex_Mono, Newsreader } from "next/font/google";
import { Providers } from "@/components/providers";
import "./globals.css";

const archivo = Archivo({ subsets: ["latin"], axes: ["wdth"], variable: "--font-archivo" });
const newsreader = Newsreader({
  subsets: ["latin"],
  style: ["normal", "italic"],
  variable: "--font-newsreader",
});
const plexMono = IBM_Plex_Mono({ subsets: ["latin"], weight: ["400", "500"], variable: "--font-plex-mono" });

export const metadata: Metadata = {
  title: "SEC Insights — answers from 10-K and 10-Q filings",
  description:
    "Ask questions about any US public company. Answers cite the exact passages in its SEC filings and chart reported financials from XBRL data.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f4f5f2" },
    { media: "(prefers-color-scheme: dark)", color: "#0e1115" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${archivo.variable} ${newsreader.variable} ${plexMono.variable} h-full`}
    >
      <body className="min-h-full">
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
