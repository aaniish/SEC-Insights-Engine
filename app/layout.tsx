import type { Metadata, Viewport } from "next";
import { Inter, Newsreader } from "next/font/google";
import { Providers } from "@/components/providers";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });
const newsreader = Newsreader({
  subsets: ["latin"],
  axes: ["opsz"],
  style: ["normal", "italic"],
  variable: "--font-newsreader",
});

export const metadata: Metadata = {
  title: "SEC Insights — answers from 10-K and 10-Q filings",
  description:
    "Ask questions about any US public company. Answers cite the exact passages in its SEC filings and chart reported financials from XBRL data.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#ffffff" },
    { media: "(prefers-color-scheme: dark)", color: "#020409" },
  ],
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" suppressHydrationWarning className={`${inter.variable} ${newsreader.variable}`}>
      <body className="min-h-dvh">
        <Providers>
          <div className="canvas flex min-h-dvh flex-col">{children}</div>
        </Providers>
      </body>
    </html>
  );
}
