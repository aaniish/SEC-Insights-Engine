"use client";

import { type CSSProperties, useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

/** One line per thing the app does. Odd and even lines travel in opposite directions. */
const LINES = ["Reads the 10-K.", "Quotes the passage.", "Charts the numbers.", "Shows what changed."];

/** How much of the scroll around the middle of the screen the lines hold still, lined up. */
const REST = 0.22;

/**
 * Lines that slide in from alternating sides as the statement scrolls up the page, line up
 * while it sits mid-screen, and drift apart again as it leaves. Static when motion is reduced.
 */
export function DriftStatement() {
  const ref = useRef<HTMLParagraphElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;

    let frame = 0;
    const update = () => {
      frame = 0;
      const rect = el.getBoundingClientRect();
      const viewport = window.innerHeight;
      // 1 as the block enters at the bottom, 0 when centered, -1 as it leaves at the top.
      const offset = (rect.top + rect.height / 2 - viewport / 2) / ((viewport + rect.height) / 2);
      const travel = Math.min(1, Math.max(0, (Math.abs(offset) - REST) / (1 - REST))) ** 1.5;
      el.style.setProperty("--drift", (Math.sign(offset) * travel).toFixed(4));
      el.style.setProperty("--fade", (1 - 0.75 * travel).toFixed(3));
    };
    const schedule = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    const stop = () => {
      window.removeEventListener("scroll", schedule);
      window.removeEventListener("resize", schedule);
    };

    // Only track scrolling while the statement is near the screen.
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (!entry?.isIntersecting) return stop();
        window.addEventListener("scroll", schedule, { passive: true });
        window.addEventListener("resize", schedule);
        update();
      },
      { rootMargin: "25% 0px" },
    );
    observer.observe(el);
    return () => {
      observer.disconnect();
      stop();
      cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <p ref={ref} className="text-center font-display text-[clamp(2.3rem,9vw,7.5rem)] leading-[1.04]">
      {LINES.map((line, i) => (
        <span
          key={line}
          style={{ "--lane": i % 2 ? 1 : -1 } as CSSProperties}
          className={cn("drift-line block whitespace-nowrap", i % 2 === 1 && "text-navy-ink italic")}
        >
          {line}
        </span>
      ))}
    </p>
  );
}
