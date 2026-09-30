"use client";

import { memo, type ReactNode } from "react";
import { type Components, Streamdown } from "streamdown";
import { CitationMarker } from "@/components/citations/citations";

const components: Components = {
  a: (props) => {
    const href = typeof props.href === "string" ? props.href : undefined;
    const cite = href?.match(/^#cite-(\d+)$/);
    if (cite) return <CitationMarker n={Number(cite[1])} />;
    return (
      <a href={href} target="_blank" rel="noreferrer" className="text-amber-ink underline underline-offset-2">
        {props.children as ReactNode}
      </a>
    );
  },
};

/** Streaming markdown answer; "[n](#cite-n)" links become citation markers. */
export const Answer = memo(function Answer({
  markdown,
  streaming,
}: {
  markdown: string;
  streaming: boolean;
}) {
  return (
    <Streamdown
      className="answer"
      mode={streaming ? "streaming" : "static"}
      isAnimating={streaming}
      linkSafety={{ enabled: false }}
      controls={{ table: false, code: false }}
      components={components}
    >
      {markdown}
    </Streamdown>
  );
});
