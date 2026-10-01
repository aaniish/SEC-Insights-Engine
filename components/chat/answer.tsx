"use client";

import { memo, type ReactNode } from "react";
import { type Components, Streamdown } from "streamdown";
import { CitationMarker } from "@/components/citations/citations";

/** Answers only ever need to link into EDGAR; anything else the model writes stays plain text. */
const SEC_LINK = /^https:\/\/(www\.)?sec\.gov\//;

const components: Components = {
  a: (props) => {
    const href = typeof props.href === "string" ? props.href : undefined;
    const cite = href?.match(/^#cite-(\d+)$/);
    if (cite) return <CitationMarker n={Number(cite[1])} />;
    if (!href || !SEC_LINK.test(href)) return <span>{props.children as ReactNode}</span>;
    return (
      <a href={href} target="_blank" rel="noreferrer" className="text-navy-ink underline underline-offset-2">
        {props.children as ReactNode}
      </a>
    );
  },
  // Images in model output could load third-party URLs (tracking pixels), so none are rendered.
  img: () => null,
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
      skipHtml
      controls={{ table: false, code: false }}
      components={components}
    >
      {markdown}
    </Streamdown>
  );
});
