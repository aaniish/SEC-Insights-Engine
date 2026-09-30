import { generateText, Output } from "ai";
import { z } from "zod";
import { chatModel, reasoningOptions } from "./models";
import { suggestionsPrompt } from "./prompts";

const schema = z.object({ questions: z.array(z.string()).min(1).max(3) });

/** Three follow-up questions from the cheap model; failures just mean no suggestions. */
export async function suggestFollowUps(input: {
  question: string;
  answer: string;
  tickers: string[];
}): Promise<string[]> {
  if (!input.answer.trim()) return [];
  try {
    const { output } = await generateText({
      model: chatModel("fast"),
      output: Output.object({ schema }),
      prompt: suggestionsPrompt(input),
      providerOptions: reasoningOptions("low"),
      maxOutputTokens: 2000,
    });
    return output.questions.slice(0, 3);
  } catch (error) {
    console.error("suggestions failed", error);
    return [];
  }
}
