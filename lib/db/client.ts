import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import { requireEnv } from "@/lib/env";
import * as schema from "./schema";

function createDb() {
  return drizzle({
    client: neon(requireEnv("DATABASE_URL")),
    schema,
    casing: "snake_case",
  });
}

let instance: ReturnType<typeof createDb> | undefined;

export function db() {
  instance ??= createDb();
  return instance;
}
