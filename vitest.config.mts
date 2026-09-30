import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

try {
  process.loadEnvFile(".env.local"); // live tests need SEC_EMAIL_ADDRESS
} catch {
  // CI provides env directly; unit tests need none.
}

export default defineConfig({
  resolve: {
    alias: { "@": fileURLToPath(new URL("./", import.meta.url)) },
  },
  test: {
    environment: "node",
    include: ["**/*.test.ts"],
    exclude: ["node_modules", ".next"],
  },
});
