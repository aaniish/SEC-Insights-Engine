# === Base Stage ===
# Use an official Node.js runtime as a parent image
# Use a version compatible with your project (e.g., Node 20)
FROM node:20-slim AS base

# Set working directory
WORKDIR /app

# Install pnpm
ENV PNPM_HOME="/pnpm"
ENV PATH="$PNPM_HOME:$PATH"
RUN corepack enable
RUN corepack prepare pnpm@latest --activate

# === Dependencies Stage ===
FROM base AS deps

WORKDIR /app

# Copy dependency manifests
COPY package.json pnpm-lock.yaml ./

# Install production dependencies
RUN pnpm install --prod --frozen-lockfile

# === Build Stage ===
FROM base AS builder

WORKDIR /app

# Copy dependency manifests and install ALL dependencies (including dev)
COPY package.json pnpm-lock.yaml ./
RUN pnpm install --frozen-lockfile

# Copy the rest of the application code
COPY . .

# Build the Next.js application
RUN pnpm build

# === Runner Stage ===
FROM base AS runner

WORKDIR /app

# Set NODE_ENV to production
ENV NODE_ENV production
# Disable telemetry
ENV NEXT_TELEMETRY_DISABLED 1

# Copy required artifacts from the builder stage
# Copy standalone output
COPY --from=builder /app/.next/standalone ./ 
# Copy static assets
COPY --from=builder /app/.next/static ./.next/static 
# Copy public assets
COPY --from=builder /app/public ./public

# Expose the port the app runs on (default 3000)
EXPOSE 3000

# Set the user to run the application (optional but recommended)
# RUN addgroup --system --gid 1001 nodejs
# RUN adduser --system --uid 1001 nextjs
# USER nextjs

# Command to run the application
# Uses the Node.js server included in the standalone output
CMD ["node", "server.js"] 