# SEC Insights Engine

An AI-powered application for extracting insights from SEC filings.

## Features

- Chat interface for asking questions about public company SEC filings
- Supports 10-K and 10-Q reports analysis
- Company selection to focus queries on specific organizations
- Dark and light mode themes
- Responsive design with custom black/grey and orange theme

## Setup

### Prerequisites

- Node.js 18+
- Docker and Docker Compose
- Git

### Environment Setup

1. Clone the repository:
   ```bash
   git clone <your-repository-url>
   cd sec-insights-engine
   ```

2. Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file and fill in your:
   - OpenAI API key
   - SEC API credentials
   - Other required API keys

3. Start the application with Docker Compose:
   ```bash
   docker-compose up --build
   ```

4. Open your browser to [http://localhost:3000](http://localhost:3000)

## Git Repository Setup

If you're setting up this project as a new Git repository:

1. Initialize Git:
   ```bash
   git init
   ```

2. Add your files:
   ```bash
   git add .
   ```

3. Make your first commit:
   ```bash
   git commit -m "Initial commit"
   ```

4. Add a remote repository:
   ```bash
   git remote add origin <your-repository-url>
   ```

5. Push to the remote repository:
   ```bash
   git push -u origin main
   ```

## Development

### File Structure

- `sec-insights-engine/`: Frontend Next.js application
- `sec-insights-backend/`: Python backend for SEC data processing
- `docker-compose.yml`: Docker configuration for the full stack

### API Keys Protection

Important: Never commit your `.env` file to version control. It contains sensitive API keys and is already in the `.gitignore` file.

## License

[MIT License](LICENSE) 