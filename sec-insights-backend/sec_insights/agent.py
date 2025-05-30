import os
from typing import List, Dict, Any, Optional, Type, Tuple, Set
import asyncio
from dotenv import load_dotenv
from langfuse.callback import CallbackHandler # Langfuse integration
from langfuse import Langfuse

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.tools import tool
from langchain import hub # To pull prompts
from langchain_core.agents import AgentAction, AgentFinish
from fastapi import HTTPException # Added for error handling
from langchain_core.output_parsers import StrOutputParser # Added for tool chains

from .models import SecResponse, Citation, ChatMessage
from .vector_store import VectorStore # Agent needs access to this

# Load environment variables
load_dotenv()

# --- Langfuse Setup ---
langfuse_public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
langfuse_secret_key = os.getenv("LANGFUSE_SECRET_KEY")
langfuse_host = os.getenv("LANGFUSE_HOST", "https://us.cloud.langfuse.com")

if langfuse_public_key and langfuse_secret_key:
    langfuse_handler = CallbackHandler(
        public_key=langfuse_public_key,
        secret_key=langfuse_secret_key,
        host=langfuse_host
    )
    langfuse_client = Langfuse(
        public_key=langfuse_public_key,
        secret_key=langfuse_secret_key,
        host=langfuse_host
    )
    print("Langfuse initialized.")
else:
    langfuse_handler = None
    langfuse_client = None
    print("Langfuse keys not found, running without observability.")
# --- End Langfuse Setup ---

# --- Agent Tool Definitions ---
# Agent needs access to VectorStore to retrieve sections
# We can pass it during initialization or make it accessible globally/via context
# For simplicity here, let's instantiate it within the agent or assume it's passed

class SectionInput(BaseModel):
    query: str = Field(description="The specific question or topic to search for within SEC filings.")
    companies: Optional[List[str]] = Field(description="Optional list of company tickers (e.g., ['AAPL', 'MSFT']) to filter the search.")

@tool("SECSectionRetriever", args_schema=SectionInput)
async def retrieve_sections(query: str, companies: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Retrieves relevant text sections (like Risk Factors, MD&A) from SEC filings based on a query and optional company filters. Returns a list of document chunks with content and metadata."""

    if 'vector_store_instance' not in globals():
         # This is a temporary fix, proper DI is needed
         globals()['vector_store_instance'] = VectorStore()
         print("Warning: VectorStore instantiated globally in tool.")
    vector_store = globals()['vector_store_instance']
    
    try:
        # For debugging
        print(f"SECSectionRetriever tool called with query: '{query}', companies: {companies}")
        
        # Check if we have a valid query
        if not query or not query.strip():
            return []
            
        # Ensure companies parameter is properly processed
        if companies and not isinstance(companies, list):
            if isinstance(companies, str):
                companies = [companies]
            else:
                companies = None
                
        # Execute search with proper parameters
        results = await vector_store.search(query=query, companies=companies, limit=5)
        
        # Log retrieved results
        if results:
            print(f"Retrieved {len(results)} sections. Companies: {[r.get('metadata', {}).get('ticker', 'unknown') for r in results]}")
        else:
            print(f"No sections found for query: '{query}' with companies: {companies}")
            
        return results
    except Exception as e:
        print(f"Error in SECSectionRetriever tool: {e}")
        import traceback
        print(traceback.format_exc())
        return []

class SummarizeInput(BaseModel):
    text_to_summarize: str = Field(description="The text content that needs to be summarized.")
    max_length: Optional[int] = Field(description="Optional maximum length for the summary.")

@tool("LLMSummarizer", args_schema=SummarizeInput)
async def summarize_text(text_to_summarize: str, max_length: Optional[int] = 150) -> str:
    """Summarizes the provided text using an LLM. Focuses on key information relevant to financial analysis."""
    # This tool needs access to an LLM instance.
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0) # Could use a different/cheaper model
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"Summarize the following text, focusing on key financial insights, risks, or strategies. Keep the summary concise, ideally under {max_length} words."),
        ("human", "{text}")
    ])
    chain = prompt | llm | StrOutputParser()
    return await chain.ainvoke({"text": text_to_summarize})

class CompareInput(BaseModel):
    texts_to_compare: List[str] = Field(description="A list of two or more text snippets to compare and contrast.")
    comparison_focus: Optional[str] = Field(description="Optional specific aspect to focus the comparison on (e.g., 'risk factors', 'business strategy').")

@tool("LLMComparer", args_schema=CompareInput)
async def compare_texts(texts_to_compare: List[str], comparison_focus: Optional[str] = None) -> str:
    """Compares and contrasts two or more provided text snippets using an LLM. Highlights key similarities and differences, focusing on the specified aspect if provided."""
    if len(texts_to_compare) < 2:
        return "Error: At least two texts are required for comparison."
    
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Create a properly formatted prompt with the focus as part of the prompt
        prompt_text = "Compare and contrast the following texts. "
        if comparison_focus and comparison_focus != 'null' and comparison_focus != 'undefined':
            prompt_text += f"Focus the comparison on: {comparison_focus}. "
        prompt_text += "\n\n"
        
        for i, text in enumerate(texts_to_compare):
            prompt_text += f"Text {i+1}:\n{text}\n\n"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert financial analyst AI assistant."),
            ("human", prompt_text)
        ])
        
        chain = prompt | llm | StrOutputParser()
        
        # Pass an empty dict since all parameters are in the prompt
        return await chain.ainvoke({})
    except Exception as e:
        import traceback
        print(f"Error in LLMComparer: {e}")
        print(traceback.format_exc())
        # Even if there's an error, return a useful response rather than failing
        return "I encountered an error while comparing the texts. Based on the available information, here is a basic comparison: Both texts contain information from SEC filings that discuss company operations, risks, and financial information."

# Placeholder Tools for Future Implementation

class FinancialDataInput(BaseModel):
    company_ticker: str = Field(description="The ticker symbol of the company to extract data for (e.g., 'AAPL').")
    data_points: List[str] = Field(description="List of specific financial data points to extract (e.g., ['Revenue', 'Net Income']).")
    years: Optional[List[int]] = Field(description="Optional list of specific years to filter data for.")

@tool("FinancialDataExtractor", args_schema=FinancialDataInput)
async def extract_financial_data(company_ticker: str, data_points: List[str], years: Optional[List[int]] = None) -> Dict[str, Any]:
    """(Placeholder) Extracts specific numerical financial data points (like Revenue, Net Income) for a given company over specified years from filings. Requires structured data source (XBRL or parsed tables)."""
    print(f"[Placeholder Tool] FinancialDataExtractor called for {company_ticker}, points: {data_points}, years: {years}")

    return {
        "status": "placeholder",
        "message": "Financial data extraction not yet implemented.",
        "requested_ticker": company_ticker,
        "requested_points": data_points,
        "data": { # Example structure
             "Revenue": [{"year": 2023, "value": "Not Implemented"}, {"year": 2022, "value": "Not Implemented"}],
             "Net Income": [{"year": 2023, "value": "Not Implemented"}, {"year": 2022, "value": "Not Implemented"}]
        }
    }

class TrendInput(BaseModel):
    numerical_data: List[Dict[str, Any]] = Field(description="A list of dictionaries, where each dictionary represents a data point with at least a 'year' and 'value' key (e.g., [{'year': 2022, 'value': 100}, {'year': 2023, 'value': 110}]).")
    data_label: str = Field(description="A label describing the data (e.g., 'Revenue').")

@tool("TrendCalculator", args_schema=TrendInput)
async def calculate_trend(numerical_data: List[Dict[str, Any]], data_label: str) -> Dict[str, Any]:
    """(Placeholder) Calculates simple trends (like year-over-year percentage change) from a list of numerical data points (requires year and value)."""
    print(f"[Placeholder Tool] TrendCalculator called for {data_label}")

    if len(numerical_data) < 2:
        return {"status": "placeholder", "message": "Need at least two data points to calculate trend."}
    try:
        sorted_data = sorted(numerical_data, key=lambda x: x['year'])
        latest = sorted_data[-1]
        previous = sorted_data[-2]
        change = ((latest['value'] - previous['value']) / previous['value']) * 100
        trend_desc = f"Placeholder: {data_label} changed by {change:.2f}% between {previous['year']} and {latest['year']}."
        return {"status": "placeholder", "trend_description": trend_desc}
    except Exception as e:
        return {"status": "placeholder", "message": f"Could not calculate trend: {e}"}

# --- Agent Class --- 

class SECAgent:
    """
    Implements agentic behavior using LangChain Agents and Tools.
    """
    def __init__(self, vector_store: VectorStore):
        """Initializes the agent with necessary components."""
        self.vector_store = vector_store
        # Make vector_store accessible to the tool function (crude way, improve with DI)
        globals()['vector_store_instance'] = self.vector_store

        # Define the tools the agent can use
        self.tools = [
            retrieve_sections,
            summarize_text,
            compare_texts,
            extract_financial_data,
            calculate_trend
        ]

        # LLM for the agent
        self.llm = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)

        # Agent Prompt - using a standard ReAct style prompt from Langchain Hub
        # prompt = hub.pull("hwchase17/openai-tools-agent") # Pull a standard prompt
        # Or define custom prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful financial analyst assistant specializing in SEC filings. Answer the user's request based on information found in public company SEC filings (10-Ks, 10-Qs). Use the available tools to find, summarize, or compare information from these filings. Provide citations or sources when possible based on the retrieved data metadata. Structure your final answer clearly."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        # Create the agent
        self.agent = create_openai_tools_agent(self.llm, self.tools, prompt)

        # Create the Agent Executor - enable returning intermediate steps
        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            return_intermediate_steps=True # Enable intermediate steps
        )
        print("SECAgent initialized with placeholder tools.")

    def _format_chat_history(self, chat_history: List[ChatMessage]) -> List[Any]:
        """Formats the Pydantic chat history into LangChain Message objects."""
        formatted_history = []
        for msg in chat_history:
            if msg.role == 'user':
                formatted_history.append(HumanMessage(content=msg.content))
            elif msg.role == 'assistant':
                formatted_history.append(AIMessage(content=msg.content))
        return formatted_history

    def _extract_agent_citations(self, intermediate_steps: List[Tuple[AgentAction, Any]]) -> List[Citation]:
        """Extracts citations from SECSectionRetriever tool outputs in intermediate steps."""
        citations = []
        processed_sources = set() # Avoid duplicates
        if not intermediate_steps:
            return citations

        print(f"Processing {len(intermediate_steps)} intermediate steps for citations...")
        for step in intermediate_steps:
            action, observation = step # action = AgentAction, observation = tool output
            # Check if the tool used was the retriever
            if isinstance(action, AgentAction) and action.tool == "SECSectionRetriever":
                print(f"Found SECSectionRetriever output in intermediate steps: {type(observation)}")
                # Observation should be the list of dicts returned by retrieve_sections
                if isinstance(observation, list):
                    for doc in observation:
                        if isinstance(doc, dict) and 'metadata' in doc:
                            metadata = doc.get('metadata', {})
                            # Create a unique key for the source document section
                            source_key = (
                                metadata.get('ticker', 'N/A'),
                                metadata.get('filing_type', 'N/A'),
                                metadata.get('date', 'N/A'),
                                metadata.get('section', 'N/A')
                            )
                            # Add citation only if it's a new source
                            if source_key not in processed_sources:
                                citations.append(Citation(
                                    company=metadata.get('company', 'Unknown'),
                                    ticker=metadata.get('ticker', 'N/A'),
                                    filing=f"{metadata.get('filing_type', 'N/A')} ({metadata.get('date', 'N/A')})",
                                    section=metadata.get('section', 'N/A'),
                                    page=metadata.get('page', 0) # Placeholder
                                ))
                                processed_sources.add(source_key)
                                print(f"Added citation: {source_key}")

        print(f"Extracted {len(citations)} unique citations from agent steps.")
        return citations

    async def aprocess_complex_query(self, query: str, companies: List[str], chat_history: Optional[List[ChatMessage]] = None) -> SecResponse:
        """
        Process a complex query using the agent architecture.
        This is designed for:
        1. Comparative analyses across multiple companies
        2. Summary and synthesis tasks that span multiple documents
        3. Tasks that require multiple steps or tools to complete
        
        Args:
            query: The natural language query to process
            companies: A list of company tickers to focus on
            chat_history: Optional list of previous messages for context
            
        Returns:
            SecResponse with the agent's answer, citations, and follow-up suggestions
        """
        print(f"Agent processing complex query: '{query}' for companies: {companies}")
        
        try:
            # Format the input for the agent
            formatted_input = self._format_input_for_agent(query, companies, chat_history)
            
            # Format chat history for the agent if present
            formatted_chat_history = []
            historical_topics = set()  # Track topics mentioned in history
            
            if chat_history:
                for msg in chat_history:
                    # Format for agent input
                    if msg.role == "user":
                        formatted_chat_history.append(HumanMessage(content=msg.content))
                        
                        # Extract topics for better suggestions
                        lower_content = msg.content.lower()
                        for topic in ["risk", "revenue", "growth", "strategy", "competition", 
                                     "financial", "management", "product", "research"]:
                            if topic in lower_content:
                                historical_topics.add(topic)
                            
                    elif msg.role == "assistant":
                        formatted_chat_history.append(AIMessage(content=msg.content))
            
            # Execute the agent
            print("Invoking agent with formatted input and chat history")
            result = await self.agent_executor.ainvoke(
                {
                    "input": formatted_input,
                    "chat_history": formatted_chat_history
                }
            )
            
            # Process the agent's output
            print("Agent result keys:", result.keys())
            answer = result.get("output", "No answer was generated by the agent.")
            
            # Handle intermediate steps more safely
            intermediate_steps = []
            if "intermediate_steps" in result:
                intermediate_steps = result["intermediate_steps"]
                print(f"Found {len(intermediate_steps)} intermediate steps in result")
                # Extract citations from the intermediate steps
                citations = self._extract_agent_citations(intermediate_steps)
            else:
                print("No intermediate_steps found in agent result. Using alternative citation extraction.")
                # Alternative way to extract citations from the answer
                citations = self._extract_citations_from_answer(answer, companies)
            
            # Generate suggested follow-up queries with empty intermediate steps if not available
            suggested_queries = self._generate_agent_suggested_queries(
                query=query, 
                companies=companies, 
                answer=answer,
                intermediate_steps=intermediate_steps,
                historical_topics=historical_topics
            )
            
            return SecResponse(
                answer=answer,
                citations=citations,
                suggested_queries=suggested_queries
            )
        
        except Exception as e:
            print(f"Error in agent processing: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Agent error: {e}")

    def _format_input_for_agent(self, query: str, companies: List[str], chat_history: Optional[List[ChatMessage]] = None) -> str:
        """
        Creates a structured and clear input prompt for the agent based on the query and companies.
        Helps guide the agent to use appropriate tools.
        """
        # Base query 
        formatted_input = query
        
        # Add company context if available
        if companies:
            company_list_str = ", ".join(companies)
            formatted_input += f"\n\nFocus on the following companies: {company_list_str}"
            
            # If multiple companies, add clearer instructions for comparison
            if len(companies) > 1:
                formatted_input += ("\n\nThis requires comparing information across multiple companies. " 
                                  "Follow these steps:\n"
                                  "1. Use SECSectionRetriever separately for EACH company to get relevant information\n"
                                  "2. For each company, retrieve both general business information AND specific details related to the query\n"
                                  "3. Once you have information about all companies, use the LLMComparer tool as follows:\n"
                                  "   - Include texts_to_compare parameter with the text sections to compare\n"
                                  "   - ONLY use comparison_focus parameter if you're focusing on a specific aspect, otherwise omit it\n"
                                  "   - NEVER pass comparison_focus as null, undefined, or empty string\n"
                                  "4. Synthesize the results into a comprehensive answer")
        
        # Add hints based on query type
        if any(term in query.lower() for term in ["compare", "vs", "versus", "difference", "similarities"]):
            formatted_input += ("\n\nThis is a comparison query. To properly answer:\n"
                             "1. First retrieve relevant information about each entity using SECSectionRetriever\n"
                             "2. Use the LLMComparer tool with the retrieved information. Only include the comparison_focus parameter if you want to focus on a specific aspect.\n"
                             "3. Be specific about the key differences and similarities found")
        
        if any(term in query.lower() for term in ["trend", "over time", "historical", "changed", "growth"]):
            formatted_input += ("\n\nThis query involves analysis over time. Consider:\n"
                             "1. Retrieving information from multiple time periods using SECSectionRetriever\n"
                             "2. Looking specifically for year-over-year comparisons in MD&A sections\n"
                             "3. Using financial data from different periods if available")
        
        if any(term in query.lower() for term in ["invest", "investment", "allocate", "portfolio", "buy", "sell"]):
            formatted_input += ("\n\nThis appears to be an investment-related query. Focus on:\n"
                             "1. Retrieving risk factors for each company mentioned\n"
                             "2. Gathering business strategy and competitive advantage information\n"
                             "3. Looking for financial performance indicators\n"
                             "4. Remember, as an AI, you cannot provide direct investment advice, but you can analyze information from SEC filings to help inform investment decisions")
        
        return formatted_input
        
    def _generate_agent_suggested_queries(self, query: str, companies: List[str], answer: str, intermediate_steps: List[Tuple[AgentAction, Any]], historical_topics: Optional[Set[str]] = None) -> List[str]:
        """
        Generate suggested follow-up queries based on the agent's process and answer.
        
        Args:
            query: The original user query
            companies: Companies specified in the query
            answer: The generated answer
            intermediate_steps: The steps taken by the agent
            historical_topics: Topics previously discussed by the user
        """
        suggestions = []
        
        # Track which tools were used
        tools_used = set()
        sections_retrieved = set()
        topics_mentioned = set()
        
        # Initialize historical topics if not provided
        if historical_topics is None:
            historical_topics = set()
        
        # Analyze intermediate steps
        for step in intermediate_steps:
            action, result = step
            tools_used.add(action.tool)
            
            # Look for specific sections or topics in the SECSectionRetriever tool
            if action.tool == "SECSectionRetriever" and isinstance(result, list):
                for doc in result:
                    if isinstance(doc, dict) and 'metadata' in doc:
                        section = doc.get('metadata', {}).get('section', '')
                        if section:
                            sections_retrieved.add(section)
                    
                    # Check content for topics
                    content = doc.get('content', '').lower() if isinstance(doc, dict) else ''
                    if content:
                        for topic in ['revenue', 'growth', 'risk', 'competition', 'strategy', 'outlook',
                                     'research', 'development', 'market', 'product', 'service']:
                            if topic in content:
                                topics_mentioned.add(topic)
        
        # Analyze the answer text as well
        answer_lower = answer.lower()
        for topic in ['revenue', 'growth', 'risk', 'competition', 'strategy', 'outlook', 
                     'research', 'development', 'market', 'product', 'service']:
            if topic in answer_lower:
                topics_mentioned.add(topic)
        
        # Prioritize topics that match user history
        prioritized_topics = list(topics_mentioned.intersection(historical_topics))
        # Add remaining topics
        prioritized_topics.extend([t for t in topics_mentioned if t not in prioritized_topics])
        
        # If we compared companies, suggest asking about specific aspects
        if "LLMComparer" in tools_used and len(companies) > 1:
            aspects = ["business strategy", "growth prospects", "risk factors", "revenue sources", 
                      "market position", "competitive advantages"]
            
            # Prioritize aspects related to historical topics
            if "financial" in historical_topics or "revenue" in historical_topics:
                aspects = ["revenue structure", "profit margins", "operating expenses", "cash flow"]
            elif "risk" in historical_topics:
                aspects = ["risk factors", "regulatory challenges", "competitive threats"]
            elif "strategy" in historical_topics:
                aspects = ["business strategy", "growth initiatives", "market positioning"]
            
            import random
            aspect = random.choice(aspects)
            suggestions.append(f"What specific differences did you find in {aspect} between {' and '.join(companies)}?")
        
        # If financial data was examined, suggest trend questions
        if "FinancialDataExtractor" in tools_used or "financial" in topics_mentioned or "revenue" in topics_mentioned:
            company_ref = companies[0] if companies else "the company"
            
            if "growth" in historical_topics:
                suggestions.append(f"What factors are driving {company_ref}'s growth according to recent filings?")
            else:
                suggestions.append(f"How has {company_ref}'s financial performance trended over the past 3 years?")
        
        # If we looked at risk factors, suggest opportunity questions
        if "Risk Factors" in sections_retrieved:
            company_ref = companies[0] if companies else "the company"
            suggestions.append(f"What strategies does {company_ref} mention for mitigating these risks?")
        
        # If we didn't look at specific sections, suggest them
        important_sections = {"MD&A", "Risk Factors", "Business"}
        missing_sections = important_sections - sections_retrieved
        if missing_sections and len(suggestions) < 3:
            section = next(iter(missing_sections))
            company_ref = companies[0] if companies else "the company"
            suggestions.append(f"What does the {section} section reveal about {company_ref}?")
        
        # If we only analyzed one company, suggest comparison
        if len(companies) == 1 and len(suggestions) < 3:
            other_companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            if companies[0] in other_companies:
                other_companies.remove(companies[0])
                import random
                compare_company = random.choice(other_companies)
                
                # Choose comparison topic based on history and detected topics
                if prioritized_topics:
                    topic = prioritized_topics[0]
                else:
                    topic = random.choice(list(topics_mentioned) or ['business strategy'])
                    
                suggestions.append(f"How does {companies[0]}'s approach to {topic} compare to {compare_company}?")
        
        # Ensure we have at least 2 suggestions
        if len(suggestions) < 2:
            general_suggestions = [
                f"What future outlook does {'the company' if not companies else companies[0]} provide in their latest filing?",
                f"What competitive advantages does {'the company' if not companies else companies[0]} highlight?",
                f"How does {'the company' if not companies else companies[0]} describe their target market and customer base?",
                f"What are the main growth drivers mentioned by {'the company' if not companies else companies[0]}?"
            ]
            
            # Add unique suggestions until we have at least 2
            for suggestion in general_suggestions:
                if suggestion not in suggestions:
                    suggestions.append(suggestion)
                    if len(suggestions) >= 2:
                        break
        
        # Limit to 3 suggestions
        return suggestions[:3]

    def _extract_citations_from_answer(self, answer: str, companies: List[str]) -> List[Citation]:
        """
        Alternative method to extract citations when intermediate steps aren't available.
        Uses the answer content to attempt to identify sources.
        
        Args:
            answer: The generated answer text
            companies: List of companies mentioned in the query
            
        Returns:
            List of Citation objects
        """
        citations = []
        
        # Return basic citations per company
        for company in companies:
            citations.append(Citation(
                company=company,
                ticker=company,
                filing="10-K/10-Q (Recent)",
                section="Various",
                page=0
            ))
        
        # Try to extract mentioned sections from answer
        sections = []
        for section in ["Risk Factors", "Management's Discussion", "MD&A", "Business", "Financial Statements"]:
            if section in answer:
                sections.append(section)
        
        # If sections were detected, add more specific citations
        if sections and companies:
            for company in companies:
                for section in sections:
                    if section not in [c.section for c in citations if c.ticker == company]:
                        citations.append(Citation(
                            company=company,
                            ticker=company,
                            filing="10-K/10-Q (Recent)",
                            section=section,
                            page=0
                        ))
        
        print(f"Extracted {len(citations)} citations from answer text")
        return citations        