from typing import List, Optional, Dict, Any
import asyncio
import os
from dotenv import load_dotenv
import re # For routing logic

from .models import SecResponse, Citation, ChatMessage
from .vector_store import VectorStore
from .agent import SECAgent # Import the agent

# Import necessary LangChain components
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema import Document
from fastapi import HTTPException # To re-raise agent errors
from langchain_core.messages import AIMessage, HumanMessage # Needed for history formatting

# Load environment variables (for OPENAI_API_KEY)
load_dotenv()

# Keywords to trigger agent routing
AGENT_KEYWORDS = ['compare', 'contrast', 'summarize', 'trend', 'analyze', 'difference', 'similarity']

class RAGPipeline:
    def __init__(self):
        """Initializes the RAG Pipeline components, including the SECAgent."""
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")

        # Initialize core components
        self.vector_store = VectorStore() # Initialize vector store first
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, api_key=openai_api_key)

        # Define the RAG chain prompt template with history
        rag_template_with_history = ChatPromptTemplate.from_messages([
            ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
        ])
        # Note: Removed the "Answer based *only* on context" constraint to allow history usage.
        # Adjust system prompt as needed.
        self.rag_prompt_with_history = rag_template_with_history

        # Define the standard RAG chain using LCEL
        self.rag_chain = (
            {
                # Context retriever gets the full input dict including companies
                "context": RunnableLambda(self._retrieve_rag_context),
                # Pass the original input through for later steps
                "original_input": RunnablePassthrough()
            }
            # Now, select only the parts needed for the prompt
            | RunnablePassthrough.assign(
                question=lambda x: x["original_input"]["question"],
                chat_history=lambda x: x["original_input"]["chat_history"]
            )
            | self.rag_prompt_with_history
            | self.llm
            | StrOutputParser()
        )
        print("RAG Chain initialized with history support.")

        # Initialize the SECAgent, passing the vector store
        try:
            self.agent = SECAgent(vector_store=self.vector_store)
            print("SECAgent initialized within RAGPipeline.")
        except Exception as e:
            print(f"Warning: Failed to initialize SECAgent: {e}. Agent functionality will be unavailable.")
            self.agent = None # Disable agent if init fails

    async def _retrieve_rag_context(self, inputs: Dict[str, Any]) -> str:
        """Retrieves context specifically for the standard RAG chain."""
        # The input here is the dict {'question': question, 'companies': companies}
        # passed directly to this lambda function from the chain definition
        query = inputs["question"]
        companies = inputs.get("companies", [])

        retrieved_docs = await self.vector_store.search(query=query, companies=companies, limit=3)
        if not retrieved_docs:
            return "No relevant context found."
        # Format for RAG chain (slightly different from agent's retriever)
        context = "\n\n---\n\n".join([
            f"Source: {doc['metadata'].get('filing_type', 'N/A')} ({doc['metadata'].get('company', 'N/A')}, {doc['metadata'].get('date', 'N/A')}), Section: {doc['metadata'].get('section', 'N/A')}\n" \
            f"Content: {doc['content']}"
            for doc in retrieved_docs
        ])
        return context

    def _extract_rag_citations(self, retrieved_docs: List[Dict[str, Any]]) -> List[Citation]:
        """Extracts citations from documents retrieved by the RAG chain."""
        # This is the same logic as before, just renamed for clarity
        citations = []
        processed_sources = set()
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            source_key = (metadata.get('ticker'), metadata.get('filing_type'), metadata.get('date'), metadata.get('section'))
            if source_key not in processed_sources:
                 citations.append(Citation(
                    company=metadata.get('company', 'Unknown'),
                    ticker=metadata.get('ticker', 'N/A'),
                    filing=f"{metadata.get('filing_type', 'N/A')} ({metadata.get('date', 'N/A')})",
                    section=metadata.get('section', 'N/A'),
                    page=metadata.get('page', 0) # Placeholder
                ))
                 processed_sources.add(source_key)
        return citations

    def _should_route_to_agent(self, query: str, companies: List[str]) -> bool:
        """Simple routing logic based on keywords or multiple companies."""
        if not self.agent:
            return False # Agent disabled
        if len(companies) > 1:
            print("Routing to agent: Multiple companies requested.")
            return True
        query_lower = query.lower()
        for keyword in AGENT_KEYWORDS:
            if keyword in query_lower:
                print(f"Routing to agent: Keyword '{keyword}' found.")
                return True
        return False

    def _format_history_for_chain(self, chat_history: Optional[List[ChatMessage]]) -> list:
        """Formats Pydantic ChatMessage list to LangChain Message list."""
        if not chat_history:
            return []
        formatted = []
        for msg in chat_history:
            if msg.role == 'user':
                formatted.append(HumanMessage(content=msg.content))
            elif msg.role == 'assistant':
                formatted.append(AIMessage(content=msg.content))
        return formatted

    async def aprocess_query(self, query: str, companies: List[str], chat_history: Optional[List[ChatMessage]] = None) -> SecResponse:
        """
        Asynchronously processes a query, routing to RAG chain or Agent.
        Handles chat history for both routes.
        """
        print(f"Processing query: '{query}' for companies: {companies}")
        formatted_chat_history = self._format_history_for_chain(chat_history)

        if self._should_route_to_agent(query, companies):
            # Route to Agent
            if self.agent:
                 print("--- Routing to SECAgent ---")
                 try:
                     # Agent's aprocess_complex_query already handles history formatting internally
                     return await self.agent.aprocess_complex_query(query, companies, chat_history)
                 except Exception as e:
                     print(f"Error invoking agent: {e}")
                     raise HTTPException(status_code=500, detail=f"Agent failed: {e}")
            else:
                print("Agent routing intended but agent is not available. Falling back to RAG.")

        # Default to RAG Chain
        print("--- Routing to RAG Chain ---")
        try:
            # Need retrieved docs for citations
            # The retrieval context doesn't use history directly, but the LLM call does
            retrieved_docs = await self.vector_store.search(query=query, companies=companies, limit=3)
            if not retrieved_docs:
                # Check history? Or just return no info?
                # For now, stick to returning based on retrieved docs.
                return SecResponse(answer="Could not find relevant information in the available filings for this specific query.", citations=[])

            # Invoke RAG chain - Pass the combined input dictionary
            chain_input = {
                "question": query,
                "companies": companies,
                "chat_history": formatted_chat_history
            }
            answer = await self.rag_chain.ainvoke(chain_input)

            # Extract citations using the retrieved documents
            citations = self._extract_rag_citations(retrieved_docs)

            # Generate contextually relevant suggested queries
            suggested_queries = await self._generate_suggested_queries(
                query=query, 
                answer=answer, 
                companies=companies, 
                retrieved_docs=retrieved_docs
            )

            return SecResponse(
                answer=answer,
                citations=citations,
                suggested_queries=suggested_queries
            )
        except Exception as e:
            print(f"Error in RAG chain: {e}")
            raise HTTPException(status_code=500, detail=f"RAG processing failed: {e}")

    async def _generate_suggested_queries(self, query: str, answer: str, companies: List[str], retrieved_docs: List[Dict[str, Any]]) -> List[str]:
        """
        Generate contextually relevant suggested follow-up queries based on the 
        current query, answer, and retrieved documents.
        """
        import random
        from collections import Counter
        
        # Extract topics from retrieved docs and answer
        all_text = answer + " " + " ".join([doc.get('content', '') for doc in retrieved_docs])
        all_text = all_text.lower()
        
        # Content-based topic detection
        topic_keywords = {
            'revenue': ['revenue', 'sales', 'income', 'earnings', 'profit'],
            'growth': ['growth', 'increase', 'expand', 'growing', 'grew'],
            'risks': ['risk', 'threat', 'challenge', 'uncertainty', 'potential issues'],
            'competition': ['competition', 'competitor', 'market share', 'industry', 'rival'],
            'strategy': ['strategy', 'plan', 'approach', 'initiative', 'objective'],
            'products': ['product', 'service', 'offering', 'solution', 'device'],
            'innovation': ['innovation', 'research', 'development', 'r&d', 'technology'],
            'financial': ['financial', 'balance sheet', 'cash flow', 'assets', 'liabilities'],
            'management': ['management', 'executive', 'leadership', 'ceo', 'officer'],
            'outlook': ['outlook', 'forecast', 'future', 'guidance', 'projection']
        }
        
        # Detect topics in the content
        detected_topics = []
        for topic, keywords in topic_keywords.items():
            for keyword in keywords:
                if keyword in all_text:
                    detected_topics.append(topic)
                    break
        
        # Get metadata from documents
        sections = set()
        detected_companies = set()
        years_mentioned = set()
        
        # Extract metadata from docs
        for doc in retrieved_docs:
            metadata = doc.get('metadata', {})
            # Extract section types
            section = metadata.get('section', '')
            if section:
                sections.add(section)
            
            # Extract companies 
            company = metadata.get('company', '')
            ticker = metadata.get('ticker', '')
            if company:
                detected_companies.add(company)
            if ticker:
                detected_companies.add(ticker)
            
            # Extract dates/years
            date = metadata.get('date', '')
            if date and len(date) >= 4:
                # Try to extract year from date string
                try:
                    year = int(date[:4])
                    if 1990 <= year <= 2030:  # Reasonable year range
                        years_mentioned.add(year)
                except:
                    pass
            
            # Look for years in content
            content = doc.get('content', '').lower()
            import re
            year_pattern = r'\b(19|20)\d{2}\b'
            years = re.findall(year_pattern, content)
            for year in years:
                try:
                    years_mentioned.add(int(year))
                except:
                    pass
        
        # Pattern detection in query
        query_patterns = {
            'comparison': any(term in query.lower() for term in ['compare', 'vs', 'versus', 'difference between']),
            'trend': any(term in query.lower() for term in ['trend', 'over time', 'change in', 'historical']),
            'recommendation': any(term in query.lower() for term in ['invest', 'buy', 'sell', 'recommend', 'should i']),
            'explanation': any(term in query.lower() for term in ['why', 'how', 'explain', 'what is', 'define']),
            'listing': any(term in query.lower() for term in ['list', 'what are', 'enumerate', 'examples of'])
        }
        
        # Select target companies for suggestions
        target_companies = companies or list(detected_companies)
        if not target_companies and random.random() < 0.7:  # 70% chance to add a company if none detected
            potential_companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            target_companies = [random.choice(potential_companies)]
        
        # Base suggestions on detected patterns
        suggestions = []
        
        # If we detected a comparison query, suggest more specific comparisons
        if query_patterns['comparison'] and len(target_companies) >= 2:
            aspects = ["revenue growth", "profit margins", "business strategy", 
                      "research and development investments", "market position"]
            aspect = random.choice(aspects)
            companies_str = " and ".join(target_companies[:2])
            suggestions.append(f"How do {companies_str} compare in terms of {aspect}?")
        
        # If we detected an investment/recommendation query, suggest follow-ups
        if query_patterns['recommendation']:
            if target_companies:
                company = target_companies[0]
                suggestions.append(f"What are the biggest risks for investing in {company}?")
                if len(suggestions) < 3:
                    suggestions.append(f"What competitive advantages does {company} have?")
        
        # If we detected trends, suggest more specific trend questions
        if query_patterns['trend'] or 'growth' in detected_topics:
            if target_companies:
                company = target_companies[0]
                metrics = ["revenue", "profit margin", "market share", "R&D spending", "employee headcount"]
                metric = random.choice(metrics)
                suggestions.append(f"What has been the trend in {company}'s {metric} over the past 3 years?")
        
        # Add section-specific questions
        if 'Risk Factors' in sections and len(suggestions) < 3:
            company_ref = target_companies[0] if target_companies else "the company"
            suggestions.append(f"What strategies does {company_ref} have to mitigate these risks?")
        
        if 'MD&A' in sections and len(suggestions) < 3:
            company_ref = target_companies[0] if target_companies else "the company"
            suggestions.append(f"What does management say about future growth opportunities for {company_ref}?")
        
        # Add topic-based suggestions
        topic_to_question = {
            'revenue': "What are the main sources of revenue?",
            'products': "What new products or services are being developed?",
            'innovation': "How much does the company invest in R&D?",
            'competition': "Who are the main competitors and what's the competitive landscape?",
            'risks': "What are the top 3 risk factors mentioned?",
            'strategy': "What is the long-term business strategy?",
            'management': "Have there been any recent changes in executive leadership?",
            'financial': "How strong is the balance sheet and cash position?",
            'outlook': "What is the company's guidance for the next fiscal year?"
        }
        
        # Add up to 2 topic-based questions
        added_topics = set()
        for topic in random.sample(detected_topics, min(len(detected_topics), 4)):
            if len(suggestions) >= 3:
                break
            
            if topic in topic_to_question and topic not in added_topics:
                company_ref = target_companies[0] if target_companies else "the company"
                suggestions.append(f"{topic_to_question[topic].replace('the company', company_ref)}")
                added_topics.add(topic)
        
        # Add a comparison question if we have only one company
        if len(target_companies) == 1 and len(suggestions) < 3 and not query_patterns['comparison']:
            other_companies = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
            if target_companies[0] in other_companies:
                other_companies.remove(target_companies[0])
            compare_company = random.choice(other_companies)
            
            # Pick a basis for comparison
            comparison_bases = ["business model", "growth strategy", "financial performance", "market position"]
            if detected_topics:
                # Try to base comparison on a detected topic
                if 'revenue' in detected_topics:
                    comparison_basis = "revenue sources"
                elif 'innovation' in detected_topics:
                    comparison_basis = "R&D investments"
                elif 'risks' in detected_topics:
                    comparison_basis = "risk factors"
                else:
                    comparison_basis = random.choice(comparison_bases)
            else:
                comparison_basis = random.choice(comparison_bases)
            
            suggestions.append(f"How does {target_companies[0]} compare to {compare_company} in terms of {comparison_basis}?")
        
        # If we still need more suggestions, add generic but useful questions
        if len(suggestions) < 2:
            generic_questions = [
                "What significant changes have occurred in the latest quarterly report?",
                "What are the key growth drivers mentioned in recent filings?",
                "How does the company allocate its capital?",
                "What risks does the company face from regulatory changes?",
                "How has the company's market position changed in recent years?"
            ]
            
            while len(suggestions) < 3 and generic_questions:
                question = generic_questions.pop(random.randrange(len(generic_questions)))
                if target_companies:
                    company_ref = target_companies[0]
                    question = question.replace("the company", company_ref)
                suggestions.append(question)
        
        # Ensure all suggestions are unique and we have at most 3
        suggestions = list(dict.fromkeys(suggestions))  # Remove duplicates while preserving order
        return suggestions[:3]

# Removed previous _extract_citations and _retrieve_context specific variable names
# Renamed to _extract_rag_citations and _retrieve_rag_context for clarity