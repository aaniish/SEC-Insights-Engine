from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import asyncio

class VectorStore:
    """
    Handles storage and retrieval of document embeddings using ChromaDB and SentenceTransformers.
    """
    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "sec_filings", embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the VectorStore.

        Args:
            persist_directory: Directory to store ChromaDB data.
            collection_name: Name of the ChromaDB collection.
            embedding_model_name: Name of the SentenceTransformer model.
        """
        print(f"Initializing VectorStore with model: {embedding_model_name}")
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection = self.client.get_or_create_collection(collection_name)
        print(f"VectorStore initialized. Collection '{collection_name}' loaded/created.")
    
    async def add_documents(self, documents: List[Dict[str, Any]]):
        """
        Asynchronously adds documents to the vector store after generating embeddings.
        
        Args:
            documents: List of document dictionaries. Each dict should have 'content'
                       and 'metadata' keys. Metadata should be a flat dict.
        """
        if not documents:
            return

        contents = [doc["content"] for doc in documents]
        metadatas = [doc["metadata"] for doc in documents]
        # Ensure IDs are unique and ChromaDB compatible (strings)
        ids = [f"{doc['metadata'].get('ticker', 'NA')}_{doc['metadata'].get('filing', 'NA')}_{i}" for i, doc in enumerate(documents)]

        print(f"Generating embeddings for {len(contents)} documents...")
        # Run embedding generation in a separate thread to avoid blocking the event loop
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(None, self.embedding_model.encode, contents)
        print("Embeddings generated.")

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(), # Convert numpy array to list
                metadatas=metadatas,
                documents=contents # Store the original text content as well
            )
            print(f"Added {len(documents)} documents to collection '{self.collection.name}'.")
        except Exception as e:
            print(f"Error adding documents to ChromaDB: {e}")
            # Consider more specific error handling or re-raising

    async def search(self, query: str, companies: List[str] = None, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Search the vector store for relevant document chunks based on the query and optional company filters.
        
        Args:
            query: The natural language query string
            companies: Optional list of company tickers to filter results (e.g., ["AAPL", "MSFT"])
            limit: Maximum number of results to return
            
        Returns:
            List of document chunks with content and metadata
        """
        if not query:
            return []
        
        # Print search params for debugging
        print(f"Searching for: '{query}' with company filter: {companies}, limit: {limit}")
        
        try:
            # Get vector embedding for the query
            query_embedding = await self._get_embedding(query)
            
            # Initialize search parameters
            search_params = {"k": limit * 3}  # Get more results initially for post-filtering
            
            # Use company filter if provided
            where_filter = None
            if companies and len(companies) > 0:
                where_filter = {"ticker": {"$in": companies}}
                print(f"Applying company filter: {where_filter}")
            
            # Execute the search with the appropriate filter
            results = self.collection.query(
                query_embeddings=query_embedding,  # Already a list from _get_embedding
                n_results=search_params["k"],
                where=where_filter,
                include=["documents", "metadatas"]
            )
            
            # Extract and format the results
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            
            if not documents or not metadatas:
                print("No results found in vector store")
                return []
            
            # Format the results
            formatted_results = []
            for i, (doc, meta) in enumerate(zip(documents, metadatas)):
                formatted_results.append({
                    "content": doc,
                    "metadata": meta
                })
            
            # Additional logging to debug results
            if formatted_results:
                print(f"Found {len(formatted_results)} results. First result from: {formatted_results[0]['metadata'].get('company', 'Unknown')}")
            
            # Return the top 'limit' results
            return formatted_results[:limit]
            
        except Exception as e:
            print(f"Error during vector search: {e}")
            import traceback
            print(traceback.format_exc())
            return []

    async def _get_embedding(self, query: str):
        """
        Generates an embedding for a given query using the SentenceTransformer model.
        
        Args:
            query: The natural language query string
            
        Returns:
            Embedding for the query
        """
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(None, self.embedding_model.encode, [query])
        # Convert numpy array to list to ensure compatibility with ChromaDB
        return embedding.tolist()