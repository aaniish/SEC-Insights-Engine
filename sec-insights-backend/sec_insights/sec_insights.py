@app.post("/api/message")
async def process_message(body: dict = Body(...)):
    """
    Process an incoming message from the user.
    The message is passed to the RAG pipeline, which generates a response.
    """
    try:
        # Process the message
        query = body.get("message", "")
        selected_companies = body.get("selectedCompanies", [])
        stream = body.get("stream", False)
        
        # Get chat history from request
        chat_history = body.get("chatHistory", [])
        
        # Format the chat history if present
        formatted_chat_history = []
        if chat_history:
            for message in chat_history:
                try:
                    if isinstance(message, dict) and "content" in message and "role" in message:
                        formatted_chat_history.append({
                            "role": message["role"],
                            "content": message["content"]
                        })
                except Exception as e:
                    # Skip malformed messages
                    print(f"Error processing chat message: {e}")
                    continue
        
        log.info(f"Processing message: {query}")
        log.info(f"Selected companies: {selected_companies}")
        
        # Check if the message is empty
        if not query:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        # Process the message
        if stream:
            return StreamingResponse(
                rag_pipeline.process_streaming(
                    query=query, 
                    companies=selected_companies,
                    chat_history=formatted_chat_history
                ),
                media_type="text/event-stream"
            )
        else:
            return await rag_pipeline.process(
                query=query, 
                companies=selected_companies,
                chat_history=formatted_chat_history
            )
    except Exception as e:
        log.error(f"Error processing message: {e}")
        log.exception("Details:")
        raise HTTPException(status_code=500, detail="Internal Server Error") 