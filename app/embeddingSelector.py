from llama_index.core.selectors import BaseSelector, SelectorResult, SingleSelection
import logging

class EmbeddingAwareSelector(BaseSelector):
    """Custom selector that uses document embeddings to make informed decisions"""
    
    def __init__(self, llm, vector_index, embed_model):
        self.llm = llm
        self.vector_index = vector_index
        self.embed_model = embed_model
    
    def _select(self, choices, query) -> SelectorResult:
        # Get top similar nodes to understand document relevance
        retriever = self.vector_index.as_retriever(similarity_top_k=3)
        relevant_nodes = retriever.retrieve(str(query))
        
        # Calculate relevance score
        if relevant_nodes:
            max_score = max(node.score for node in relevant_nodes)
        else:
            max_score = 0
        logging.info(f"Document relevance: {max_score}")

        # Use LLM to make final decision with document context
        context = f"Query: {query}\n"
        context += f"Document relevance: {max_score}\n"
        if relevant_nodes:
            context += f"Top relevant content: {relevant_nodes[0].text[:200]}...\n"
        
        context += "\nAvailable tools:\n"
        for i, choice in enumerate(choices):
            context += f"{i}: {choice.name} - {choice.description}\n"
        
        prompt = f"""
        Based on the query and document context, select the most appropriate tool:
        
        {context}
        
        Rules:
        - If query is moderately relevant to document and asks for overview/summary → summary_tool
        - If query is highly relevant (relevance score > 0.3) to the given content and asks for specific details → search_tool  
        - If query is NOT relevant (relevance score < 0.15) to document → action_tool
        
        Return only the tool index (0, 1, or 2) without any other text.
        """
        
        response = self.llm.complete(prompt)
        logging.info(f"Response: {response.text}")

        try:
            selected_index = int(response.text.strip())
            return SelectorResult(
                selections=[SingleSelection(index=selected_index, reason=f"Selected based on document relevance: {max_score}")]
            )
        except:
            # Fallback to default selection
            return SelectorResult(
                selections=[SingleSelection(index=1, reason="Fallback to search tool")]
            )
    
    def _aselect(self, choices, query) -> SelectorResult:
        # Async version - same as sync for now
        return self._select(choices, query)
    
    def _get_prompts(self) -> dict:
        # Return empty dict for now
        return {}
    
    def _update_prompts(self, prompts: dict) -> None:
        # No-op for now
        pass