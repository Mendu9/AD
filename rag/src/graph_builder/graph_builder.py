"""Graph builder for LangGraph"""

from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.node.reactnode import RAGNodes

class GraphBuilder:
    """Builds and manages the LangGraph workflow"""
    
    def __init__(self, retriever, llm):
        """
        Initialize graph builder
        
        Args:
            retriever: Document retriever instance
            llm: Language model instance
        """
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None
    
    def build(self):
        """
        Building the RAG workflow graph
        """
        builder = StateGraph(RAGState)
        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)
        builder.set_entry_point("retriever")
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)
        self.graph = builder.compile()
        return self.graph
    
    def run(self, question: str) -> dict:
        """
        Run the RAG workflow
        Args
            question: User question
        Returns:
            Final state with answer
        """
        if self.graph is None:
            self.build()
        
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)