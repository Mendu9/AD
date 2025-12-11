"""LangGraph nodes for RAG workflow + ReAct Agent inside generate_content"""

from typing import List, Optional, Dict, Any
from src.state.rag_state import RAGState

from langchain_core.documents import Document
from langchain_core.tools import Tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import os
from dotenv import load_dotenv
load_dotenv()
# Wikipedia tool
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from langchain_community.tools.pubmed.tool import PubmedQueryRun
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")


class RAGNodes:
    """Contains node functions for RAG workflow"""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self._agent = None  # lazy-init agent
        self.tool_sources: List[Dict[str, Any]] = []

    def retrieve_docs(self, state: RAGState) -> RAGState:
        """Classic retriever node"""
        docs = self.retriever.invoke(state.question)
        return RAGState(
            question=state.question,
            retrieved_docs=docs,
            answer=state.answer,
            tool_sources=getattr(state, "tool_sources", []),
        )


    def _log_and_return(self, *, tool_name: str, source_type: str, query: str, result: Any):

        if isinstance(result, str):
            snippet = result[:500]
        else:
            snippet = str(result)[:500]

        self.tool_sources.append(
            {
                "tool": tool_name,
                "source_type": source_type,
                "query": query,
                "snippet": snippet,
            }
        )
        return result
    
    def _build_tools(self) -> List[Tool]:
        """Build retriever + wikipedia tools"""

        def retriever_tool_fn(query: str) -> str:
            docs: List[Document] = self.retriever.invoke(query)
            if not docs:
                return "No documents found."
            merged = []
            for i, d in enumerate(docs[:8], start=1):
                meta = d.metadata if hasattr(d, "metadata") else {}
                title = meta.get("title") or meta.get("source") or f"doc_{i}"
                merged.append(f"[{i}] {title}\n{d.page_content}")
            text = "\n\n".join(merged)
            return self._log_and_return(
                tool_name="retriever",
                source_type="local_vectorstore",
                query=query,
                result=text,
            )

        retriever_tool = Tool(
            name="retriever",
            description="Fetch passages from indexed corpus.",
            func=retriever_tool_fn,
        )
        
        # wiki
        wiki_api = WikipediaAPIWrapper(top_k_results=3, lang="en")
        wiki = WikipediaQueryRun(api_wrapper=wiki_api)

        def wikipedia_tool_fn(query: str) -> str:
            result = wiki.run(query)
            return self._log_and_return(
                tool_name="wikipedia",
                source_type="wikipedia",
                query=query,
                result=result,
            )

        wikipedia_tool = Tool(
            name="wikipedia",
            description="Search Wikipedia for general medical / general knowledge.",
            func=wikipedia_tool_fn,
        )

        # Arxiv
        arxiv_api = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=2000)
        arxiv = ArxivQueryRun(api_wrapper=arxiv_api)

        def arxiv_tool_fn(query: str) -> str:
            result = arxiv.run(query)
            return self._log_and_return(
                tool_name="arxiv",
                source_type="arxiv",
                query=query,
                result=result,
            )

        arxiv_tool = Tool(
            name="arxiv",
            description="Search arXiv for research papers (neuroimaging, biomarkers, etc.).",
            func=arxiv_tool_fn,
        )

        # PubMed
        pubmed = PubmedQueryRun()

        def pubmed_tool_fn(query: str) -> str:
            result = pubmed.run(query)
            return self._log_and_return(
                tool_name="pubmed",
                source_type="pubmed",
                query=query,
                result=result,
            )

        pubmed_tool = Tool(
            name="pubmed",
            description="Search PubMed for medical literature.",
            func=pubmed_tool_fn,
        )

        tavily_client=TavilySearch(max_results=5)
        
        def tavily_tool_fn(query: str):
            result = tavily_client.invoke(query)
            return self._log_and_return(
                tool_name="tavily",
                source_type="web_search",
                query=query,
                result=result,
            )

        tavily_tool = Tool(
            name="tavily_search",
            description="Focused web search for up-to-date information.",
            func=tavily_tool_fn,
        )
        
        
        return [retriever_tool, wikipedia_tool,arxiv_tool,pubmed_tool,tavily_tool,]

    def _build_agent(self):
        """ReAct agent with tools"""
        tools = self._build_tools()
        system_prompt = (
            "You are a helpful RAG agent. "
            "Prefer 'retriever' for user-provided Alzheimer PDFs; "
            "use 'wikipedia' for general medical knowledge; "
            "use 'arxiv' for research papers; "
            "use 'pubmed' for medical literature; "
            "use 'tavily_tool' for general web search. "
            "Return only the final useful answer for the user."
        )
        self._agent = create_react_agent(self.llm, tools=tools,prompt=system_prompt)

    def generate_answer(self, state: RAGState) -> RAGState:
        """
        Generate answer using ReAct agent with retriever + wikipedia + tavily.
        """
        if self._agent is None:
            self._build_agent()
        self.tool_sources = []
        result = self._agent.invoke({"messages": [HumanMessage(content=state.question)]})

        messages = result.get("messages", [])
        answer: Optional[str] = None
        if messages:
            answer_msg = messages[-1]
            answer = getattr(answer_msg, "content", None)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer or "Could not generate answer.",
            tool_sources=self.tool_sources
        )
