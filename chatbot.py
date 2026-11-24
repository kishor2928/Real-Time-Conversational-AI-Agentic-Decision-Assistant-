"""
Core conversational agent.
Uses LangChain + Ollama with optional tool calling and RAG context injection.
"""

from typing import List, Dict, Any, Optional, Tuple
import time
import logging

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.config import get_settings
from app.agent.tools import AVAILABLE_TOOLS
from app.agent.memory import memory
from app.rag.retriever import retrieve_context, format_docs
from app.schemas import ToolCallResult

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a helpful decision-support assistant.
You have access to tools and a knowledge base.
Be concise, practical, and honest when you don't know something.
When using tools, explain the result in plain language for the user.
If RAG context is provided, prefer it over inventing facts.
"""


class DecisionAssistant:
    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOllama(
            model=self.settings.ollama_model,
            base_url=self.settings.ollama_base_url,
            temperature=0.3,
        )
        self.llm_with_tools = self.llm.bind_tools(AVAILABLE_TOOLS)
        self.tools_by_name = {t.name: t for t in AVAILABLE_TOOLS}

    def _build_messages(
        self,
        history: List[Dict[str, str]],
        user_message: str,
        rag_context: str = "",
    ) -> List:
        messages = [SystemMessage(content=SYSTEM_PROMPT)]

        if rag_context:
            messages.append(
                SystemMessage(
                    content=f"Relevant knowledge base excerpts:\n\n{rag_context}"
                )
            )

        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))

        messages.append(HumanMessage(content=user_message))
        return messages

    def _run_tools(self, tool_calls: List[Any]) -> Tuple[List[ToolMessage], List[ToolCallResult]]:
        tool_messages = []
        results = []

        for tc in tool_calls:
            name = tc["name"]
            args = tc["args"]
            tool_id = tc.get("id", name)

            if name not in self.tools_by_name:
                content = f"Unknown tool: {name}"
                success = False
            else:
                try:
                    content = self.tools_by_name[name].invoke(args)
                    success = True
                except Exception as e:
                    content = f"Tool error: {str(e)}"
                    success = False

            tool_messages.append(
                ToolMessage(content=str(content), tool_call_id=tool_id)
            )
            results.append(
                ToolCallResult(
                    tool_name=name,
                    arguments=args,
                    result=content,
                    success=success,
                )
            )
        return tool_messages, results

    def chat(
        self,
        message: str,
        session_id: Optional[str] = None,
        use_tools: bool = True,
        use_rag: bool = True,
    ) -> Dict[str, Any]:
        start = time.perf_counter()
        session_id = memory.get_or_create_session(session_id)
        history = memory.get_history(session_id)

        # RAG
        sources = []
        rag_context = ""
        if use_rag:
            docs = retrieve_context(message)
            rag_context = format_docs(docs)
            sources = [d.metadata.get("source", "unknown") for d in docs]

        messages = self._build_messages(history, message, rag_context)

        tool_results: List[ToolCallResult] = []
        final_content = ""

        try:
            if use_tools:
                # first pass – may request tools
                ai_msg = self.llm_with_tools.invoke(messages)
                if ai_msg.tool_calls:
                    tool_msgs, tool_results = self._run_tools(ai_msg.tool_calls)
                    messages.append(ai_msg)
                    messages.extend(tool_msgs)
                    # second pass with tool results
                    ai_msg = self.llm.invoke(messages)

                final_content = ai_msg.content or ""
            else:
                ai_msg = self.llm.invoke(messages)
                final_content = ai_msg.content or ""

        except Exception as e:
            logger.exception("LLM call failed")
            final_content = (
                "Sorry, I ran into a problem talking to the local model. "
                "Is Ollama running and the model pulled?"
            )

        # persist
        memory.add_message(session_id, "user", message)
        memory.add_message(session_id, "assistant", final_content)

        latency = (time.perf_counter() - start) * 1000

        return {
            "response": final_content,
            "session_id": session_id,
            "tool_calls": tool_results,
            "sources": list(set(sources)),
            "latency_ms": round(latency, 1),
        }


# singleton
assistant = DecisionAssistant()
