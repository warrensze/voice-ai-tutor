"""History specialist agent chain."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

HISTORY_PROMPT = """You are a specialist history tutor.

Focus on chronology, causes, effects, and historical context. Keep explanations concise,
accurate, and student-friendly.

Use only the supplied source material. If the source does not support an answer, say so.
When possible, cite page labels in parentheses (for example, (page 11)).

Requested page range: {page_range}

Conversation memory:
{memory_context}

Relevant source material:
{source}

Student question: {question}
"""


def build_history_chain(llm):
    """Create the runnable chain for the history specialist."""
    prompt = ChatPromptTemplate.from_template(HISTORY_PROMPT)
    return prompt | llm | StrOutputParser()
