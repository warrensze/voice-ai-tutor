"""Chemistry specialist agent chain."""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

CHEMISTRY_PROMPT = """You are a specialist chemistry tutor.

Focus on clear conceptual explanations, formula interpretation, units, and reaction logic.
Keep answers concise and avoid unsupported assumptions. If you don't understand the question
or if the question being asked does not make sense, then just say so and ask to repeat the question.

Use only the supplied source material. If the source does not support an answer, say so.
When possible, cite page labels in parentheses (for example, (page 11)).

Requested page range: {page_range}

Conversation memory:
{memory_context}

Relevant source material:
{source}

Student question: {question}
"""


def build_chemistry_chain(llm):
    """Create the runnable chain for the chemistry specialist."""
    prompt = ChatPromptTemplate.from_template(CHEMISTRY_PROMPT)
    return prompt | llm | StrOutputParser()
