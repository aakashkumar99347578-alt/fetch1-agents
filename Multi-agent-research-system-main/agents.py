import os
import sys
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url

load_dotenv()

# Optional OpenTelemetry / OpenInference setup (failsafe if not installed)
def _setup_telemetry():
    if os.getenv("ENABLE_TRACING", "").lower() in ("true", "1"):
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from openinference.instrumentation.langchain import LangChainInstrumentor
            from opentelemetry.sdk.resources import Resource

            provider = TracerProvider(resource=Resource.create({"service.name": os.getenv("OTEL_SERVICE_NAME", "research-agents")}))
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if otlp_endpoint:
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(SimpleSpanProcessor(exporter))
            trace.set_tracer_provider(provider)
            LangChainInstrumentor().instrument()
        except Exception:
            pass

_setup_telemetry()

# Model setup: Use an active, supported Gemini model (gemini-3.5-flash, gemini-3.5-flash-lite, or gemini-3.8-flash)
model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.7)

def extract_text_content(content) -> str:
    """Helper to extract clean string content whether returned as str, list of blocks, or AIMessage."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if hasattr(content, "content"):
        return extract_text_content(content.content)
    return str(content)

# 1st agent: Search Agent
def build_search_agent():
    return create_react_agent(
        model=llm,
        tools=[web_search]
    )

# 2nd agent: Reader Agent
def build_reader_agent():
    return create_react_agent(
        model=llm,
        tools=[scrape_url]
    )

# 3rd component: Writer Chain
writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# 4th component: Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

