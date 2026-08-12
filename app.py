import json
import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    force=True,
)
logger = logging.getLogger("llm_trace")

# Approx $/1M tokens for estimate only — override via env if needed.
# Hosted NIM free tier may bill $0; treat these as budgeting estimates.
DEFAULT_PRICING_PER_M = {
    "meta/llama-3.3-70b-instruct": {"input": 0.60, "output": 1.80},
    "meta/llama-3.1-70b-instruct": {"input": 0.60, "output": 1.80},
    "meta/llama-3.1-8b-instruct": {"input": 0.05, "output": 0.15},
}

def _secret(key: str, default: str | None = None) -> str | None:
    value = os.getenv(key)
    if value:
        return value
    try:
        return st.secrets.get(key, default)
    except Exception:
        return default

NVIDIA_API_KEY = _secret("NVIDIA_API_KEY")
TAVILY_API_KEY = _secret("TAVILY_API_KEY")
MODEL_NAME = _secret("NIM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
NIM_BASE_URL = _secret("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
MAX_OUTPUT_TOKENS = int(_secret("NIM_MAX_OUTPUT_TOKENS", "2048") or 2048)
SNIPPET_CHARS = int(_secret("TAVILY_SNIPPET_CHARS", "400") or 400)
_raw_input_price = _secret("NIM_INPUT_PRICE_PER_M")
_raw_output_price = _secret("NIM_OUTPUT_PRICE_PER_M")
INPUT_PRICE_PER_M = float(_raw_input_price) if _raw_input_price is not None else None
OUTPUT_PRICE_PER_M = float(_raw_output_price) if _raw_output_price is not None else None
if not NVIDIA_API_KEY:
    st.error("Missing NVIDIA_API_KEY. Set it in environment or Streamlit secrets.")
    st.stop()

if not TAVILY_API_KEY:
    st.error("Missing TAVILY_API_KEY. Set it in environment or Streamlit secrets.")
    st.stop()

nim_client = OpenAI(base_url=NIM_BASE_URL, api_key=NVIDIA_API_KEY)
tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

st.set_page_config(page_title="AI News Generator", page_icon="📰", layout="wide")

st.title("🤖 AI News Generator, powered by NVIDIA NIM + Tavily")
st.markdown("Generate comprehensive blog posts about any topic using AI agents.")

with st.sidebar:
    st.header("Content Settings")

    topic = st.text_area(
        "Enter your topic",
        height=100,
        placeholder="Enter the topic you want to generate content about..."
    )

    st.markdown("### Advanced Settings")
    model_options = {
        "Fast — Llama 3.1 8B": "meta/llama-3.1-8b-instruct",
        "Balanced — Llama 3.3 70B": "meta/llama-3.3-70b-instruct",
    }
    default_model_index = next(
        (i for i, mid in enumerate(model_options.values()) if mid == MODEL_NAME),
        0,
    )
    model_label = st.selectbox("Model", list(model_options.keys()), index=default_model_index)
    selected_model = model_options[model_label]

    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_results = st.slider("Search results", 3, 8, 4)
    search_depth = st.selectbox("Search depth", ["basic", "advanced"], index=0)
    max_tokens = st.slider("Max output tokens", 512, 4096, min(MAX_OUTPUT_TOKENS, 2048), step=256)

    st.markdown("---")

    generate_button = st.button("Generate Content", type="primary", use_container_width=True)

    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Enter your desired topic in the text area above
        2. Adjust the temperature if needed (higher = more creative)
        3. Click 'Generate Content' to start
        4. Wait for the AI to generate your article
        5. Download the result as a markdown file
        """)

def _pricing_for(model: str) -> tuple[float | None, float | None]:
    if INPUT_PRICE_PER_M is not None and OUTPUT_PRICE_PER_M is not None:
        return INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M
    rates = DEFAULT_PRICING_PER_M.get(model)
    if not rates:
        return None, None
    return rates["input"], rates["output"]

def estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    input_rate, output_rate = _pricing_for(model)
    if input_rate is None or output_rate is None:
        return None
    return (prompt_tokens / 1_000_000) * input_rate + (completion_tokens / 1_000_000) * output_rate

def search_sources(topic: str, max_results: int, search_depth: str, snippet_chars: int) -> str:
    response = tavily_client.search(
        query=topic,
        max_results=max_results,
        search_depth=search_depth,
        include_answer=False,
    )
    results = response.get("results", [])
    if not results:
        return "No search results found."

    lines = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")[:snippet_chars]
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {content}")
    return "\n\n".join(lines)

@st.cache_data(ttl=300, show_spinner=False)
def cached_search_sources(topic: str, max_results: int, search_depth: str, snippet_chars: int) -> str:
    return search_sources(topic, max_results, search_depth, snippet_chars)

def build_trace(
    model: str,
    content: str,
    latency_ms: float,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str | None,
    response_id: str | None,
) -> dict:
    total_tokens = prompt_tokens + completion_tokens
    cost = estimate_cost_usd(model, prompt_tokens, completion_tokens)
    return {
        "model": model,
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 6) if cost is not None else None,
        "finish_reason": finish_reason,
        "response_id": response_id,
        "response_preview": content[:500],
    }

def stream_llm(
    messages: list[dict],
    temperature: float,
    model: str,
    max_tokens: int,
    trace_out: dict,
):
    request_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    logger.info("LLM REQUEST\n%s", json.dumps({**request_payload, "stream": True}, indent=2, ensure_ascii=False))

    started = time.perf_counter()
    stream = nim_client.chat.completions.create(**request_payload)

    chunks: list[str] = []
    finish_reason = None
    response_id = None
    prompt_tokens = 0
    completion_tokens = 0

    for chunk in stream:
        if chunk.id:
            response_id = chunk.id
        if chunk.usage:
            prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
            completion_tokens = chunk.usage.completion_tokens or completion_tokens
        if not chunk.choices:
            continue
        choice = chunk.choices[0]
        if choice.finish_reason:
            finish_reason = choice.finish_reason
        delta = choice.delta.content
        if delta:
            chunks.append(delta)
            yield delta

    content = "".join(chunks)
    latency_ms = (time.perf_counter() - started) * 1000
    trace = build_trace(model, content, latency_ms, prompt_tokens, completion_tokens, finish_reason, response_id)
    trace_out["trace"] = trace
    logger.info("LLM RESPONSE META\n%s", json.dumps(trace, indent=2, ensure_ascii=False))
    logger.info("LLM RESPONSE BODY\n%s", content)

def generate_content(
    topic: str,
    temperature: float,
    max_results: int,
    search_depth: str,
    model: str,
    max_tokens: int,
    snippet_chars: int,
) -> tuple[str, dict]:
    if not topic.strip():
        raise ValueError("Please enter a topic before generating content.")

    sources = cached_search_sources(topic.strip(), max_results, search_depth, snippet_chars)

    prompt = f"""Write a markdown blog post about: {topic}

Sources (use only these; cite as [Source: URL]):
{sources}

Include: H1 title, executive summary, 3-4 H3 sections, conclusion, References list.
Be factual, concise, and engaging."""

    trace_out: dict = {}

    def _stream():
        yield from stream_llm(
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            model=model,
            max_tokens=max_tokens,
            trace_out=trace_out,
        )

    return _stream, trace_out

if generate_button:
    try:
        with st.status("Searching the web...", expanded=False) as search_status:
            stream_fn, trace_out = generate_content(
                topic, temperature, max_results, search_depth, selected_model, max_tokens, SNIPPET_CHARS
            )
            search_status.update(label="Search complete", state="complete")

        st.markdown("### Generated Content")
        gen_loader = st.status("Generating article with NVIDIA NIM...", state="running", expanded=False)

        def stream_with_loader():
            first_chunk = True
            for chunk in stream_fn():
                if first_chunk:
                    gen_loader.update(label="Article generated", state="complete")
                    first_chunk = False
                yield chunk

        result = st.write_stream(stream_with_loader)
        trace = trace_out.get("trace", {})

        with st.expander("LLM trace (tokens / cost)"):
            st.json(trace)

        st.download_button(
            label="Download Content",
            data=result,
            file_name=f"{topic.lower().replace(' ', '_')}_article.md",
            mime="text/markdown",
        )
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
st.markdown("---")
st.markdown("Built with Streamlit, NVIDIA NIM, and Tavily")
