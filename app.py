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
MODEL_NAME = _secret("NIM_MODEL_NAME", "meta/llama-3.3-70b-instruct")
NIM_BASE_URL = _secret("NIM_BASE_URL", "https://integrate.api.nvidia.com/v1")
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
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_results = st.slider("Search results", 3, 10, 5)

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

def search_sources(topic: str, max_results: int) -> str:
    response = tavily_client.search(
        query=topic,
        max_results=max_results,
        search_depth="advanced",
        include_answer=False,
    )
    results = response.get("results", [])
    if not results:
        return "No search results found."

    lines = []
    for i, item in enumerate(results, start=1):
        title = item.get("title", "Untitled")
        url = item.get("url", "")
        content = item.get("content", "")
        lines.append(f"[{i}] {title}\nURL: {url}\nSnippet: {content}")
    return "\n\n".join(lines)

def call_llm(messages: list[dict], temperature: float) -> tuple[str, dict]:
    request_payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": temperature,
    }
    logger.info("LLM REQUEST\n%s", json.dumps(request_payload, indent=2, ensure_ascii=False))

    started = time.perf_counter()
    response = nim_client.chat.completions.create(**request_payload)
    latency_ms = (time.perf_counter() - started) * 1000

    content = response.choices[0].message.content or ""
    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0
    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)
    cost = estimate_cost_usd(MODEL_NAME, prompt_tokens, completion_tokens)

    trace = {
        "model": MODEL_NAME,
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "estimated_cost_usd": round(cost, 6) if cost is not None else None,
        "finish_reason": response.choices[0].finish_reason,
        "response_id": getattr(response, "id", None),
        "response_preview": content[:500],
    }
    logger.info("LLM RESPONSE META\n%s", json.dumps(trace, indent=2, ensure_ascii=False))
    logger.info("LLM RESPONSE BODY\n%s", content)
    return content, trace

def generate_content(topic: str, temperature: float, max_results: int) -> tuple[str, dict]:
    if not topic.strip():
        raise ValueError("Please enter a topic before generating content.")

    sources = search_sources(topic, max_results)

    prompt = f"""
You are a senior research analyst and content writer.
Create a comprehensive markdown blog post about: {topic}

Use ONLY the search results below as your factual basis. Do not invent sources.

Search results:
{sources}

Requirements:
1. Include:
   - An attention-grabbing H1 title
   - Executive summary
   - Well-structured sections with H3 headings
   - Recent developments, trends, expert insights, and key statistics
   - A concise conclusion
2. Cite claims inline using [Source: URL].
3. End with a References section listing all unique source URLs used.
4. Keep the writing engaging but factual and precise.
"""

    return call_llm(
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )

if generate_button:
    with st.spinner("Generating content... This may take a moment."):
        try:
            result, trace = generate_content(topic, temperature, max_results)
            st.markdown("### Generated Content")
            st.markdown(result)

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
