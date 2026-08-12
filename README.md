# AI News generator

Streamlit app that generates grounded blog posts using open source LLM models from NVIDIA and **Tavily** for web search.

## Setup

1. Copy `.env.example` to `.env` and fill in:
   - `NVIDIA_API_KEY` — from [NVIDIA API Catalog](https://build.nvidia.com/)
   - `TAVILY_API_KEY` — from [Tavily](https://tavily.com/)
   - Optional: `NIM_MODEL_NAME` (default `meta/llama-3.3-70b-instruct`)

2. Install and run:
   ```bash
   pip install -r requirements.txt
   streamlit run app.py
   ```

## Deploy on Streamlit Community Cloud

In **Settings > Secrets**, add:

```toml
NVIDIA_API_KEY="nvapi-..."
TAVILY_API_KEY="tvly-..."
NIM_MODEL_NAME="meta/llama-3.3-70b-instruct"
```

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
