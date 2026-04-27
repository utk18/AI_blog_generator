import os
import json
import tempfile
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import GenerateContentConfig, GoogleSearch, HttpOptions, Tool

# Load environment variables
load_dotenv()

def _configure_google_credentials_from_secrets():
    """Support Streamlit Cloud secrets for service-account auth."""
    # Option 1: secrets section named [gcp_service_account]
    if "gcp_service_account" in st.secrets:
        service_account_info = dict(st.secrets["gcp_service_account"])
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(service_account_info, tmp_file)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name
        return

    # Option 2: one JSON string secret named GOOGLE_APPLICATION_CREDENTIALS_JSON
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in st.secrets:
        raw_json = st.secrets["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
        service_account_info = json.loads(raw_json)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp_file:
            json.dump(service_account_info, tmp_file)
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name

_configure_google_credentials_from_secrets()

PROJECT_ID = os.getenv("GCP_PROJECT_ID") or st.secrets.get("GCP_PROJECT_ID")
LOCATION = os.getenv("GCP_LOCATION") or st.secrets.get("GCP_LOCATION", "global")
MODEL_NAME = os.getenv("VERTEX_MODEL_NAME") or st.secrets.get("VERTEX_MODEL_NAME", "gemini-2.0-flash")

if not PROJECT_ID:
    st.error("Missing GCP_PROJECT_ID. Set it in environment or Streamlit secrets.")
    st.stop()

genai_client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION,
    http_options=HttpOptions(api_version="v1"),
)

# Streamlit page config
st.set_page_config(page_title="AI News Generator", page_icon="📰", layout="wide")

# Title and description
st.title("🤖 AI News Generator, powered by Google Vertex AI")
st.markdown("Generate comprehensive blog posts about any topic using AI agents.")

# Sidebar
with st.sidebar:
    st.header("Content Settings")
    
    # Make the text input take up more space
    topic = st.text_area(
        "Enter your topic",
        height=100,
        placeholder="Enter the topic you want to generate content about..."
    )
    
    # Add more sidebar controls if needed
    st.markdown("### Advanced Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    # Add some spacing
    st.markdown("---")
    
    # Make the generate button more prominent in the sidebar
    generate_button = st.button("Generate Content", type="primary", use_container_width=True)
    
    # Add some helpful information
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. Enter your desired topic in the text area above
        2. Adjust the temperature if needed (higher = more creative)
        3. Click 'Generate Content' to start
        4. Wait for the AI to generate your article
        5. Download the result as a markdown file
        """)

def generate_content(topic):
    if not topic.strip():
        raise ValueError("Please enter a topic before generating content.")

    prompt = f"""
You are a senior research analyst and content writer.
Create a comprehensive markdown blog post about: {topic}

Requirements:
1. Use Google Search grounding to gather recent, reliable sources.
2. Include:
   - An attention-grabbing H1 title
   - Executive summary
   - Well-structured sections with H3 headings
   - Recent developments, trends, expert insights, and key statistics
   - A concise conclusion
3. Cite claims inline using [Source: URL].
4. End with a References section listing all unique source URLs.
5. Keep the writing engaging but factual and precise.
"""

    response = genai_client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
        config=GenerateContentConfig(
            temperature=temperature,
            tools=[Tool(google_search=GoogleSearch())],
        ),
    )
    return response.text

# Main content area
if generate_button:
    with st.spinner('Generating content... This may take a moment.'):
        try:
            result = generate_content(topic)
            st.markdown("### Generated Content")
            st.markdown(result)
            
            # Add download button
            st.download_button(
                label="Download Content",
                data=result,
                file_name=f"{topic.lower().replace(' ', '_')}_article.md",
                mime="text/markdown"
            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit and powered by Google Vertex AI")