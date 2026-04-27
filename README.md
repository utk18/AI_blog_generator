
# AI News generator

This project uses Streamlit with Google Vertex AI (Gemini) to build an AI news generator with grounded web research.

## Installation and setup

**Set up Google Cloud credentials**:
   - Create a service account with Vertex AI access (for example, Vertex AI User).
   - Download the service account JSON key.
   - Set `GOOGLE_APPLICATION_CREDENTIALS` to the absolute path of that JSON key.
   - Set your project and region in `.env`.


**Install Dependencies**:
   Ensure you have Python 3.11 or later installed.
   ```bash
   pip install streamlit google-genai python-dotenv
   ```

## Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Ensure these files exist at repo root:
   - `app.py`
   - `requirements.txt`
   - `runtime.txt`
3. In Streamlit Community Cloud, create a new app and select this repo and `app.py`.
4. In app **Settings > Secrets**, add:

```toml
GCP_PROJECT_ID="your-gcp-project-id"
GCP_LOCATION="global"
VERTEX_MODEL_NAME="gemini-2.0-flash"

[gcp_service_account]
type="service_account"
project_id="your-gcp-project-id"
private_key_id="..."
private_key="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
client_email="...@...iam.gserviceaccount.com"
client_id="..."
auth_uri="https://accounts.google.com/o/oauth2/auth"
token_uri="https://oauth2.googleapis.com/token"
auth_provider_x509_cert_url="https://www.googleapis.com/oauth2/v1/certs"
client_x509_cert_url="https://www.googleapis.com/robot/v1/metadata/x509/..."
universe_domain="googleapis.com"
```

5. Save secrets and redeploy.

Notes:
- Do not commit your service account JSON file to git.
- Grant the service account Vertex AI permissions in your GCP project.

---

## Contribution

Contributions are welcome! Please fork the repository and submit a pull request with your improvements.
