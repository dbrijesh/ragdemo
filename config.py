import os
import urllib.request


def _on_ec2() -> bool:
    """Return True when running on an EC2 instance.

    IMDSv2 instances return HTTP 401 for unauthenticated requests — that is still
    a response from the metadata service, so we ARE on EC2. Only a connection error
    or timeout means we are running outside EC2.
    """
    try:
        urllib.request.urlopen(
            "http://169.254.169.254/latest/meta-data/instance-id",
            timeout=0.5,
        )
        return True
    except urllib.error.HTTPError:
        # Got an HTTP response (e.g. 401 IMDSv2) — metadata service is reachable → EC2
        return True
    except Exception:
        return False


AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# On EC2 use the instance IAM role (AWS_PROFILE = None).
# Locally fall back to the "brijesh" profile (or whatever AWS_PROFILE env var says).
AWS_PROFILE: str | None = (
    None if _on_ec2()
    else (os.getenv("AWS_PROFILE", "brijesh") or None)
)

BEDROCK_LLM_MODEL   = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
BEDROCK_EMBED_MODEL = "amazon.titan-embed-text-v2:0"

CHROMA_PATH      = "./chroma_db"
COLLECTION_NAME  = "knowledge_base"
DOCUMENTS_PATH   = "./documents"
TEST_DATA_PATH   = "./test_data/test_questions.jsonl"

CHUNK_SIZE       = 250   # words — smaller = more chunks = richer t-SNE clusters
CHUNK_OVERLAP    = 40
TOP_K            = 3
EMBED_DIMENSIONS = 1024
