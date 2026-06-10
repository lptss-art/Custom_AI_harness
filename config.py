import os
from openai import AsyncOpenAI
from google import genai
from dotenv import load_dotenv

load_dotenv()

# DeepSeek client setup (requires OPENAI_API_KEY and OPENAI_BASE_URL in .env, or defaults for DeepSeek)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")

deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# Gemini client setup
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)

# Engine Constants
K_FACTOR = 32
BASE_ELO = 1200
MAX_CONCURRENT_MATCHES = 5
DEFAULT_PROPOSE_COUNT = 5
