"""
VERTEX — Central configuration
Supports Groq (recommended), OpenRouter, and OpenAI.
Priority: Groq → OpenRouter → OpenAI
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


class Settings:
    # ── LLM providers ─────────────────────────────────────────────────────────
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    openrouter_api_key: str = os.getenv("OPENROUTER_API_KEY", "")

    # Embedding model loader
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER","fastembed")
    embedding_model: str = os.getenv("EMBEDDING_MODEL","BAAI/bge-small-en-v1.5")

    # Model — overridden per provider below in get_llm()
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))

    # ── External data APIs ────────────────────────────────────────────────────
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    news_api_key: str = os.getenv("NEWS_API_KEY", "")
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "VERTEX vertex@demo.com")

    # ── App settings ──────────────────────────────────────────────────────────
    fastapi_port: int = int(os.getenv("FASTAPI_PORT", "8000"))
    debate_max_rounds: int = int(os.getenv("DEBATE_MAX_ROUNDS", "3"))
    debate_confidence_threshold: float = float(
        os.getenv("DEBATE_CONFIDENCE_THRESHOLD", "0.75")
    )
    chroma_persist_dir: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    graph_persist_path: str = os.getenv(
        "GRAPH_PERSIST_PATH", "./data/graph_store/knowledge_graph.pkl"
    )
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    @property
    def active_provider(self) -> str:
        """Which LLM provider is active based on available keys."""
        if self.groq_api_key and self.groq_api_key != "your_groq_key_here":
            return "groq"
        if self.openrouter_api_key and self.openrouter_api_key != "your_openrouter_key_here":
            return "openrouter"
        if self.openai_api_key and self.openai_api_key != "your_openai_key_here":
            return "openai"
        return "none"

    def validate(self) -> list[str]:
        """Return list of missing required keys."""
        missing = []
        if self.active_provider == "none":
            missing.append("GROQ_API_KEY (or OPENROUTER_API_KEY or OPENAI_API_KEY)")
        if not self.alpha_vantage_key:
            missing.append("ALPHA_VANTAGE_API_KEY")
        if not self.news_api_key:
            missing.append("NEWS_API_KEY")
        return missing


settings = Settings()


def get_llm(temperature: float = None):
    """
    Factory function — returns the correct LangChain LLM based on
    which API key is configured. Import this everywhere instead of
    instantiating ChatOpenAI directly.

    Usage:
        from vertex.config import get_llm
        llm = get_llm()
        response = llm.invoke([...])
    """
    temp = temperature if temperature is not None else settings.llm_temperature
    provider = settings.active_provider

    # ── Groq ──────────────────────────────────────────────────────────────────
    if provider == "groq":
        from langchain_groq import ChatGroq
        model = settings.llm_model
        # Groq model name mapping — use best available
        groq_models = {
            # If user left default OpenAI model name, map to best Groq equivalent
            "gpt-4o-mini": "llama-3.3-70b-versatile",
            "gpt-4o": "llama-3.3-70b-versatile",
            "gpt-4": "llama-3.3-70b-versatile",
            "gpt-3.5-turbo": "llama-3.1-8b-instant",
        }
        model = groq_models.get(model, model)
        return ChatGroq(
            model=model,
            temperature=temp,
            api_key=settings.groq_api_key,
            max_retries=3,
        )

    # ── OpenRouter ────────────────────────────────────────────────────────────
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        model = settings.llm_model
        # Map to good free OpenRouter models if default names given
        openrouter_models = {
            "gpt-4o-mini": "meta-llama/llama-3.3-70b-instruct:free",
            "gpt-4o": "meta-llama/llama-3.3-70b-instruct:free",
            "gpt-4": "meta-llama/llama-3.3-70b-instruct:free",
            "llama-3.3-70b-versatile": "meta-llama/llama-3.3-70b-instruct:free",
        }
        model = openrouter_models.get(model, model)
        return ChatOpenAI(
            model=model,
            temperature=temp,
            api_key=settings.openrouter_api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "VERTEX Financial Intelligence",
            },
            max_retries=3,
        )

    # ── OpenAI (fallback) ─────────────────────────────────────────────────────
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            temperature=temp,
            api_key=settings.openai_api_key,
            max_retries=3,
        )

    raise ValueError(
        "No LLM provider configured. Set GROQ_API_KEY in your .env file.\n"
        "Get a free key at: https://console.groq.com"
    )
