"""Environment setup: load API keys from the project .env file.

This module ensures all exercises use the .env file at the project root
for API credentials, rather than relying on system environment variables.
"""

import os
from pathlib import Path

from dotenv import dotenv_values

# Project root is two levels up from aisb_utils/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv() -> None:
    """Load API keys from the project .env file, overriding any system env vars.

    This ensures we use exclusively the credentials from .env,
    not any pre-existing environment variables.
    """
    if not DOTENV_PATH.exists():
        raise FileNotFoundError(
            f"No .env file found at {DOTENV_PATH}. "
            "Copy .env.example to .env and add your OpenRouter API key."
        )

    env_values = dotenv_values(DOTENV_PATH)

    # Set each value from .env, overriding any existing env vars
    for key, value in env_values.items():
        if value is not None:
            os.environ[key] = value

    # Clear any custom base URL that might point to a non-standard endpoint,
    # so we use the default OpenRouter API (https://openrouter.ai/api/v1)
    if "OPENROUTER_BASE_URL" not in env_values:
        os.environ.pop("OPENROUTER_BASE_URL", None)

    # Clear system-level inspect_ai configuration that may conflict
    # (e.g., telemetry providers that aren't installed locally)
    for key in list(os.environ):
        if key.startswith("INSPECT_") and key not in env_values:
            os.environ.pop(key)
