"""Download the model and tokenizer into the Hugging Face cache.

Run once before starting the server:  python scripts/download_model.py
After that, scripts/run_server.sh runs fully offline (HF_HUB_OFFLINE=1), so the
pinned model is always loaded from the local cache and never re-downloaded.
"""

import os
import sys
from pathlib import Path

# Running `python scripts/download_model.py` puts only the scripts/ folder on
# Python's import path. Add the project root so `import app` works, and switch
# to it so the settings find .env (and your pinned revision) from any folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

from app.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    print(f"Downloading {settings.model_name}@{settings.model_revision}")
    # from_pretrained fetches only the files this model actually needs.
    AutoTokenizer.from_pretrained(settings.model_name, revision=settings.model_revision)
    AutoModelForSequenceClassification.from_pretrained(
        settings.model_name, revision=settings.model_revision
    )
    print("Done")


if __name__ == "__main__":
    main()
