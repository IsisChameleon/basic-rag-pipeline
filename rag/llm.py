from __future__ import annotations

from google.genai import Client, types
from langfuse import get_client, observe

from rag.observability import ObservationType


class LLMClient:
    """Gemini text generation. The google-genai client is created lazily on
    first generate() -- and the missing-API-key error raised there -- so the
    app boots and ingest/search keep working without a key (only /answer
    needs one). No warm_up: there is no model to load, just an HTTP client."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model
        self._client: Client | None = None

    def _get_client(self) -> Client:
        if self._client is None:
            if not self._api_key:
                raise RuntimeError(
                    "GOOGLE_API_KEY is not set -- /answer needs it to call Gemini. "
                    "Add it to your local .env (see .env.example)."
                )
            self._client = Client(api_key=self._api_key)
        return self._client

    @observe(as_type=ObservationType.GENERATION, name="gemini-generate")
    def generate(self, prompt: str, *, system: str) -> str:
        response = self._get_client().models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )
        # Record model + token usage on the generation span when Langfuse is
        # active. google-genai exposes counts on response.usage_metadata; skip
        # if absent -- update_current_generation no-ops when tracing is
        # disabled.
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            get_client().update_current_generation(
                model=self._model,
                usage_details={
                    "input": getattr(usage, "prompt_token_count", None),
                    "output": getattr(usage, "candidates_token_count", None),
                },
            )
        return response.text or ""
