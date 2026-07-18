"""The digest agent's composition root. Composes the shared RagContainer and
adds the one thing only this process needs: a Gmail connection.

Gmail belongs here, not in RagContainer, so the RAG API still boots with no
Gmail OAuth. build_gmail_service holds the entire local-credential story --
credentials.json, token.json, the consent flow -- in one function; on AgentCore
that function is what gets replaced (the token comes from the Identity vault),
and EmailSource, the graph, and everything below are untouched."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agent.email_source import EmailSource  # no google imports; safe at top level
from core.settings import Settings
from rag.container import RagContainer, build_rag_container

# Read-only at the credential level: with this scope the token physically
# cannot send, delete, or modify mail -- writes are impossible, not just
# un-bound. This is the capability-separation guarantee, enforced by Google.
GMAIL_SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


@dataclass
class AgentContainer:
    settings: Settings
    rag: RagContainer
    email_source: EmailSource

    def warm_up(self) -> None:
        self.rag.warm_up()


def build_gmail_service(settings: Settings) -> Any:
    """Return an authorised Gmail API resource for local development.

    Loads token.json if present and still valid, refreshes it silently when it
    has a refresh token, and otherwise runs the one-time browser consent
    against credentials.json, persisting the result. Imports the google
    libraries lazily so the rest of agent/ (and its tests) need neither the
    packages nor a token.
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds: Credentials | None = None
    if settings.gmail_token_path.exists():
        creds = Credentials.from_authorized_user_file(
            str(settings.gmail_token_path), GMAIL_SCOPES
        )
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                str(settings.gmail_credentials_path), GMAIL_SCOPES
            )
            creds = flow.run_local_server(port=0)  # opens the browser once
        settings.gmail_token_path.write_text(creds.to_json())
    return build("gmail", "v1", credentials=creds)


def build_agent_container(settings: Settings | None = None) -> AgentContainer:
    settings = settings or Settings()
    return AgentContainer(
        settings=settings,
        rag=build_rag_container(settings),
        # Developing offline? Swap in GoldenEmailSource() here: it replays
        # data/golden_emails.json (written by scripts/smoke_gmail.py) with the
        # same load() shape -- no OAuth, no API calls, stable input every run.
        email_source=EmailSource(build_gmail_service(settings)),
    )
