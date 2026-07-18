# Gmail OAuth setup (digest agent)

How the digest agent gets read-only access to Gmail. The agent uses the
**Gmail REST API** directly via `google-api-python-client` (not the Gmail MCP
API), so credential acquisition can be swapped for AgentCore Identity on
deploy without touching `agent/email_source.py`.

Scope: `https://www.googleapis.com/auth/gmail.readonly` (set in code). Read-only
at the credential level means writes are impossible, not merely un-bound.

## Two OAuth clients, one project, two contexts

Local dev and the deployed AgentCore backend need **different** Google OAuth
client types, because the OAuth redirect differs:

| | Local build (now) | AgentCore backend (day 2) |
|---|---|---|
| Runs the flow | a script on the laptop | AWS, server-side |
| Redirect target | `http://localhost:<random-port>/` (loopback) | AgentCore's fixed callback URL |
| Google client type | **Desktop app** | **Web application** |
| Token lives in | `token.json` on disk | AgentCore Token Vault |

"Desktop app" here is Google's term for a *public/installed client using a
loopback redirect* -- not "a GUI app". A local Python script is exactly that.
Day 2 does not remove the Google client; AgentCore Identity *wraps* it: you feed
the Web client's id/secret to `CreateOauth2CredentialProvider`, then register
the unique `callbackUrl` it returns as the Web client's redirect URI.

## Configured so far

| Item | Value |
|------|-------|
| Google Cloud project | `agentic-workflows` (reused an existing project) |
| API enabled | Gmail API (`gmail.googleapis.com`) |
| Consent screen — app name | `test-agentic-workflows` |
| Consent screen — user support email | isisdesade@gmail.com |
| Consent screen — audience | External |
| Local OAuth client (Desktop app) | `test-agentic-workflows-oauth-client-localhost` |
| Day-2 OAuth client (Web application) | `test-agentic-workflows-oauth-client` (unused until AgentCore) |

## Still to do (local)

- [ ] Add own Gmail address as a **Test user** (an unverified External app only
      admits listed testers; skipping this causes `access_denied` at consent).
- [ ] Download the **Desktop** client's JSON to repo root as `credentials.json`.
- [ ] First run: browser opens once, consent granted, `token.json` written and
      refreshed silently thereafter.

Both `credentials.json` (client secret) and `token.json` (user tokens) are
gitignored — never commit either.
