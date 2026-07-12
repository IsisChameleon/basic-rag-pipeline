# AWS deployment options (RAG pipeline in a LangGraph agent workflow)

Three production deployment options for this repo on AWS, prepared for a
bank-context architecture discussion. Diagrams live in Eraser (one file, three
diagrams):

- **Eraser file**: https://app.eraser.io/workspace/vQJJREtfqgsR8H2NwYds
- [Option 1 — Agentic RAG on Amazon EKS](https://app.eraser.io/workspace/vQJJREtfqgsR8H2NwYds?diagram=9VoQxILY_h4_A8jf3mlS&layout=canvas)
- [Option 2 — Bedrock AgentCore Runtime](https://app.eraser.io/workspace/vQJJREtfqgsR8H2NwYds?diagram=6-6nz0RPZXuh0KoSxU73&layout=canvas)
- [Option 3 — ECS Fargate service + tasks](https://app.eraser.io/workspace/vQJJREtfqgsR8H2NwYds?diagram=ayZcBwuir7xBCpHhVHd7&layout=canvas)

## How the repo maps to a bank-grade AWS deployment

All three options share the same production substitutions — the choice between
them is purely a compute/ops decision, the RAG data plane is identical:

| In this repo (dev) | On AWS (prod) | Why |
|---|---|---|
| Gemini (`GOOGLE_API_KEY`) | **Amazon Bedrock — Claude**, via PrivateLink VPC endpoints | Prompts/completions never leave AWS; auth is the IAM role (one less secret) |
| Local sentence-transformers (embed + rerank) | **Bedrock Titan Text Embeddings v2 + Bedrock Rerank** | Drops the ~2 GB torch image and CPU inference cost |
| Embedded ChromaDB | **Aurora PostgreSQL + pgvector** (or OpenSearch Serverless) | Embedded Chroma is a file on one container's disk — doesn't survive multi-replica scaling |
| In-process ingest jobs (`POST /ingest` → 202 + job id) | Queue + ephemeral compute (K8s Jobs / Fargate tasks / Knowledge Bases sync) | Long crawls shouldn't share the API's lifecycle |
| LangGraph agent wraps retrieve → rerank → generate as a tool | Same, checkpointing state to Postgres | Durable state across restarts / turns |
| Langfuse via compose profile | Self-hosted Langfuse in-VPC, or CloudWatch GenAI Observability | LLM traces contain customer data — keep them inside the VPC |

Compliance layer expected in all options: Bedrock Guardrails, KMS
customer-managed keys, Secrets Manager, WAF, private subnets, least-privilege
IAM.

## Option 1 — Amazon EKS

The "platform team" answer. Current best practice:

- **EKS Auto Mode** — Karpenter runs as a managed component (node
  autoscaling/consolidation without operating the controller yourself).
- **EKS Pod Identity** (successor to IRSA) — pods assume IAM roles to reach
  Bedrock/Aurora/S3 with no static credentials.
- Agent API (FastAPI + LangGraph) as a Deployment with HPA behind an ALB
  (AWS Load Balancer Controller); ingestion as Kubernetes Jobs / Argo
  Workflows; Langfuse self-hosted in-cluster.
- Aurora pgvector doubles as the LangGraph checkpointer (the pattern in AWS's
  stateful-LangGraph-on-EKS reference).
- GitOps deploys: GitHub Actions → ECR → Argo CD.

**Pick when**: the bank already runs EKS / wants portability and full control.
**Cost**: highest operational overhead of the three.

## Option 2 — Amazon Bedrock AgentCore Runtime

The AWS-native (2025+) answer for hosting agents. The LangGraph agent ships as
a container to ECR; AgentCore runs it. Retrieval is exposed to the agent as an
MCP tool via **AgentCore Gateway**; ingestion becomes fully managed with
**Bedrock Knowledge Bases** (S3 → chunk → embed → OpenSearch Serverless).
Built-in **Memory** (session + long-term), **Identity** (OAuth against
Cognito/corporate IdP), and **Observability** (OTEL → CloudWatch).

### Why AgentCore Runtime beats plain Lambda for agents

1. **Session duration** — up to 8 h vs Lambda's hard 15-min cap; multi-step
   LangGraph workflows with several LLM calls routinely blow past 15 min.
2. **Isolation** — a dedicated microVM per session (own CPU/memory/filesystem,
   memory sanitized at teardown), vs Lambda's reused warm sandboxes.
3. **Statefulness** — session affinity keeps a conversation on the same
   microVM; Lambda is stateless per invocation.
4. **Pricing** — consumption-based: CPU is not billed while the agent waits on
   LLM I/O (most of an agent's wall-clock time).
5. **Batteries included** — Memory / Identity / Gateway / Observability as
   services instead of DIY DynamoDB + API Gateway + custom middleware.

(If asked about **Lambda MicroVMs**, launched June 2026: also 8-hour isolated
sessions, but they're the general-purpose primitive for running
untrusted/LLM-generated code; AgentCore Runtime is the purpose-built agent
host with the identity/memory/tooling layer on top.)

### Session lifecycle (when is a session recycled?)

No signal is required to end a session — the runtime decides:

- **Idle timeout** (default 15 min, configurable 60 s – 8 h via
  `idleRuntimeSessionTimeout`): after a request finishes the microVM sits
  `Idle`, warm for follow-up turns; no invocation within the timeout →
  terminated and memory sanitized. The normal way sessions end.
- **Max lifetime** (default and ceiling 8 h, `maxLifetime`): even a busy
  session is terminated at the cap. "Up to 8 hours" means a session *may*
  live that long if it keeps working — most die at the idle timeout.
- **Explicit stop** — `StopRuntimeSession` API kills the VM immediately
  (including in-flight streaming). Worth calling when the session is known to
  be over: CPU is billed only while processing, but **memory is billed for as
  long as the session exists**.
- **Health checks** — the container exposes `/ping` returning `"Healthy"`
  (idle, safe to reap) or `"HealthyBusy"` (background task running — keep
  alive past the idle timeout). This is how a long async job (e.g. sitemap
  ingestion as an agent tool) survives with no incoming traffic. Unhealthy →
  terminated regardless.

One-liner: *sessions are reaped on idle timeout, capped at 8 h, can be stopped
explicitly, and background work extends life via `HealthyBusy` — state that
must outlive the session belongs in the LangGraph checkpointer or AgentCore
Memory, not the VM.*

**Cold start**: yes — the first request of a new session provisions a microVM
(order of seconds); later turns in the session hit the warm VM.

## Option 3 — Amazon ECS on Fargate (service + tasks)

The pragmatic middle: much less to operate than EKS, no Kubernetes expertise
needed, natural fit for the existing design.

- API as a long-lived **ECS Service** on Fargate behind ALB, target-tracking
  auto scaling.
- `POST /ingest` → SQS → Step Functions → **standalone Fargate tasks**
  (`RunTask`) that crawl/extract/chunk/embed, then scale to zero — ingestion
  is only paid for while it runs. EventBridge schedules re-crawls.
- Aurora Serverless v2 (pgvector + checkpoints), Bedrock via VPC endpoints,
  Langfuse as a sibling ECS service.
- Blue/green deploys via CodeDeploy.

**Pick when**: a single service like this one is going to production and no
container platform exists yet — often the right first step.

### Closing line for the interview

"I'd start on ECS Fargate or AgentCore depending on whether the bank treats
the agent as an app or as part of an agent platform, and I'd only reach for
EKS if a container platform already exists — the RAG data plane (Bedrock +
pgvector/OpenSearch + S3, all behind PrivateLink) is identical in all three,
so the choice is purely a compute/ops decision."

## FAQ (concepts, mapped to GCP equivalents where useful)

### Amazon Bedrock

A single API in front of a catalogue of foundation models (Anthropic Claude,
Amazon Nova/Titan, Meta Llama, Mistral, Cohere…) — the analogue of Vertex AI
Model Garden. No instances; `InvokeModel`/`Converse` with IAM-signed requests,
pay per token. All three model calls this repo makes go there: generation
(Claude ← Gemini), embeddings (Titan v2 ← local sentence-transformer),
reranking (Bedrock Rerank ← local cross-encoder). Bedrock also layers managed
services on top: Knowledge Bases (managed RAG), Guardrails, AgentCore.

### Aurora and pgvector

Aurora is AWS's managed Postgres-compatible engine (analogue: Cloud SQL /
AlloyDB — *not* Supabase, which is a BaaS bundle of Postgres + auth + APIs).
Cloud-native storage: compute/storage separated, 6-way replication across 3
AZs, Serverless v2 autoscaling. **pgvector is not a database** — it's a
Postgres extension adding a `vector` column type, distance operators
(`<=>` cosine, `<->` L2) and HNSW/IVFFlat indexes. Embeddings live in an
ordinary SQL table next to relational data; retrieval is a SQL query; the same
Postgres holds the LangGraph checkpoint tables.

### KMS vs Secrets Manager

- **KMS** = encryption keys for data at rest (S3, Aurora, ECR, CloudWatch logs
  all reference a KMS key). Banks use customer-managed keys (CMKs) so key
  usage is auditable in CloudTrail and rotation/revocation is theirs. Not
  something app code touches.
- **Secrets Manager** = runtime credentials the app reads: Aurora password (or
  none, with IAM database auth), Langfuse keys, HF token if local models are
  kept. Moving Gemini → Bedrock *eliminates* a secret (no `GOOGLE_API_KEY`
  equivalent — auth is the task/pod IAM role).

### WAF

Web application firewall attached to ALB/CloudFront/API Gateway: managed rules
(SQLi, XSS, bad bots), IP allow/deny, rate limiting. GCP analogue: Cloud
Armor. For an LLM API the rate limiting matters most — first defence against
someone flooding `/answer` and running up the token bill.

### Langfuse vs LangSmith vs raw OTEL

LangSmith is LangChain's commercial SaaS (self-hosting only at enterprise
tier); Langfuse is the open-source, self-hostable equivalent (why it's in this
repo's compose file, and why it suits a bank — traces with customer data never
leave the VPC). Raw OTEL → CloudWatch/Cloud Trace gives generic APM (spans,
latency, whatever attributes you attach). Langfuse consumes the same trace
data (v3 accepts OTLP — it's effectively an OTEL backend with LLM semantics)
but adds the LLM-specific product: prompt/completion rendering per generation,
session/conversation views, cost roll-ups, **prompt management** (versioned
prompts deployed independently of code), **datasets + evals**, human
annotation queues. Not "an SDK with a light UI": the platform needs Postgres,
ClickHouse, Redis and S3/MinIO (see `docker-compose.yaml`). Rule of thumb:
"is it slow / what does it cost" → OTEL is enough; prompt iteration +
eval/annotation workflows → Langfuse.

### Lambda warm-container risk / why agents forced per-session isolation

Lambda reuses a warm execution environment across sequential invocations of
the same function (never across accounts/functions). Anything left behind
(`/tmp`, module globals, caches) is visible to the next invocation. Manageable
with hygiene for short stateless handlers; agents break the assumptions — they
hold a user's whole conversation and retrieved documents in memory for a long
time, and increasingly execute LLM-generated code, where "trust the code to be
hygienic" stops being a security model. Hence per-session microVMs, destroyed
and sanitized at session end.

### Statefulness and session affinity

Two layers:

1. **Session affinity** (sticky routing): every call with the same
   `runtimeSessionId` lands on the same microVM, so process memory (the
   constructed LangGraph graph, caches, temp files) survives across turns
   within a conversation. Lambda gives no such guarantee — a 10-turn
   conversation might touch 10 sandboxes, each turn rebuilding state.
2. **Durable state** (LangGraph checkpointer): graph state persisted after
   each step to Postgres/DynamoDB/AgentCore Memory — survives VM teardown,
   enables human-in-the-loop interrupts and resume/replay.

Best practice is both: affinity makes turns fast, the checkpointer makes them
durable.

### AgentCore Observability — what's automatic, what you do

Automatic (zero code): runtime metrics — invocations, sessions, latency,
errors, throttles — in CloudWatch. For traces (LLM calls, tool calls, RAG
spans):

1. Add `aws-opentelemetry-distro` (ADOT) to the container and run under
   `opentelemetry-instrument` with `AGENT_OBSERVABILITY_ENABLED=true` (the
   starter toolkit wires this).
2. Add the framework auto-instrumentor (for LangGraph: the LangChain
   OpenInference/OTEL instrumentation) — every node/LLM call/tool call becomes
   a span with token usage.
3. One-time per account: enable **CloudWatch Transaction Search** (~10 min
   before spans appear).

Everything shows in the CloudWatch console under **GenAI Observability**
(Agents / Sessions / Traces views). RAG-specific metrics (retrieval latency,
chunk counts, rerank scores) are custom spans/attributes around the
retrieve/rerank steps, inline in the same traces. It's standard OTEL, so you
can dual-export — CloudWatch for ops, Langfuse for prompt/eval work — from the
same instrumentation.

## References

- [Host agents with Bedrock AgentCore Runtime — AWS docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agents-tools-runtime.html)
- [Serverless LangGraph multi-agent systems with AgentCore — AWS ML Blog](https://aws.amazon.com/blogs/machine-learning/build-highly-scalable-serverless-langgraph-multi-agent-systems-in-aws-with-amazon-bedrock-agentcore/)
- [Isolated sessions for agents — AgentCore docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-sessions.html)
- [Configure AgentCore lifecycle settings — AWS docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-lifecycle-settings.html)
- [Handle asynchronous and long-running agents — AWS docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-long-run.html)
- [Stop a running session — AWS docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-stop-session.html)
- [Get started with AgentCore Observability — AWS docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-get-started.html)
- [Observability Quickstart — AgentCore starter toolkit](https://aws.github.io/bedrock-agentcore-starter-toolkit/user-guide/observability/quickstart.html)
- [Stateful LangGraph IT service-desk agent on EKS — AWS Open Source Blog](https://aws.amazon.com/blogs/opensource/building-a-stateful-it-service-desk-agent-with-langgraph-on-amazon-eks/)
- [EKS Auto Mode best practices — AWS docs](https://docs.aws.amazon.com/eks/latest/best-practices/automode.html)
- [What you need to know about Lambda MicroVMs — theburningmonk](https://theburningmonk.com/2026/06/what-you-need-to-know-about-lambda-microvms/)
- [Lambda MicroVMs vs AgentCore Runtime — AWS Builders](https://dev.to/aws-builders/lambda-microvms-vs-agentcore-runtime-when-to-use-each-for-production-agents-5gm7)
- [Architect for AWS Fargate on ECS — AWS docs](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/AWS_Fargate.html)
