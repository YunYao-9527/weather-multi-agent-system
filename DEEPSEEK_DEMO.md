# DeepSeek Public Demo

The live service uses the official endpoint `https://api.deepseek.com` and model `deepseek-flash`. Secrets are managed in the existing Render service environment through merge updates, not committed to source or Blueprint files.

Relevant environment keys: DEEPSEEK_API_KEY, AGENT_LLM_PROVIDER, AGENT_LLM_MODEL, AGENT_LLM_MODE, AGENT_LLM_ENABLED. Compatible SDK clients use the OpenAI adapter with the DeepSeek base URL; this does not send the DeepSeek credential to OpenAI.

## Public Usage Limits

- `PUBLIC_DEMO=1` enables the application-level budget.
- `LLM_DAILY_CALL_LIMIT=20` permits at most 20 provider calls per UTC day per process. Failed upstream attempts consume a reservation; automatic SDK retries are disabled for the compatible SDK clients.
- `LLM_INPUT_CHAR_LIMIT=24000` bounds serialized model input.
- Output is capped at 900 tokens per provider call in the deployed configuration.
- Thinking mode is disabled for the official DeepSeek endpoint to keep the demonstration bounded and responsive.
- AgentOps additionally allows at most three items in a public model-evaluation batch. Rule evaluations do not consume model quota.

Counters are held in memory and reset on process restart, redeployment or free-instance spin-down. These controls are abuse reduction, not a permanent spending ledger or a provider-enforced monetary cap. Monitor actual usage in the DeepSeek console and use a dedicated demonstration credential.

The application does not offer a public endpoint for changing the upstream key or base URL. Never paste API credentials into the website, prompts, screenshots or repository. Do not submit real customer or payment information to the synthetic business demonstration.

Real-model acceptance checks were completed on 2026-10-22. Unit tests use mocks and synthetic credentials, not the production key.

