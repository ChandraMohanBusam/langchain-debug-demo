# langchain-debug-demo

A production-grade reference project demonstrating how to debug and log
LangChain and LangGraph agents **without relying on paid observability tools**.

Built by Chandra Mohan Busam | Principal Engineer and AI Engineer

**Architecture Notes:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
Design decisions and reasoning behind the three-level debugging approach, config-driven handler stack, and why LangGraph needs a different philosophy than LangChain.

---

## What This Project Covers

| Technique | LangChain Agent | LangGraph Agent |
|---|---|---|
| verbose=True / set_debug(True) | Yes | N/A |
| FileCallbackHandler | Yes | N/A |
| Custom BaseCallbackHandler | Yes (4 handlers) | N/A |
| Token usage tracking | Yes | N/A |
| Loop detection | Yes (LoopDetectionHandler) | Yes (revision_count guard) |
| stream_mode="debug" | N/A | Yes |
| Checkpointer (MemorySaver) | N/A | Yes |
| get_state() inspection | N/A | Yes |
| Time Travel / State History | N/A | Yes |
| Slack alerts | Yes | N/A |
| Microsoft Teams alerts | N/A | Yes |

---

## Project Structure

```
langchain-debug-demo/
│
├── docs/
│   └── ARCHITECTURE.md             # Design decisions and why things work the way they do
│
├── langchain_agent/                # AI Deployment Agent (LangChain ReAct)
│   ├── agent.py                    # Main entry point (3 debugging levels)
│   ├── callbacks.py                # DeploymentAuditHandler, TokenUsageHandler,
│   │                               # LoopDetectionHandler, SlackAlertHandler, TeamsAlertHandler
│   ├── config.json                 # Master toggle: handlers, log paths, alert provider
│   ├── config_loader.py            # Reads config.json and builds active handler list
│   ├── tools.py                    # DownloadBuild, TransferToServer, DeployOnServer, RestartServices
│   └── logs/                       # Created at runtime
│       ├── deployment_audit.log
│       ├── token_usage.log
│       ├── langchain_file_callback.log
│       └── loop_detection.log
│
├── langgraph_agent/                # Document Review Agent (LangGraph State Machine)
│   ├── runner.py                   # Main entry point (4 debugging modes)
│   ├── graph.py                    # Graph definition, nodes, state, routing
│   └── teams_alert.py              # Microsoft Teams Incoming Webhook integration
│
├── LICENSE
├── requirements.txt
├── .env.example                    # Copy to .env and fill in your keys
└── README.md
```

---

## 1. Installation

```bash
# Clone the repo
git clone https://github.com/ChandraMohanBusam/langchain-debug-demo.git
cd langchain-debug-demo

# Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Then edit .env with your actual keys (see Sections 3 and 4 below)
```

---

## 2. Running the Agents

### LangChain Agent (AI Deployment Agent)

The agent simulates deploying a build to a server. The `TransferToServer` tool
fails checksum verification on the first two attempts, causing the agent to loop.
Each debug level reveals this differently.

```bash
cd langchain_agent

# Level 1: verbose + global debug mode
# Prints everything to terminal. Fast to enable, hard to read.
python agent.py --mode level1

# Level 2: FileCallbackHandler
# Writes LangChain's built-in formatted log to a file.
python agent.py --mode level2
# Then check: logs/langchain_file_callback.log

# Level 3: Full custom callback stack (recommended)
# Structured logs per concern. Loop detection. Slack alerts.
python agent.py --mode level3
# Then check: logs/deployment_audit.log
#             logs/token_usage.log
#             logs/loop_detection.log
```

### LangGraph Agent (Document Review Agent)

The agent reviews a vague contract document. The reviewer keeps requesting
revisions because the content is ambiguous, demonstrating a state machine loop.

```bash
cd langgraph_agent

# Mode 1: Standard run, no debugging
python runner.py --mode standard

# Mode 2: stream_mode="debug" - structured real-time events
python runner.py --mode stream_debug

# Mode 3: Checkpointer + get_state() after run
python runner.py --mode inspect_state

# Mode 4: Full state history + time travel demo (recommended)
python runner.py --mode time_travel
```

---

## 3. config.json Reference (langchain_agent)

All logging and alerting behaviour for the LangChain agent is controlled
by a single file: `langchain_agent/config.json`. No code changes are needed
to toggle handlers or switch alert channels.

### Full config.json with explanations

```json
{
  "logging": {
    "file_callback_handler": {
      "enabled": true,
      "output_file": "logs/langchain_file_callback.log"
    },
    "audit_log": {
      "enabled": true,
      "output_file": "logs/deployment_audit.log"
    },
    "token_usage_log": {
      "enabled": true,
      "output_file": "logs/token_usage.log"
    },
    "loop_detection_log": {
      "enabled": true,
      "output_file": "logs/loop_detection.log",
      "loop_threshold": 2
    },
    "log_level": "INFO"
  },
  "alerts": {
    "provider": "slack",
    "notify_on": {
      "llm_error": true,
      "chain_error": true,
      "tool_error": true,
      "loop_detected": true
    },
    "slack": {
      "webhook_url_env": "SLACK_WEBHOOK_URL"
    },
    "teams": {
      "webhook_url_env": "TEAMS_WEBHOOK_URL"
    }
  }
}
```

### Config flags explained

| Flag | Type | What It Controls |
|---|---|---|
| `logging.file_callback_handler.enabled` | bool | LangChain's built-in FileCallbackHandler. Writes formatted chain output to file. |
| `logging.audit_log.enabled` | bool | DeploymentAuditHandler. Structured per-step log: prompts, responses, tool calls. |
| `logging.token_usage_log.enabled` | bool | TokenUsageHandler. Tracks prompt/completion/total tokens per LLM call. |
| `logging.loop_detection_log.enabled` | bool | LoopDetectionHandler. Detects repeated identical tool calls. |
| `logging.loop_detection_log.loop_threshold` | int | How many identical calls before a loop warning fires. Default: 2. |
| `logging.log_level` | string | Python log level: DEBUG, INFO, WARNING, ERROR. |
| `alerts.provider` | string | Which channel to alert: `"slack"`, `"teams"`, or `"none"`. |
| `alerts.notify_on.llm_error` | bool | Alert when an individual LLM call fails. |
| `alerts.notify_on.chain_error` | bool | Alert when the entire agent chain fails (critical). |
| `alerts.notify_on.tool_error` | bool | Alert when a deployment tool raises an exception. |
| `alerts.notify_on.loop_detected` | bool | Alert when LoopDetectionHandler fires. |
| `alerts.slack.webhook_url_env` | string | Name of the env variable that holds the Slack webhook URL. |
| `alerts.teams.webhook_url_env` | string | Name of the env variable that holds the Teams webhook URL. |

### Common config scenarios

**Switching from Slack to Teams alerts:**
Change one line in config.json:
```json
"provider": "teams"
```

**Disable all alerts but keep logging:**
```json
"provider": "none"
```

**Only alert on critical chain failures, suppress everything else:**
```json
"notify_on": {
  "llm_error": false,
  "chain_error": true,
  "tool_error": false,
  "loop_detected": false
}
```

**Tighten loop detection for sensitive agents:**
```json
"loop_threshold": 1
```

**Minimal logging (audit only, no token tracking):**
```json
"file_callback_handler": { "enabled": false },
"token_usage_log": { "enabled": false },
"loop_detection_log": { "enabled": false }
```

---

## 4. Setting Up Slack Incoming Webhook (for langchain_agent)

Slack's Incoming Webhook lets you post messages to a Slack channel
using a simple HTTP POST. No OAuth, no bot tokens needed.

### Step 1: Create a Slack App

1. Go to [https://api.slack.com/apps](https://api.slack.com/apps)
2. Click **Create New App**
3. Choose **From scratch**
4. Enter an App Name (e.g., `LangChain Monitor`) and select your workspace
5. Click **Create App**

### Step 2: Enable Incoming Webhooks

1. In the left sidebar, click **Incoming Webhooks**
2. Toggle **Activate Incoming Webhooks** to **On**

### Step 3: Add a Webhook to Your Workspace

1. Scroll down and click **Add New Webhook to Workspace**
2. Choose the channel where you want alerts to appear (e.g., `#ai-agent-alerts`)
3. Click **Allow**
4. You will see a new Webhook URL that looks like:

```
https://hooks.slack.com/services/TXXXXXXXX/BXXXXXXXX/XXXXXXXXXXXXXXXXXXXXXXXX
```

5. Click **Copy** to copy this URL

### Step 4: Configure in Python

Add the webhook URL to your `.env` file:

```bash
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/TXXXXXXXX/BXXXXXXXX/XXXXXXXXXXXXXXXXXXXXXXXX
```

Access it in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()
webhook_url = os.getenv("SLACK_WEBHOOK_URL")
```

### Step 5: Test the Connection

```python
import requests, json, os
from dotenv import load_dotenv

load_dotenv()

response = requests.post(
    os.getenv("SLACK_WEBHOOK_URL"),
    data=json.dumps({"text": "Test alert from langchain-debug-demo"}),
    headers={"Content-Type": "application/json"}
)
print(response.status_code)  # Should print 200
```

### Security Notes

- Never hardcode the webhook URL in your source code
- Store it only in `.env` and add `.env` to `.gitignore`
- Rotate the webhook if it is ever accidentally committed to Git
- If your Slack workspace uses Enterprise Grid, you may need admin approval

---

## 5. Setting Up Microsoft Teams Incoming Webhook (for langgraph_agent)

Teams Incoming Webhooks allow you to post formatted cards to a Teams channel.

### Step 1: Open the Target Channel

1. Open Microsoft Teams
2. Navigate to the channel where you want alerts (e.g., `AI Monitoring`)
3. Click the three-dot menu (...) next to the channel name
4. Click **Connectors**

### Step 2: Configure Incoming Webhook

1. In the Connectors search box, search for **Incoming Webhook**
2. Click **Configure** next to Incoming Webhook
3. Enter a name for the webhook (e.g., `LangGraph Monitor`)
4. Optionally upload an icon image
5. Click **Create**
6. You will see a URL that looks like:

```
https://your-org.webhook.office.com/webhookb2/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx@xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx/IncomingWebhook/xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx/xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
```

7. Click **Copy** to copy this URL, then click **Done**

> Note: If you do not see the Connectors option, your Teams admin may have
> disabled it. Contact your IT administrator to enable it for your channel.

### Step 3: Configure in Python

Add the webhook URL to your `.env` file:

```bash
TEAMS_WEBHOOK_URL=https://your-org.webhook.office.com/webhookb2/...
```

Access it in Python:

```python
import os
from dotenv import load_dotenv

load_dotenv()
webhook_url = os.getenv("TEAMS_WEBHOOK_URL")
```

### Step 4: Test the Connection

```python
import requests, json, os
from dotenv import load_dotenv

load_dotenv()

payload = {
    "@type": "MessageCard",
    "@context": "http://schema.org/extensions",
    "themeColor": "0076D7",
    "summary": "Test Alert",
    "sections": [{
        "activityTitle": "Test Alert from langchain-debug-demo",
        "activityText": "Teams webhook is configured correctly.",
    }]
}

response = requests.post(
    os.getenv("TEAMS_WEBHOOK_URL"),
    data=json.dumps(payload),
    headers={"Content-Type": "application/json"}
)
print(response.status_code)  # Should print 200
```

### Step 5: Newer Teams Setup (if Connectors is disabled)

Microsoft is migrating Teams connectors to Power Automate workflows.
If Incoming Webhooks are disabled in your organization:

1. Go to [https://make.powerautomate.com](https://make.powerautomate.com)
2. Create a new flow with trigger **When an HTTP request is received**
3. Add an action **Post message in a chat or channel** (Teams)
4. Copy the auto-generated HTTP POST URL
5. Use that URL as your `TEAMS_WEBHOOK_URL`

### Security Notes

- Store the webhook URL only in `.env`, never in source code
- Add `.env` to `.gitignore`
- Teams webhook URLs include embedded authentication tokens, treat them as secrets
- Revoke and recreate the webhook in Teams settings if it is ever exposed

---

## 6. Debugging Philosophy: LangChain vs LangGraph

Understanding why the debugging approaches differ is as important as knowing the tools.

### LangChain (Event-Driven)

LangChain treats execution as a **relay race**. Each step hands data to the next.
Debugging focuses on the handoff between steps: what was passed in, what came out.

The right mental model: **"What event happened at step N?"**

Tools from least to most precise:
1. `set_debug(True)` - see everything, terminal only
2. `FileCallbackHandler` - persist the same output to a file
3. `BaseCallbackHandler` subclass - filter to exactly what you care about

### LangGraph (State-Snapshot-Driven)

LangGraph treats execution as a **video game save system**. The State object is
explicit and saved at every node. Debugging focuses on what the State looked like
before and after each node ran.

The right mental model: **"What did the State object contain at node N?"**

Tools from least to most precise:
1. `stream_mode="debug"` - see every task and state update in real time
2. `get_state()` - inspect the full state after the run
3. `get_state_history()` + Time Travel - rewind to any checkpoint and replay

### Why This Difference Matters

In LangChain, if your agent loops, you grep your log file for repeated tool calls.
In LangGraph, if your agent loops, you call `get_state_history()` and see exactly
which node cycled, what the state looked like each time, and rewind to the
last clean state to replay with your fix applied.

---

## 7. Observability Tools Comparison

These tools are excellent complements to the techniques above, especially for
non-developers who need cost tracking and visual tracing.

| Tool | Free Tier | Paid Start | Best For |
|---|---|---|---|
| LangSmith | 5k traces, 14-day retention | $39/seat | Deep LangChain/LangGraph trace UI |
| Langfuse | 50k events/month | $29/month | Open-source, self-hostable option |
| Arize Phoenix | Open source (local) | Cloud plans vary | LlamaIndex + LLM evals |
| Braintrust | Free tier available | $249/month | Eval-first teams, dataset management |

**When to use observability tools vs. custom callbacks:**

Use custom callbacks when you need surgical precision at the code level,
when you are mid-sprint and cannot add a new platform dependency,
or when you need persistent structured logs that feed into your existing
monitoring stack (Datadog, LogRocket, CloudWatch).

Use observability tools when non-engineers need to review cost and utilization,
when you need visual trace UI for complex multi-agent workflows,
or when you need long-term trace retention for compliance.

They are not alternatives. They are different layers of the same observability stack.

---

## 8. Key Concepts Reference

### LangChain Callback Events

| Event | When It Fires | Best Used For |
|---|---|---|
| on_llm_start | LLM receives a prompt | Log prompts for analysis |
| on_llm_end | LLM returns a response | Log responses, capture tokens |
| on_llm_error | LLM call fails | Rate limit / timeout alerting |
| on_tool_start | Agent calls a tool | Track deployment step starts |
| on_tool_end | Tool returns output | Verify tool outputs |
| on_tool_error | Tool raises exception | Tool failure alerting |
| on_agent_action | ReAct thought/action cycle | See agent reasoning |
| on_agent_finish | Agent produces final answer | Log final outcomes |
| on_chain_error | Entire chain fails | Critical failure alerting |

### LangGraph Debugging Tools

| Tool | What It Shows | When To Use |
|---|---|---|
| stream_mode="debug" | Every task and state update in real time | First look at execution flow |
| get_state() | Full State snapshot after run | Understand final state |
| get_state_history() | All checkpoints, newest first | Find where loop started |
| Time Travel (re-invoke with checkpoint config) | Replay from saved state | Test fix without full restart |
| LangGraph Studio | Visual graph IDE with breakpoints | Complex multi-agent workflows |

---

## Author

Chandra Mohan Busam
Principal Engineer | AI Engineer
[GitHub](https://github.com/ChandraMohanBusam) |
[LinkedIn](https://linkedin.com/in/chandramohanbusam)
