"""
callbacks.py
Custom LangChain Callback Handlers for the AI Deployment Agent

Handlers included:
1. DeploymentAuditHandler   - Logs every LLM prompt, response, tool call, and error to file
2. TokenUsageHandler        - Tracks token consumption per run
3. LoopDetectionHandler     - Detects when the agent is repeating the same tool call
4. SlackAlertHandler        - Sends error and loop alerts to a Slack channel
5. TeamsAlertHandler        - Sends error and loop alerts to a Microsoft Teams channel

Alert channel (Slack vs Teams) is controlled by config.json -> alerts.provider.
No code changes needed to switch channels.
"""

import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import requests
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.file import FileCallbackHandler
from langchain.schema import AgentAction, AgentFinish, LLMResult

# ---------------------------------------------------------------------------
# Logging setup - writes to langchain_agent/logs/deployment_audit.log
# ---------------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

audit_logger = logging.getLogger("DeploymentAudit")
audit_file_handler = logging.FileHandler("logs/deployment_audit.log")
audit_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
audit_logger.addHandler(audit_file_handler)

token_logger = logging.getLogger("TokenUsage")
token_file_handler = logging.FileHandler("logs/token_usage.log")
token_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(message)s")
)
token_logger.addHandler(token_file_handler)

loop_logger = logging.getLogger("LoopDetection")
loop_file_handler = logging.FileHandler("logs/loop_detection.log")
loop_file_handler.setFormatter(
    logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
)
loop_logger.addHandler(loop_file_handler)


# ---------------------------------------------------------------------------
# 1. DeploymentAuditHandler
#    Captures: every prompt sent to LLM, every LLM response, every tool call,
#    every tool result, every agent action, and all errors.
#    Writes to: logs/deployment_audit.log
# ---------------------------------------------------------------------------
class DeploymentAuditHandler(BaseCallbackHandler):
    """
    Surgical audit logging for the AI Deployment Agent.

    Unlike set_debug(True) which dumps everything to terminal,
    this handler captures only what matters for deployment debugging
    and writes it to a persistent log file.
    """

    def __init__(self):
        self.step_count = 0
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        audit_logger.info(f"=== NEW AGENT RUN STARTED | run_id={self.run_id} ===")

    # --- LLM Events ---

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs
    ) -> None:
        """Fires when the LLM receives a prompt. Logs the full prompt for inspection."""
        self.step_count += 1
        audit_logger.info(
            f"[Step {self.step_count}] LLM START | "
            f"model={serialized.get('name', 'unknown')} | "
            f"prompt_preview={prompts[0][:200].replace(chr(10), ' ')}..."
        )

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Fires when the LLM responds. Logs the response text."""
        if response.generations and response.generations[0]:
            text = response.generations[0][0].text
            audit_logger.info(
                f"[Step {self.step_count}] LLM END | "
                f"response_preview={text[:300].replace(chr(10), ' ')}..."
            )

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Fires on LLM failure (timeout, rate limit, etc.)."""
        audit_logger.error(
            f"[Step {self.step_count}] LLM ERROR | error={str(error)}"
        )

    # --- Tool Events ---

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Fires when a tool is invoked. Critical for tracking deployment steps."""
        tool_name = serialized.get("name", "unknown_tool")
        audit_logger.info(
            f"[Step {self.step_count}] TOOL START | "
            f"tool={tool_name} | input={input_str}"
        )

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Fires when a tool completes. Logs the output for verification."""
        audit_logger.info(
            f"[Step {self.step_count}] TOOL END | "
            f"output={output[:300]}"
        )

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Fires when a tool raises an exception."""
        audit_logger.error(
            f"[Step {self.step_count}] TOOL ERROR | error={str(error)}"
        )

    # --- Agent Events ---

    def on_agent_action(self, action: AgentAction, **kwargs) -> None:
        """Fires on each ReAct thought/action cycle. Shows the agent's reasoning."""
        audit_logger.info(
            f"[Step {self.step_count}] AGENT ACTION | "
            f"tool={action.tool} | "
            f"input={str(action.tool_input)[:200]} | "
            f"thought={action.log[:200].replace(chr(10), ' ')}..."
        )

    def on_agent_finish(self, finish: AgentFinish, **kwargs) -> None:
        """Fires when the agent produces a final answer."""
        audit_logger.info(
            f"[Step {self.step_count}] AGENT FINISH | "
            f"run_id={self.run_id} | "
            f"output={str(finish.return_values)[:300]}"
        )
        audit_logger.info(
            f"=== AGENT RUN COMPLETE | run_id={self.run_id} | total_steps={self.step_count} ==="
        )

    # --- Chain Events ---

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs
    ) -> None:
        """Fires when the overall chain/agent starts."""
        audit_logger.info(
            f"CHAIN START | chain={serialized.get('name', 'unknown')} | "
            f"input_keys={list(inputs.keys())}"
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Fires when the chain completes."""
        audit_logger.info(
            f"CHAIN END | output_keys={list(outputs.keys())}"
        )

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Fires when the entire chain fails. This is the final failure point."""
        audit_logger.error(
            f"CHAIN ERROR | run_id={self.run_id} | error={str(error)}"
        )


# ---------------------------------------------------------------------------
# 2. TokenUsageHandler
#    Tracks prompt tokens, completion tokens, and total tokens per LLM call.
#    Writes cumulative totals to: logs/token_usage.log
#    Note: Does not work with streaming=True (tokens not returned in chunks).
# ---------------------------------------------------------------------------
class TokenUsageHandler(BaseCallbackHandler):
    """
    Tracks token consumption across the entire agent run.

    Important: streaming=True disables token reporting from the LLM provider.
    If you use streaming, switch to tiktoken for manual counting.
    """

    def __init__(self):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_tokens = 0
        self.llm_call_count = 0

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Captures token metrics from the LLM provider's response."""
        if not response.llm_output:
            return

        # OpenAI uses 'token_usage', Anthropic may use different keys
        # Always print response.llm_output first when switching providers
        token_info = response.llm_output.get("token_usage", {})

        prompt_tokens = token_info.get("prompt_tokens", 0)
        completion_tokens = token_info.get("completion_tokens", 0)
        total_tokens = token_info.get("total_tokens", 0)

        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_tokens += total_tokens
        self.llm_call_count += 1

        token_logger.info(
            f"LLM Call #{self.llm_call_count} | "
            f"prompt={prompt_tokens} | completion={completion_tokens} | "
            f"total={total_tokens} | "
            f"cumulative_total={self.total_tokens}"
        )

        print(
            f"[TokenUsage] Call #{self.llm_call_count}: "
            f"prompt={prompt_tokens}, completion={completion_tokens}, "
            f"total={total_tokens} | Run total so far: {self.total_tokens}"
        )

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Prints a final token summary when the agent completes."""
        token_logger.info(
            f"RUN COMPLETE | total_llm_calls={self.llm_call_count} | "
            f"total_prompt_tokens={self.total_prompt_tokens} | "
            f"total_completion_tokens={self.total_completion_tokens} | "
            f"grand_total_tokens={self.total_tokens}"
        )
        print(
            f"\n[TokenUsage SUMMARY] "
            f"LLM calls: {self.llm_call_count} | "
            f"Prompt tokens: {self.total_prompt_tokens} | "
            f"Completion tokens: {self.total_completion_tokens} | "
            f"Grand total: {self.total_tokens}"
        )


# ---------------------------------------------------------------------------
# 3. LoopDetectionHandler
#    Detects when the agent calls the same tool with the same input repeatedly.
#    Triggers a warning after LOOP_THRESHOLD repeated calls.
#    Writes to: logs/loop_detection.log
# ---------------------------------------------------------------------------
LOOP_THRESHOLD = 2  # Default - overridden by config.json -> logging.loop_detection_log.loop_threshold


class LoopDetectionHandler(BaseCallbackHandler):
    """
    Detects infinite loops in ReAct agents.

    A loop is defined as: the same tool called with the same input
    more than threshold times in a single run.

    threshold and the alert handler are injected by ConfigLoader at startup,
    so no code change is needed to adjust sensitivity or switch alert channels.

    This is the handler that would have caught the TransferToServer loop
    in the AI Deployment Agent scenario.
    """

    def __init__(self, slack_handler=None, threshold: int = LOOP_THRESHOLD):
        # Tracks (tool_name, input_str) call frequency
        self.call_counter: Counter = Counter()
        self.threshold = threshold
        # Generic alert_handler: accepts either SlackAlertHandler or TeamsAlertHandler
        # ConfigLoader passes whichever is active based on config.json -> alerts.provider
        self.alert_handler = slack_handler  # named slack_handler for backward compat

    def on_tool_start(
        self, serialized: Dict[str, Any], input_str: str, **kwargs
    ) -> None:
        """Increments counter for each tool call and checks for loop pattern."""
        tool_name = serialized.get("name", "unknown_tool")
        call_key = f"{tool_name}::{input_str}"
        self.call_counter[call_key] += 1
        count = self.call_counter[call_key]

        if count > self.threshold:
            message = (
                f"LOOP DETECTED: Tool '{tool_name}' called {count} times "
                f"with identical input: '{input_str[:100]}'. "
                f"Agent may be stuck. Check tool output handling."
            )
            loop_logger.warning(message)
            print(f"\n[LoopDetection] WARNING: {message}\n")

            # Escalate to configured alert channel (Slack or Teams)
            if self.alert_handler:
                self.alert_handler.send_loop_alert(tool_name, input_str, count)

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Logs a summary of all tool call frequencies for post-run analysis."""
        if self.call_counter:
            loop_logger.info(
                f"TOOL CALL FREQUENCY SUMMARY: {dict(self.call_counter)}"
            )


# ---------------------------------------------------------------------------
# 4. SlackAlertHandler
#    Sends real-time error and loop alerts to a Slack channel via Incoming Webhook.
#    Configure SLACK_WEBHOOK_URL in your .env file.
# ---------------------------------------------------------------------------
class SlackAlertHandler(BaseCallbackHandler):
    """
    Sends error alerts to a Slack channel using an Incoming Webhook.

    Setup:
        1. Create a Slack Incoming Webhook (see README.md for full guide)
        2. Add SLACK_WEBHOOK_URL=https://hooks.slack.com/... to your .env
        3. Set config.json -> alerts.provider = "slack"

    notify_on is injected by ConfigLoader from config.json -> alerts.notify_on.
    Set individual flags to false to suppress specific alert types without
    removing the handler from the stack.

    Differentiating on_llm_error vs on_chain_error:
        - on_llm_error fires on every individual LLM call failure (may retry)
        - on_chain_error fires ONCE when the entire agent gives up
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        notify_on: Optional[dict] = None,
    ):
        self.webhook_url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
        # notify_on controls which event types trigger alerts
        # Defaults to all enabled if not provided
        self.notify_on = notify_on or {
            "llm_error": True,
            "chain_error": True,
            "tool_error": True,
            "loop_detected": True,
        }
        if not self.webhook_url:
            print(
                "[SlackAlertHandler] WARNING: SLACK_WEBHOOK_URL not set. "
                "Alerts will be printed to console only."
            )

    def _post_to_slack(self, payload: dict) -> None:
        """Posts a JSON payload to the configured Slack webhook URL."""
        if not self.webhook_url:
            print(f"[SlackAlert - Console Fallback] {payload.get('text', '')}")
            return
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if response.status_code != 200:
                print(
                    f"[SlackAlertHandler] Failed to post alert. "
                    f"Status: {response.status_code} | Response: {response.text}"
                )
        except requests.exceptions.RequestException as e:
            print(f"[SlackAlertHandler] Network error posting to Slack: {e}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Fires on individual LLM call failure. Useful for rate limit and timeout alerts."""
        if not self.notify_on.get("llm_error", True):
            return
        payload = {
            "text": (
                f":warning: *LLM Error - AI Deployment Agent*\n"
                f">*Error:* `{str(error)}`\n"
                f">*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f">*Action:* Individual LLM call failed (may auto-retry)"
            )
        }
        self._post_to_slack(payload)

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Fires when the entire agent chain fails. This is the critical alert."""
        if not self.notify_on.get("chain_error", True):
            return
        payload = {
            "text": (
                f":fire: *CRITICAL - Agent Chain Failure - AI Deployment Agent*\n"
                f">*Error:* `{str(error)}`\n"
                f">*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f">*Action:* Agent has stopped. Manual intervention required.\n"
                f">*Check:* logs/deployment_audit.log for full trace"
            )
        }
        self._post_to_slack(payload)

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        """Fires when a deployment tool raises an exception."""
        if not self.notify_on.get("tool_error", True):
            return
        payload = {
            "text": (
                f":x: *Tool Error - AI Deployment Agent*\n"
                f">*Error:* `{str(error)}`\n"
                f">*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f">*Action:* A deployment step failed. Check tool logs."
            )
        }
        self._post_to_slack(payload)

    def send_loop_alert(self, tool_name: str, input_str: str, count: int) -> None:
        """Called directly by LoopDetectionHandler when a loop is confirmed."""
        if not self.notify_on.get("loop_detected", True):
            return
        payload = {
            "text": (
                f":repeat: *Loop Detected - AI Deployment Agent*\n"
                f">*Tool:* `{tool_name}`\n"
                f">*Called:* {count} times with same input\n"
                f">*Input:* `{input_str[:150]}`\n"
                f">*Time:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f">*Action:* Agent may be stuck. Review tool output parsing."
            )
        }
        self._post_to_slack(payload)


# ---------------------------------------------------------------------------
# 5. TeamsAlertHandler
#    Mirror of SlackAlertHandler but posts to Microsoft Teams via Incoming Webhook.
#    Activated when config.json -> alerts.provider = "teams".
#    Uses Teams MessageCard format for broad compatibility.
# ---------------------------------------------------------------------------
TEAMS_SEVERITY_COLORS = {
    "warning": "FFC107",
    "error": "D32F2F",
    "info": "0076D7",
}


class TeamsAlertHandler(BaseCallbackHandler):
    """
    Sends error alerts to a Microsoft Teams channel using an Incoming Webhook.

    Setup:
        1. Create a Teams Incoming Webhook (see README.md for full guide)
        2. Add TEAMS_WEBHOOK_URL=https://your-org.webhook.office.com/... to your .env
        3. Set config.json -> alerts.provider = "teams"

    notify_on is injected by ConfigLoader from config.json -> alerts.notify_on.
    Identical event coverage as SlackAlertHandler so you can switch channels
    by changing one line in config.json.
    """

    def __init__(
        self,
        webhook_url: Optional[str] = None,
        notify_on: Optional[dict] = None,
    ):
        self.webhook_url = webhook_url or os.getenv("TEAMS_WEBHOOK_URL")
        self.notify_on = notify_on or {
            "llm_error": True,
            "chain_error": True,
            "tool_error": True,
            "loop_detected": True,
        }
        if not self.webhook_url:
            print(
                "[TeamsAlertHandler] WARNING: TEAMS_WEBHOOK_URL not set. "
                "Alerts will be printed to console only."
            )

    def _post_to_teams(self, title: str, message: str, color: str = "0076D7") -> None:
        """Posts a MessageCard to the configured Teams webhook URL."""
        payload = {
            "@type": "MessageCard",
            "@context": "http://schema.org/extensions",
            "themeColor": color,
            "summary": title,
            "sections": [{
                "activityTitle": f"**{title}**",
                "activitySubtitle": (
                    f"AI Deployment Agent | "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                ),
                "activityText": message,
                "markdown": True,
            }],
        }
        if not self.webhook_url:
            print(f"[TeamsAlert - Console Fallback] {title}: {message}")
            return
        try:
            response = requests.post(
                self.webhook_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
                timeout=5,
            )
            if response.status_code != 200:
                print(
                    f"[TeamsAlertHandler] Failed. "
                    f"Status: {response.status_code} | {response.text}"
                )
        except requests.exceptions.RequestException as e:
            print(f"[TeamsAlertHandler] Network error: {e}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        if not self.notify_on.get("llm_error", True):
            return
        self._post_to_teams(
            title="LLM Error - AI Deployment Agent",
            message=f"Individual LLM call failed (may auto-retry).\n\nError: `{str(error)}`",
            color=TEAMS_SEVERITY_COLORS["warning"],
        )

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        if not self.notify_on.get("chain_error", True):
            return
        self._post_to_teams(
            title="CRITICAL - Agent Chain Failure - AI Deployment Agent",
            message=(
                f"Agent has stopped. Manual intervention required.\n\n"
                f"Error: `{str(error)}`\n\n"
                f"Check: logs/deployment_audit.log for full trace."
            ),
            color=TEAMS_SEVERITY_COLORS["error"],
        )

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs
    ) -> None:
        if not self.notify_on.get("tool_error", True):
            return
        self._post_to_teams(
            title="Tool Error - AI Deployment Agent",
            message=f"A deployment step failed.\n\nError: `{str(error)}`",
            color=TEAMS_SEVERITY_COLORS["error"],
        )

    def send_loop_alert(self, tool_name: str, input_str: str, count: int) -> None:
        """Called directly by LoopDetectionHandler when a loop is confirmed."""
        if not self.notify_on.get("loop_detected", True):
            return
        self._post_to_teams(
            title="Loop Detected - AI Deployment Agent",
            message=(
                f"Tool `{tool_name}` called {count} times with identical input.\n\n"
                f"Input: `{input_str[:150]}`\n\n"
                f"Agent may be stuck. Review tool output parsing."
            ),
            color=TEAMS_SEVERITY_COLORS["warning"],
        )
