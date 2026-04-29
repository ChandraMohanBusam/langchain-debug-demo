"""
config_loader.py
Configuration Loader for the AI Deployment Agent

Reads config.json at startup and dynamically builds the list of
active LangChain callback handlers. The agent.py simply calls:

    handlers = ConfigLoader.build_handlers()

and passes the result to the LLM and AgentExecutor. No code changes
are needed to toggle logging on/off or switch between Slack and Teams.

Config precedence:
    config.json  ->  .env (for webhook URLs)  ->  defaults

Switching alert channels:
    Change "provider" in config.json to "slack", "teams", or "none".
    That is the only change required.
"""

import json
import logging
import os
from pathlib import Path
from typing import List

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.file import FileCallbackHandler

# ---------------------------------------------------------------------------
# Path resolution: config.json lives next to this file
# ---------------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.json"


class ConfigLoader:
    """
    Reads config.json and assembles the active callback handler stack.

    Usage:
        handlers = ConfigLoader.build_handlers()
        llm = ChatOpenAI(callbacks=handlers)
        executor = AgentExecutor(..., callbacks=handlers)
    """

    _config: dict = None

    @classmethod
    def load(cls) -> dict:
        """Loads and caches config.json. Returns the parsed config dict."""
        if cls._config is None:
            if not CONFIG_PATH.exists():
                raise FileNotFoundError(
                    f"config.json not found at {CONFIG_PATH}. "
                    f"Check that config.json is in the langchain_agent/ directory."
                )
            with open(CONFIG_PATH, "r") as f:
                cls._config = json.load(f)
            print(f"[ConfigLoader] Loaded config from {CONFIG_PATH}")
        return cls._config

    @classmethod
    def build_handlers(cls) -> List[BaseCallbackHandler]:
        """
        Reads config.json and builds the active handler list.

        Returns:
            List of instantiated callback handlers ready to pass to
            ChatOpenAI(callbacks=...) and AgentExecutor(callbacks=...).
        """
        config = cls.load()
        logging_cfg = config.get("logging", {})
        alerts_cfg = config.get("alerts", {})
        handlers = []

        # Set global log level from config
        log_level_str = logging_cfg.get("log_level", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
        logging.getLogger().setLevel(log_level)
        print(f"[ConfigLoader] Log level set to: {log_level_str}")

        # Import here to avoid circular imports
        from callbacks import (
            DeploymentAuditHandler,
            LoopDetectionHandler,
            SlackAlertHandler,
            TeamsAlertHandler,
            TokenUsageHandler,
        )

        # Build alert handler first (loop detection depends on it)
        alert_handler = cls._build_alert_handler(alerts_cfg)
        notify_on = alerts_cfg.get("notify_on", {})

        # FileCallbackHandler
        file_cb_cfg = logging_cfg.get("file_callback_handler", {})
        if file_cb_cfg.get("enabled", False):
            output_file = file_cb_cfg.get(
                "output_file", "logs/langchain_file_callback.log"
            )
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            handlers.append(FileCallbackHandler(output_file))
            print(f"[ConfigLoader] FileCallbackHandler enabled -> {output_file}")

        # DeploymentAuditHandler
        audit_cfg = logging_cfg.get("audit_log", {})
        if audit_cfg.get("enabled", False):
            handlers.append(DeploymentAuditHandler())
            print(
                f"[ConfigLoader] DeploymentAuditHandler enabled -> "
                f"{audit_cfg.get('output_file', 'logs/deployment_audit.log')}"
            )

        # TokenUsageHandler
        token_cfg = logging_cfg.get("token_usage_log", {})
        if token_cfg.get("enabled", False):
            handlers.append(TokenUsageHandler())
            print(
                f"[ConfigLoader] TokenUsageHandler enabled -> "
                f"{token_cfg.get('output_file', 'logs/token_usage.log')}"
            )

        # LoopDetectionHandler
        loop_cfg = logging_cfg.get("loop_detection_log", {})
        if loop_cfg.get("enabled", False):
            loop_threshold = loop_cfg.get("loop_threshold", 2)
            # Wire alert handler only if loop_detected notifications are on
            loop_alert = (
                alert_handler
                if notify_on.get("loop_detected", True)
                else None
            )
            handlers.append(
                LoopDetectionHandler(
                    slack_handler=loop_alert,
                    threshold=loop_threshold,
                )
            )
            print(
                f"[ConfigLoader] LoopDetectionHandler enabled -> "
                f"threshold={loop_threshold} | "
                f"alert_on_loop={loop_alert is not None}"
            )

        # Alert handler (add to stack so it receives chain/llm/tool events)
        if alert_handler is not None:
            handlers.append(alert_handler)
            provider = alerts_cfg.get("provider", "none")
            print(f"[ConfigLoader] Alert handler enabled -> provider={provider}")

        if not handlers:
            print(
                "[ConfigLoader] WARNING: No handlers enabled in config.json. "
                "The agent will run without any logging or alerting."
            )

        print(
            f"[ConfigLoader] Active handlers: "
            f"{[type(h).__name__ for h in handlers]}"
        )
        return handlers

    @classmethod
    def _build_alert_handler(cls, alerts_cfg: dict):
        """
        Builds and returns the configured alert handler, or None if disabled.

        Reads 'provider' from config to decide between Slack, Teams, or none.
        Reads the webhook URL from the environment variable named in config.
        """
        from callbacks import SlackAlertHandler, TeamsAlertHandler

        provider = alerts_cfg.get("provider", "none").lower()
        notify_on = alerts_cfg.get("notify_on", {})

        # If all notification types are off, skip building a handler
        any_enabled = any(
            v for k, v in notify_on.items() if k != "description"
        )
        if not any_enabled:
            print("[ConfigLoader] All alert types disabled in config. No alert handler created.")
            return None

        if provider == "slack":
            env_var = alerts_cfg.get("slack", {}).get(
                "webhook_url_env", "SLACK_WEBHOOK_URL"
            )
            webhook_url = os.getenv(env_var)
            if not webhook_url:
                print(
                    f"[ConfigLoader] WARNING: provider=slack but {env_var} "
                    f"is not set in .env. Alerts will fall back to console."
                )
            return SlackAlertHandler(
                webhook_url=webhook_url,
                notify_on=notify_on,
            )

        elif provider == "teams":
            env_var = alerts_cfg.get("teams", {}).get(
                "webhook_url_env", "TEAMS_WEBHOOK_URL"
            )
            webhook_url = os.getenv(env_var)
            if not webhook_url:
                print(
                    f"[ConfigLoader] WARNING: provider=teams but {env_var} "
                    f"is not set in .env. Alerts will fall back to console."
                )
            return TeamsAlertHandler(
                webhook_url=webhook_url,
                notify_on=notify_on,
            )

        elif provider == "none":
            print("[ConfigLoader] Alert provider set to 'none'. No alerts will be sent.")
            return None

        else:
            print(
                f"[ConfigLoader] Unknown provider '{provider}' in config.json. "
                f"Valid values: 'slack', 'teams', 'none'. No alert handler created."
            )
            return None

    @classmethod
    def get(cls, *keys, default=None):
        """
        Convenience method to read a nested config value.

        Usage:
            loop_threshold = ConfigLoader.get("logging", "loop_detection_log", "loop_threshold", default=2)
        """
        config = cls.load()
        value = config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value
