"""
tools.py
AI Deployment Agent - Tool Definitions
Each tool simulates a real deployment step.
The looping scenario: TransferToServer cannot confirm the file,
so the agent loops between TransferToServer and DeployOnServer.
"""

import time
import random
from langchain.tools import tool

# Simulated server state (in-memory for demo purposes)
_server_state = {
    "build_downloaded": False,
    "file_transferred": False,
    "transfer_attempts": 0,
    "deployed": False,
    "services_restarted": False,
}


@tool
def DownloadBuild(version: str) -> str:
    """
    Downloads the build artifact for a given version from the artifact store.
    Call this first before any transfer or deployment step.
    Args:
        version: The version string of the build to download (e.g., '2.1.4').
    """
    print(f"[DownloadBuild] Downloading build version: {version}")
    time.sleep(0.5)  # Simulate download time
    _server_state["build_downloaded"] = True
    return f"Build version {version} downloaded successfully to /tmp/builds/{version}.zip"


@tool
def TransferToServer(version: str, server_ip: str) -> str:
    """
    Transfers the downloaded build artifact to the target deployment server via SCP.
    Requires DownloadBuild to be called first.
    Args:
        version: The version string of the build to transfer.
        server_ip: The IP address of the target server.
    """
    if not _server_state["build_downloaded"]:
        return "ERROR: Build not downloaded yet. Call DownloadBuild first."

    _server_state["transfer_attempts"] += 1
    attempt = _server_state["transfer_attempts"]

    print(f"[TransferToServer] Attempt #{attempt} - Transferring {version} to {server_ip}")
    time.sleep(0.5)

    # Simulate flaky transfer: fails first 2 attempts, succeeds on 3rd
    # This is the root cause of the looping behavior
    if attempt < 3:
        # Transfer happens but verification fails (file size mismatch simulation)
        return (
            f"WARNING: Transfer attempt #{attempt} completed but file verification failed. "
            f"Expected checksum does not match. File may be corrupted. "
            f"Recommend retrying transfer before deploying."
        )

    _server_state["file_transferred"] = True
    return (
        f"Build {version} successfully transferred to {server_ip}:/opt/deployments/{version}.zip. "
        f"Checksum verified. Ready for deployment."
    )


@tool
def DeployOnServer(version: str, server_ip: str) -> str:
    """
    Unpacks and deploys the transferred build artifact on the target server.
    Requires TransferToServer to complete successfully with verified checksum.
    Args:
        version: The version string to deploy.
        server_ip: The IP address of the server to deploy on.
    """
    if not _server_state["file_transferred"]:
        return (
            f"ERROR: Cannot deploy. File transfer not verified for version {version}. "
            f"Please ensure TransferToServer completed with a verified checksum."
        )

    print(f"[DeployOnServer] Deploying {version} on {server_ip}")
    time.sleep(0.5)
    _server_state["deployed"] = True
    return f"Version {version} deployed successfully on {server_ip}. Application is running."


@tool
def RestartServices(server_ip: str) -> str:
    """
    Restarts the application services on the target server after deployment.
    Call this as the final step after a successful deployment.
    Args:
        server_ip: The IP address of the server where services should be restarted.
    """
    if not _server_state["deployed"]:
        return "ERROR: Cannot restart services. No active deployment found on server."

    print(f"[RestartServices] Restarting services on {server_ip}")
    time.sleep(0.5)
    _server_state["services_restarted"] = True
    return f"All services restarted successfully on {server_ip}. Deployment complete."


def reset_server_state():
    """Utility to reset server state between demo runs."""
    global _server_state
    _server_state = {
        "build_downloaded": False,
        "file_transferred": False,
        "transfer_attempts": 0,
        "deployed": False,
        "services_restarted": False,
    }
