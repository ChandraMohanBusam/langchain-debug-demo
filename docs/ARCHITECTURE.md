# Architecture Notes

Some context on why things are designed the way they are, for anyone who wants to go deeper than the README.

---

## Why three debugging levels instead of one

When I first hit the looping problem with the deployment agent, my instinct was to turn on `verbose=True` and read the output. That works, but a 10-step agent run produces a few hundred lines of mixed text in the terminal. You can find the problem if you know what you are looking for, but you cannot grep it, you cannot share it, and it is gone the moment the terminal closes.

`FileCallbackHandler` solves the persistence problem with zero custom code. But the format is still fixed. You get everything or nothing, and there is no way to add alerting when something specific goes wrong.

The custom handler stack is what I actually wanted from the start: structured logs per concern, a loop detector that fires on the third identical tool call rather than after max_iterations exhausts itself, and a Slack or Teams alert that tells me what broke and which tool caused it. The three levels are in the project because they represent a real progression, not because I needed to fill the README.

---

## Why config.json drives the handler stack

The first version of the custom stack had the handlers hardcoded in `agent.py`. Switching from Slack to Teams meant editing a Python file, which is the wrong place to make that kind of decision. The config approach came from the same reasoning that put `agent_config.json` in the deployment agent: behavior that changes between environments or teams should live in config, not in code.

The `ConfigLoader` pattern also makes it easy to disable token tracking in development (where you want fast runs) and re-enable it in production (where you care about cost). Same codebase, different config file.

---

## Why LangGraph needs a completely different approach

The callback approach works well for LangChain because the execution model is linear. Data moves forward through a chain of steps and you attach event listeners at each handoff point. When something goes wrong you look at which event fired and what the inputs and outputs were.

LangGraph has loops. A node can execute multiple times in a single run, and the State object accumulates changes across every execution. If you only track events you can see that `review_node` fired five times, but you cannot easily answer: what did the State look like the third time it fired, and what was different about the fourth time that caused the loop? That is the question the checkpointer answers. `get_state_history()` returns a snapshot of the full State after every node, so you can walk backwards through the history and find exactly where things went wrong.

Time Travel is the practical payoff. Instead of restarting the whole graph after fixing a bug, you replay from the last clean checkpoint. For graphs with expensive LLM calls in early nodes that is not a minor convenience, it meaningfully changes how you iterate.

---

## The alert handler design

`SlackAlertHandler` and `TeamsAlertHandler` implement the same interface intentionally. Both have `send_loop_alert`, `on_llm_error`, `on_chain_error`, and `on_tool_error`. The `LoopDetectionHandler` holds a reference to whichever alert handler is active and calls `send_loop_alert` on it without knowing or caring which channel it is posting to.

This means the config switch from `"provider": "slack"` to `"provider": "teams"` requires no changes anywhere in the handler logic. `ConfigLoader` instantiates the right one and injects it. Everything else stays the same.

One thing worth knowing: `on_llm_error` fires on every individual LLM call failure, including ones the framework retries automatically. `on_chain_error` fires exactly once when the entire agent gives up. In production you probably want a warning-level alert on `on_llm_error` and a critical alert on `on_chain_error`. Treating both the same way creates noise.

---

## LangGraph agent: why a Document Review scenario

The deployment agent scenario is too specific for demonstrating LangGraph concepts cleanly. Deployment is intentionally sequential: download, transfer, deploy, restart. That is a straight line, not a graph.

The document review scenario has a natural loop: classify, review, and if the reviewer requests changes, revise and go back to review. That loop is the reason LangGraph exists and the reason its debugging tools are different. A scenario without a loop would not show why `get_state_history()` matters, or why a revision counter in State is a better loop guard than a callback-level counter.
