"""The single agent scaffold shared by every model.

Keeping the loop identical across providers means score differences come from the
model, not from per-provider prompt or harness tweaks.
"""

from __future__ import annotations

import time

from pydantic import BaseModel

from agentarena.agents.base import Model
from agentarena.schemas import Message, Step, Trajectory
from agentarena.tasks import Task
from agentarena.tools import Environment


class Budget(BaseModel):
    max_steps: int = 15
    max_tokens: int = 80_000  # total input+output across the episode
    max_output_tokens: int = 4096  # per turn
    timeout_s: float = 240.0


async def run_agent(model: Model, task: Task, env: Environment, budget: Budget, traj: Trajectory) -> None:
    """Run one episode, filling ``traj`` in place so a timeout keeps the partial trace."""
    messages = [Message(role="system", content=task.system_prompt), Message(role="user", content=task.prompt)]
    tools = env.specs()
    for i in range(budget.max_steps):
        if traj.usage.input_tokens + traj.usage.output_tokens >= budget.max_tokens:
            traj.stop_reason = "max_tokens"
            return
        t0 = time.perf_counter()
        comp = await model.complete(messages, tools, max_tokens=budget.max_output_tokens)
        step = Step(index=i, text=comp.text, tool_calls=comp.tool_calls, usage=comp.usage,
                    latency_s=time.perf_counter() - t0, cached=comp.cached)
        traj.steps.append(step)
        traj.usage = traj.usage + comp.usage
        messages.append(Message(role="assistant", content=comp.text, tool_calls=comp.tool_calls,
                                provider_raw=comp.provider_raw))

        if comp.stop_reason == "refusal":
            traj.stop_reason = "refusal"
            return
        if not comp.tool_calls:
            traj.final_answer = comp.text
            traj.stop_reason = "final"
            return

        step.tool_results = [env.execute(c) for c in comp.tool_calls]
        messages.append(Message(role="tool", tool_results=step.tool_results))
        if env.submission is not None:
            traj.final_answer = str(env.submission.get("answer", ""))
            traj.stop_reason = "final"
            return
    traj.stop_reason = "max_steps"
