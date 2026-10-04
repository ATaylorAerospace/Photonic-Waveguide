"""Optimization sub-agent for inverse design via MCP."""
from strands import Agent

from src.agents.model import create_bedrock_model
from src.agents.system_prompts import OPTIMIZATION_PROMPT


def create_optimization_agent() -> Agent:
    """Create the optimization sub-agent with MCP inverse design tools."""
    from src.tools.optimization_tools import optimize_design
    from src.tools.physics_tools import solve_mode
    # No callback handler: sub-agents run inside a coordinator tool call and
    # must not stream their intermediate text to the console.
    return Agent(
        model=create_bedrock_model(),
        system_prompt=OPTIMIZATION_PROMPT,
        tools=[optimize_design, solve_mode],
        callback_handler=None,
    )
