"""Optimization sub-agent for inverse design via MCP."""
from strands import Agent
from src.agents.system_prompts import OPTIMIZATION_PROMPT


def create_optimization_agent() -> Agent:
    """Create the optimization sub-agent with MCP inverse design tools."""
    from src.tools.optimization_tools import optimize_design
    from src.tools.physics_tools import solve_mode
    return Agent(system_prompt=OPTIMIZATION_PROMPT, tools=[optimize_design, solve_mode])
