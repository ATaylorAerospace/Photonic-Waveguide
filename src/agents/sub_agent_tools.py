"""Sub-agents exposed to the coordinator as tools (the "Agents as Tools" pattern).

Each sub-agent is created on first use and kept for the life of the process,
so it carries its own conversation context across delegated queries.
"""
from functools import lru_cache

from strands import tool

from src.agents.analysis_agent import create_analysis_agent
from src.agents.optimization_agent import create_optimization_agent
from src.agents.prediction_agent import create_prediction_agent


@lru_cache(maxsize=1)
def _prediction_agent():
    return create_prediction_agent()


@lru_cache(maxsize=1)
def _optimization_agent():
    return create_optimization_agent()


@lru_cache(maxsize=1)
def _analysis_agent():
    return create_analysis_agent()


@tool
def ask_prediction_agent(query: str) -> str:
    """Delegate to the Prediction sub-agent for fast first-pass loss estimates,
    optionally validated against the exact mode solver.

    Args:
        query: The request, including all waveguide geometry and process parameters.
    """
    return str(_prediction_agent()(query))


@tool
def ask_optimization_agent(query: str) -> str:
    """Delegate to the Optimization sub-agent for inverse design of waveguide
    geometry toward a target metric.

    Args:
        query: The design goal, target metric and value, and any constraints.
    """
    return str(_optimization_agent()(query))


@tool
def ask_analysis_agent(query: str) -> str:
    """Delegate to the Analysis sub-agent for dataset statistics, batch
    comparisons and parameter correlations.

    Args:
        query: The analysis question, naming the columns, filters or batches of interest.
    """
    return str(_analysis_agent()(query))
