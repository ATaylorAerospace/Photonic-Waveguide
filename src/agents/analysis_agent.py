"""Analysis sub-agent for dataset exploration."""
from strands import Agent
from src.agents.system_prompts import ANALYSIS_PROMPT


def create_analysis_agent() -> Agent:
    """Create the analysis sub-agent with data tools."""
    from src.tools.data_tools import query_dataset
    from src.tools.visualization_tools import plot_chart
    return Agent(system_prompt=ANALYSIS_PROMPT, tools=[query_dataset, plot_chart])
