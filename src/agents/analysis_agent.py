"""Analysis sub-agent for dataset exploration."""
from strands import Agent

from src.agents.model import create_bedrock_model
from src.agents.system_prompts import ANALYSIS_PROMPT


def create_analysis_agent() -> Agent:
    """Create the analysis sub-agent with data tools."""
    from src.tools.data_tools import query_dataset, query_low_loss_waveguides, read_dataset_columns
    from src.tools.visualization_tools import plot_chart
    # No callback handler: sub-agents run inside a coordinator tool call and
    # must not stream their intermediate text to the console.
    return Agent(
        model=create_bedrock_model(),
        system_prompt=ANALYSIS_PROMPT,
        tools=[query_dataset, query_low_loss_waveguides, read_dataset_columns, plot_chart],
        callback_handler=None,
    )
