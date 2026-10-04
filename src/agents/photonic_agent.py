"""Coordinator agent for the SiN Photonic Waveguide MCP system."""
from strands import Agent

from src.agents.model import create_bedrock_model
from src.agents.sub_agent_tools import (
    ask_analysis_agent,
    ask_optimization_agent,
    ask_prediction_agent,
)
from src.agents.system_prompts import COORDINATOR_PROMPT
from src.config.agent_config import MCP_SERVER_URL
from src.tools.physics_tools import solve_mode
from src.tools.optimization_tools import optimize_design
from src.tools.mask_tools import generate_foundry_mask
from src.tools.prediction_tools import predict_loss
from src.tools.data_tools import query_dataset, query_low_loss_waveguides, read_dataset_columns
from src.tools.visualization_tools import plot_chart


def create_coordinator_agent() -> Agent:
    """Create and configure the coordinator agent with all tools."""
    # callback_handler=None: the REPL prints the final response itself; the
    # default handler would also stream it to stdout, printing it twice.
    return Agent(
        model=create_bedrock_model(),
        system_prompt=COORDINATOR_PROMPT,
        tools=[
            solve_mode,
            optimize_design,
            generate_foundry_mask,
            predict_loss,
            query_dataset,
            query_low_loss_waveguides,
            read_dataset_columns,
            plot_chart,
            ask_prediction_agent,
            ask_optimization_agent,
            ask_analysis_agent,
        ],
        callback_handler=None,
    )


def main() -> None:
    agent = create_coordinator_agent()
    print("🔬 SiN Photonic Waveguide MCP Agent Ready")
    print(f"   MCP Physics Server must be running on {MCP_SERVER_URL}")
    print("   Type your query or 'quit' to exit.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break
        response = agent(user_input)
        print(f"\nAgent: {response}\n")


if __name__ == "__main__":
    main()
