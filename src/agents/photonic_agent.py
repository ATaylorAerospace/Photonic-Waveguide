"""Coordinator agent for the SiN Photonic Waveguide MCP system."""
import os
from strands import Agent
from strands.models import BedrockModel
from src.agents.system_prompts import COORDINATOR_PROMPT
from src.tools.physics_tools import solve_mode
from src.tools.optimization_tools import optimize_design
from src.tools.mask_tools import generate_foundry_mask
from src.tools.prediction_tools import predict_loss
from src.tools.data_tools import query_dataset, query_low_loss_waveguides, read_dataset_columns
from src.tools.visualization_tools import plot_chart


def create_coordinator_agent() -> Agent:
    """Create and configure the coordinator agent with all tools."""
    model = BedrockModel(
        model_id=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-sonnet-4-20250514"),
        region_name=os.getenv("AWS_REGION", "us-west-2"),
    )
    agent = Agent(
        model=model,
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
        ],
    )
    return agent


if __name__ == "__main__":
    agent = create_coordinator_agent()
    print("🔬 SiN Photonic Waveguide MCP Agent Ready")
    print("   MCP Physics Server must be running on $MCP_SERVER_URL")
    print("   Type your query or 'quit' to exit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        response = agent(user_input)
        print(f"\nAgent: {response}\n")
