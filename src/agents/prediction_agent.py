"""Prediction sub-agent for fast-pass ML inference."""
from strands import Agent
from src.agents.system_prompts import PREDICTION_PROMPT


def create_prediction_agent() -> Agent:
    """Create the prediction sub-agent with XGBoost tools."""
    from src.tools.prediction_tools import predict_loss
    from src.tools.physics_tools import solve_mode
    return Agent(system_prompt=PREDICTION_PROMPT, tools=[predict_loss, solve_mode])
