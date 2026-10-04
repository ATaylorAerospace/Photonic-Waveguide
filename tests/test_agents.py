"""Tests for agent construction (no model calls are made)."""
import pytest

pytest.importorskip("strands")

SUB_AGENT_TOOLS = {"ask_prediction_agent", "ask_optimization_agent", "ask_analysis_agent"}
DIRECT_TOOLS = {
    "solve_mode", "optimize_design", "generate_foundry_mask", "predict_loss",
    "query_dataset", "query_low_loss_waveguides", "read_dataset_columns", "plot_chart",
}


def test_coordinator_has_direct_and_sub_agent_tools():
    from src.agents.photonic_agent import create_coordinator_agent
    agent = create_coordinator_agent()
    assert DIRECT_TOOLS | SUB_AGENT_TOOLS <= set(agent.tool_names)


def test_sub_agents_use_the_configured_model():
    from src.agents.prediction_agent import create_prediction_agent
    from src.config.aws_config import BEDROCK_MODEL_ID
    agent = create_prediction_agent()
    assert agent.model.config["model_id"] == BEDROCK_MODEL_ID
