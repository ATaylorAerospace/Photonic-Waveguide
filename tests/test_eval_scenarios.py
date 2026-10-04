"""Keep tests/evals/eval_scenarios.json in step with the MCP tool schemas."""
import json
from pathlib import Path

from mcp_server.schemas.waveguide import InverseDesignOutput, MaskGenOutput, ModeSolverOutput

OUTPUT_MODELS = {
    "solve_waveguide_mode": ModeSolverOutput,
    "optimize_waveguide": InverseDesignOutput,
    "generate_mask": MaskGenOutput,
}


def test_scenarios_reference_real_tools_and_fields():
    path = Path(__file__).parent / "evals" / "eval_scenarios.json"
    scenarios = json.loads(path.read_text())
    assert scenarios, "no eval scenarios defined"
    for scenario in scenarios:
        assert scenario["input"].strip()
        assert scenario["expected_tool"] in OUTPUT_MODELS, scenario["expected_tool"]
        fields = set(OUTPUT_MODELS[scenario["expected_tool"]].model_fields)
        missing = set(scenario["expected_fields"]) - fields
        assert not missing, f"{scenario['expected_tool']}: unknown fields {sorted(missing)}"
