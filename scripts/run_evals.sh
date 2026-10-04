#!/bin/bash
# Run the evaluation suite.
# Use `python -m pytest` so the repo root lands on sys.path and
# `mcp_server` / `src` imports resolve without installing the package.
# tests/test_eval_scenarios.py checks tests/evals/eval_scenarios.json against
# the MCP tool schemas; tests/test_mcp_integration.py exercises a running server.
python -m pytest tests/ -v
