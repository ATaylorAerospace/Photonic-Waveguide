#!/bin/bash
# Run evaluation suite
# Use `python -m pytest` so the repo root lands on sys.path and
# `mcp_server` / `src` imports resolve without installing the package.
python -m pytest tests/ -v
