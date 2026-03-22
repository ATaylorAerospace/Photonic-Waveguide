# 🔬 SiN Photonic Waveguide MCP Agent 🚀
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/ATaylorAerospace/Photonic-Waveguide?style=social)](https://github.com/ATaylorAerospace/Photonic-Waveguide)
[![AWS](https://img.shields.io/badge/AWS-Bedrock%20AgentCore-orange.svg)](https://aws.amazon.com/bedrock/)
[![Dataset](https://img.shields.io/badge/HuggingFace-90K%20rows-yellow.svg)](https://huggingface.co/datasets/Taylor658/SiN-photonic-waveguide-loss-efficiency)
[![MCP](https://img.shields.io/badge/MCP-FastMCP%20Server-green.svg)](https://github.com/jlowin/fastmcp)
[![Contact A Taylor](https://img.shields.io/badge/Contact-A%20Taylor-brightgreen.svg?logo=mail.ru&logoColor=white)](https://ataylor.getform.com/5w8wz)

An **Agentic AI Application** for analyzing, predicting, and optimizing Silicon Nitride (Si₃N₄) photonic waveguide performance. Powered by **deterministic physics tools** served via a **FastMCP server** — using **modesolverpy**, **SAX (JAX)**, and **gdsfactory** — orchestrated by **AWS Bedrock Agents**, **Strands Agents SDK**, and **Amazon Bedrock AgentCore** for production-grade agentic infrastructure.

> 🚧 **Status:** Core agents stable · MCP physics server operational · AgentCore deployment ready

---

## 🤔 The Problem

Designing high-performance silicon nitride photonic waveguides is a complex, iterative process:

* **💥 Massive Parameter Space:** Waveguide geometry, cladding materials, deposition methods, etch processes, and annealing conditions create thousands of possible configurations.
* **🌀 Slow Design Cycles:** Traditional trial-and-error fabrication is expensive and time-consuming — each iteration takes weeks.
* **⚠️ Hidden Interactions:** Fabrication parameters interact in non-obvious ways, sidewall roughness, cladding choice, and annealing all affect loss in coupled ways.
* **🔥 Batch Variability:** Real fabrication runs show batch-to-batch variation that is difficult to predict or control without data-driven insights.
* **🧪 Fine-Tuning Limitations:** Fine-tuning a foundation model on waveguide data produces black-box predictions that cannot be mathematically verified, cannot perform true gradient-based inverse design, and cannot generate fabrication-ready outputs.

---

## 💡 The Solution

**SiN Photonic Waveguide MCP Agent** replaces foundation model fine-tuning with **deterministic physics solvers** served via a FastMCP server, complemented by ML models trained on 90,000 waveguide configurations:

* **🔬 Solve waveguide modes** — compute n_eff, optical confinement, mode field diameter, and group index using fully vectorial eigenmode expansion (modesolverpy) in milliseconds.
* **⚡ Inverse design** — find optimal waveguide geometry via gradient-based optimization using automatic differentiation (SAX + JAX), not brute-force search.
* **🏭 Generate fabrication masks** — produce foundry-ready GDSII layout files automatically using gdsfactory with proper Bezier routing and I/O coupling.
* **🔮 Fast-pass ML predictions** — use XGBoost models trained on 90K configurations for instant first-pass propagation loss estimates.
* **📊 Analyze fabrication data** — compare batches, identify yield issues, and uncover parameter correlations from the dataset.
* **🧪 Explain physics** — provide analytical photonic calculations and domain knowledge on demand.

> ⚠️ **Note:** This dataset is synthetic — predictions are based on synthetic training data and may differ from real foundry results.

---

## 🧠 Why MCP Tools Instead of Fine-Tuning

| | Fine-Tuning a Foundation Model | MCP Physics Tools (This Repo) |
| --- | --- | --- |
| **Accuracy** | Limited to training data distribution | Exact physics solutions for any configuration |
| **Optimization** | No true gradient-based inverse design | JAX autodiff finds true global optima |
| **Fabrication** | No layout output | GDSII files ready for foundry submission |
| **Compute** | Expensive retraining cycles | Millisecond deterministic calculations |
| **Explainability** | Black-box predictions | Full mode profiles and convergence history |
| **Validation** | Cannot verify against wave optics | Every prediction validated against eigenmode expansion |

---

## ✨ Features

| Module | Status | Description |
| --- | --- | --- |
| 🔬 Mode Solver Tool | ✅ Live | modesolverpy eigenmode expansion via MCP |
| ⚡ Inverse Design Tool | ✅ Live | SAX + JAX gradient-based optimization via MCP |
| 🏭 Mask Generation Tool | ✅ Live | gdsfactory GDSII layout generation via MCP |
| 🔮 Prediction Agent | ✅ Live | XGBoost fast-pass loss and efficiency prediction |
| ⚙️ Optimization Agent | ✅ Live | Inverse design parameter optimization |
| 📊 Analysis Agent | ✅ Live | Batch statistics and data exploration |
| 🧪 Physics Tools | ✅ Live | Analytical photonic calculations via MCP server |
| 📈 Visualization Tools | ✅ Live | Chart and plot generation |
| 🧠 Knowledge Base | ✅ Live | RAG-based Q&A over waveguide dataset |
| 💾 Session Memory | ✅ Live | Persistent user preferences via AgentCore Memory |
| 🛡️ Safety Policies | ✅ Live | Guardrails for design feasibility and uncertainty |

---

## 🏁 Getting Started

### Prerequisites

* ✅ Python 3.12+
* ✅ AWS Account with Bedrock model access enabled
* ✅ AWS CLI configured (`aws configure`)
* ✅ Docker (for AgentCore deployment)

### 🛠️ Setup

```bash
# Clone the repository
git clone https://github.com/ATaylorAerospace/Photonic-Waveguide-MCP.git
cd Photonic-Waveguide-MCP

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install all dependencies (agent + physics + MCP)
pip install -r requirements.txt

# --- OR install modular groups ---
pip install -e ".[mcp,physics]"

# Download the dataset from HuggingFace (used for batch analysis & RAG)
python data/download_dataset.py

# Train fast-pass XGBoost models (optional — for first-pass loss estimates)
python src/models/train.py

# Start the MCP physics server
python -m mcp_server.server

# In a separate terminal — run the agent
python src/agents/photonic_agent.py
```

### 🧪 Running Tests

```bash
pytest tests/
```

---

## 🛠️ Technology Stack

| Layer | Technology |
| --- | --- |
| 🤖 Agent Framework | Strands Agents SDK (Python) |
| 🧠 LLM | Anthropic Sonnet 4 via Amazon Bedrock |
| 🔬 Mode Solving | modesolverpy (Fully Vectorial EME) |
| ⚡ Inverse Design | SAX + JAX (Autodiff Gradient Descent) |
| 🏭 Mask Generation | gdsfactory + gdstk (GDSII Layout) |
| 🔌 Tool Protocol | FastMCP (Model Context Protocol) |
| ☁️ Infrastructure | Amazon Bedrock AgentCore |
| 📊 Fast Estimation | XGBoost, scikit-learn (first-pass predictions) |
| 📦 Bulk Data | Amazon S3 Express One Zone (Apache Parquet) |
| ⚡ Simulation Cache | Amazon DynamoDB (sub-10ms point lookups) |
| 🔎 Dataset Queries | Amazon S3 Tables (SQL over Parquet in S3) |
| 🔍 Observability | OpenTelemetry + CloudWatch |
| 🐳 Deployment | Docker + AgentCore Runtime |
| 🗄️ Vector Store | Amazon OpenSearch Serverless |
| 🔐 Identity | Amazon Cognito via AgentCore Identity |

---

## 🏛️ Architecture

The application uses a **physics first MCP architecture** where deterministic Python solvers replace foundation model fine-tuning:

```
┌─────────────────────────────────────────────────────────────┐
│                      User / Engineer                        │
└──────────────────────────┬──────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────┐
│  Layer 1: Strands Agents SDK (Agent Framework)              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │Prediction│  │Optimiza- │  │ Analysis │   Coordinator     │
│  │  Agent   │  │tion Agent│  │  Agent   │◄── Agent          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘                  │
│       │              │              │                        │
│  ┌────▼─────┐  ┌────▼─────┐  ┌────▼─────┐                  │
│  │ML Tools  │  │MCP Client│  │Data      │  Physics &        │
│  │(XGBoost) │  │(calls    │  │Tools     │  Viz Tools        │
│  │          │  │ server)  │  │          │                    │
│  └──────────┘  └────┬─────┘  └──────────┘                   │
└──────────────────────┼──────────────────────────────────────┘
                       │ MCP Protocol (JSON-RPC over stdio/SSE)
┌──────────────────────▼──────────────────────────────────────┐
│  Layer 2: FastMCP Physics Server (mcp_server/)              │
│  ┌────────────────┐ ┌──────────────┐ ┌───────────────┐     │
│  │solve_waveguide_│ │optimize_     │ │generate_      │     │
│  │mode            │ │waveguide     │ │mask           │     │
│  │                │ │              │ │               │     │
│  │ modesolverpy   │ │ SAX + JAX    │ │ gdsfactory    │     │
│  │ (Eigenmode EME)│ │ (Autodiff    │ │ (GDSII Layout)│     │
│  │                │ │  Gradient    │ │               │     │
│  │                │ │  Descent)    │ │               │     │
│  └────────────────┘ └──────────────┘ └───────────────┘     │
└─────────────────────────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│  Layer 3: Amazon Bedrock AgentCore (Infrastructure)         │
│  Runtime · Memory · Gateway · Policy · Identity · Obs.      │
└─────────────────────────────────────────────────────────────┘
```

**🔬 Layer 1 — Strands Agents SDK:** Code-level agent logic with the `@tool` decorator, multi-agent orchestration using the "Agents as Tools" pattern, and model-agnostic LLM access. Tools call the MCP server for physics computations.

**⚡ Layer 2 — FastMCP Physics Server:** Three deterministic physics tools — **modesolverpy** for eigenmode expansion, **SAX + JAX** for gradient-based inverse design, and **gdsfactory** for GDSII mask generation — served via a single FastMCP server.

**☁️ Layer 3 — Amazon Bedrock AgentCore:** Production infrastructure — serverless Runtime, persistent Memory, MCP-based Gateway, natural-language Policy guardrails, Cognito Identity, and CloudWatch Observability.

---

## 🔬 MCP Physics Tools — Deep Dive

### Tool 1: `solve_waveguide_mode` (modesolverpy)

When the prediction agent suggests a waveguide configuration (e.g., a 1.5µm wide, 400nm tall SiN core with SiO₂ cladding at 1550nm TE), the MCP server passes these exact parameters to modesolverpy. It runs a **fully vectorial eigenmode expansion (EME)** to calculate the effective refractive index, optical confinement factor, and mode field diameter. This allows the agent to mathematically validate its ML-driven propagation loss predictions against actual wave optics in milliseconds without consuming massive compute.

### Tool 2: `optimize_waveguide` (SAX + JAX)

For rigorous, physics grounded inverse design to minimize insertion loss or hit a specific coupling efficiency, SAX is a photonic circuit solver built natively on Google's JAX. Because it supports **automatic differentiation**, the MCP server uses it to run **gradient descent optimization** on waveguide parameters in real-time. Instead of just querying the 90K dataset for the closest match, the agent uses SAX to mathematically "slide" down the gradient to find the absolute optimal structural geometry for a user's constraints.

### Tool 3: `generate_mask` (gdsfactory)

Once the agent has optimized the waveguide, the engineer needs to physically build it. gdsfactory is the industry-standard Python library for photonic layout generation. The MCP server exposes a `generate_mask` tool that, when the user finalizes the design, uses gdsfactory to automatically render the **GDSII file** (handling the Bezier curves and routing) and returns the file path to the user's workspace.

---

## 🗄️ Hybrid Storage Architecture

For a high-precision photonics MCP server, a single database is insufficient. The data needs fall into two distinct categories — **bulk experimental data** (the 90K rows) and **fast stateful lookups** (simulation cache) — requiring a hybrid approach.

### Layer 1: Bulk Data — Amazon S3 Express One Zone (Parquet)

The `SiN-photonic-waveguide-loss-efficiency` dataset is stored in **Apache Parquet** format on **S3 Express One Zone**.

* **Why S3 Express One Zone:** Built for AI/ML workloads, offering 10x faster access and single-digit millisecond latency vs standard S3. This is crucial when the MCP server needs to stream Parquet chunks into a JAX-based inverse design loop.
* **Why Parquet:** Enables columnar reads — SAX can grab only waveguide widths and heights without loading the entire 90K rows of metadata. Dramatically reduces memory footprint and I/O.
* **Cost Efficiency:** While storage is slightly higher than S3 Standard, request costs are 50% lower — saving money when `photonic_agent.py` frequently queries the dataset for global search.

```python
# Example: Columnar read from S3 Express One Zone
import pyarrow.parquet as pq
table = pq.read_table(
    "s3://photonic-waveguide-data--usw2-az1--x-s3/dataset.parquet",
    columns=["width_um", "height_nm", "propagation_loss_dB_cm"]
)
```

### Layer 2: Simulation Cache — Amazon DynamoDB

Individual results from `modesolverpy` runs and `gdsfactory` layout paths are stored in **DynamoDB** for sub-10ms point lookups.

* **Why DynamoDB:** Before running an expensive eigenmode expansion, the MCP server checks DynamoDB to see if that exact waveguide geometry (`Width: 1.5µm, Height: 400nm, λ: 1550nm, TE`) has been simulated before. This prevents redundant physics calculations.
* **Schema:** The geometry hash is the partition key; S-parameters, effective index (n_eff), confinement factor, and MFD are stored as the value.
* **TTL:** Cache entries expire after 30 days to prevent stale results from dominating the cache.

```
┌─────────────────────────────────────────────────────────┐
│  DynamoDB Table: photonic-simulation-cache               │
│                                                          │
│  PK: geometry_hash (SHA-256 of sorted params)           │
│  SK: "MODE#1550nm#TE" | "OPTIM#prop_loss#0.2" | "GDS"  │
│                                                          │
│  Attributes:                                             │
│    n_eff, confinement_factor, mfd_um, group_index       │
│    optimized_width_um, optimized_height_nm               │
│    gds_file_path, created_at, ttl                        │
└─────────────────────────────────────────────────────────┘
```

### Layer 3: SQL Dataset Queries — Amazon S3 Tables

**S3 Tables** provides a managed table experience directly on top of the Parquet files in S3.

* **Benefit:** Run SQL-like filters (e.g., `SELECT * WHERE propagation_loss_dB_cm < 2.0 AND polarization = 'TE'`) directly against the HuggingFace dataset stored in S3, without loading it into memory first.
* **Impact:** Makes the MCP server's knowledge tools much lighter on RAM — the agent's analysis sub-agent can query 90K rows without ever loading the full DataFrame.

```python
# Example: S3 Tables query via PyArrow dataset
import pyarrow.dataset as ds
dataset = ds.dataset(
    "s3://photonic-waveguide-data--usw2-az1--x-s3/",
    format="parquet"
)
low_loss = dataset.to_table(
    filter=(ds.field("propagation_loss_dB_cm") < 2.0) &
           (ds.field("polarization") == "TE"),
    columns=["width_um", "height_nm", "propagation_loss_dB_cm"]
)
```

### Storage Flow

```
┌──────────────────────────────────────────────────────────────┐
│  MCP Server receives tool call                               │
│                                                              │
│  1. CHECK CACHE (DynamoDB)                                   │
│     └─ geometry_hash → hit? → return cached result           │
│                                                              │
│  2. CACHE MISS → RUN PHYSICS                                 │
│     ├─ modesolverpy / SAX / gdsfactory                      │
│     └─ WRITE result to DynamoDB cache                        │
│                                                              │
│  3. DATASET QUERIES (S3 Express One Zone)                    │
│     ├─ Parquet columnar reads for inverse design seeding     │
│     └─ S3 Tables SQL filters for analysis sub-agent          │
└──────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Layout

```
sin-photonic-mcp-agent/
├── README.md                       # Project documentation
├── pyproject.toml                  # Python project config
├── requirements.txt                # Python dependencies
│
├── mcp_server/
│   ├── __init__.py
│   ├── server.py                   # FastMCP server entry point
│   ├── config.py                   # Material library & server config
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── mode_solver.py          # modesolverpy eigenmode solver
│   │   ├── inverse_design.py       # SAX + JAX inverse design engine
│   │   └── mask_gen.py             # gdsfactory GDSII generation
│   ├── storage/
│   │   ├── __init__.py
│   │   ├── cache.py                # DynamoDB simulation cache
│   │   ├── s3_dataset.py           # S3 Express One Zone Parquet reader
│   │   └── s3_tables.py            # S3 Tables SQL query interface
│   └── schemas/
│       ├── __init__.py
│       └── waveguide.py            # Pydantic I/O models
│
├── data/
│   ├── download_dataset.py         # Fetch dataset from HuggingFace
│   └── README.md                   # Data dictionary and notes
│
├── src/
│   ├── __init__.py
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── photonic_agent.py       # Coordinator agent
│   │   ├── prediction_agent.py     # ML inference sub-agent
│   │   ├── optimization_agent.py   # Design optimization sub-agent
│   │   ├── analysis_agent.py       # Data analysis sub-agent
│   │   └── system_prompts.py       # All agent system prompts
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── data_tools.py           # Dataset loading and filtering
│   │   ├── prediction_tools.py     # XGBoost inference tools
│   │   ├── optimization_tools.py   # MCP client → optimize_waveguide
│   │   ├── physics_tools.py        # MCP client → solve_waveguide_mode
│   │   ├── mask_tools.py           # MCP client → generate_mask
│   │   └── visualization_tools.py  # Chart/plot generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── train.py                # Train regression models
│   │   ├── evaluate.py             # Model evaluation and metrics
│   │   └── serve.py                # Model serving for inference
│   ├── knowledge_base/
│   │   ├── __init__.py
│   │   ├── prepare_kb.py           # Prepare data for Bedrock KB
│   │   └── upload_s3.py            # Upload to S3
│   └── config/
│       ├── __init__.py
│       ├── aws_config.py           # AWS resource configuration
│       └── agent_config.py         # Agent parameters
│
├── infrastructure/
│   ├── cloudformation/
│   │   ├── agentcore-stack.yaml    # AgentCore resources
│   │   ├── bedrock-agent-stack.yaml# Bedrock Agent + KB
│   │   └── data-stack.yaml         # S3 + OpenSearch
│   │   └── storage-stack.yaml      # S3 Express + DynamoDB + S3 Tables
│   ├── cdk/
│   │   ├── app.py                  # CDK app entry point
│   │   └── stacks/
│   │       ├── agent_stack.py
│   │       ├── data_stack.py
│   │       └── observability_stack.py
│   └── docker/
│       ├── Dockerfile              # AgentCore Runtime container
│       └── docker-compose.yml      # Local development
│
├── tests/
│   ├── test_tools.py               # Unit tests for tools
│   ├── test_agents.py              # Agent integration tests
│   ├── test_predictions.py         # ML model accuracy tests
│   ├── test_mcp_server.py          # MCP server unit tests
│   ├── test_mcp_integration.py     # MCP integration tests
│   └── evals/
│       ├── eval_config.yaml        # AgentCore Evaluations config
│       └── eval_scenarios.json     # Test scenarios
│
├── notebooks/
│   ├── 01_data_exploration.ipynb   # EDA on the dataset
│   ├── 02_model_training.ipynb     # Train and evaluate models
│   └── 03_agent_demo.ipynb         # Interactive agent demo
│
└── scripts/
    ├── setup_aws.sh                # AWS resource provisioning
    ├── deploy_agentcore.sh         # Deploy to AgentCore Runtime
    └── run_evals.sh                # Run evaluation suite
```

---

## ▶️ Usage Examples

### 🔬 Mode Solving Query

```python
agent("What is the effective refractive index and confinement factor for a "
      "1.5µm wide, 400nm tall SiN waveguide with SiO2 cladding "
      "at 1550nm TE polarization?")
```

### 🔮 Prediction Query

```python
agent("What propagation loss should I expect for a 1.5µm wide, 400nm tall "
      "SiN waveguide with SiO2 cladding, deposited by LPCVD, etched with RIE, "
      "annealed at 900°C for 3 hours, operating at 1550nm TE polarization "
      "at room temperature?")
```

### ⚙️ Optimization Query

```python
agent("I need propagation loss below 0.3 dB/cm at 1550nm TE. What waveguide "
      "dimensions and fabrication process should I use? "
      "I'm constrained to LPCVD deposition.")
```

### 🏭 Mask Generation Query

```python
agent("Generate a GDSII mask for my optimized 1.2µm wide, 350nm tall waveguide. "
      "Use a 10mm straight section with 50µm bend radius, edge couplers, "
      "and Bezier routing.")
```

### 📊 Analysis Query

```python
agent("Compare the average propagation loss across the three deposition methods. "
      "Which gives the lowest loss for TE polarization at 1550nm?")
```

### 🧪 Batch QC Query

```python
agent("Show me the yield statistics for BATCH_12. "
      "How does it compare to BATCH_23?")
```

---

## 🚀 Deployment

### Start the MCP Physics Server

```bash
# Run the FastMCP server (must be running before the agent)
python -m mcp_server.server
# Server starts on http://0.0.0.0:8000/mcp
```

### Deploy to AgentCore Runtime

```bash
# Install AgentCore CLI
pip install bedrock-agentcore

# Test locally (requires Docker)
agentcore launch --local

# Deploy to AWS
agentcore launch
```

### Docker Compose (Local Development)

```bash
# Start both MCP server and agent
docker compose -f infrastructure/docker/docker-compose.yml up
```

### Environment Variables

```bash
export AWS_REGION=us-west-2
export AWS_PROFILE=photonic-agent-dev
export BEDROCK_MODEL_ID=anthropic.claude-sonnet-4-20250514
export DATASET_PATH=data/SiN_Photonic_Waveguide_Loss_Efficiency.csv
export MODEL_ARTIFACTS_PATH=models/
export MCP_SERVER_URL=http://localhost:8000/mcp

# Hybrid Storage Architecture
export S3_BUCKET_NAME=photonic-waveguide-data--usw2-az1--x-s3
export S3_DATASET_KEY=dataset.parquet
export DYNAMODB_TABLE_NAME=photonic-simulation-cache
export S3_TABLES_DATABASE=photonic_waveguide_db
export S3_TABLES_TABLE=waveguide_measurements
```

---

## 🙏 Contributing

Contributions of all kinds are welcome:

* **🔬 Physics Tools:** Improve solver accuracy or add new backends.
* **⚡ Optimization Algorithms:** Add new inverse design strategies.
* **🏭 Mask Generation:** Expand layout capabilities and PDK support.
* **🔮 Prediction Models:** Improve ML accuracy with new architectures.
* **📖 Documentation:** Improve developer experience.
* **🧪 Tests:** Increase test coverage.

---

## 👤 Author

**A Taylor**

[![Contact A Taylor](https://img.shields.io/badge/Contact-A%20Taylor-brightgreen.svg?logo=mail.ru&logoColor=white)](https://ataylor.getform.com/5w8wz)

---

## 📄 License

MIT © A Taylor 2026
