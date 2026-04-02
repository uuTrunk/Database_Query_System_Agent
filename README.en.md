# Database Query System - Agent (Generation & DB Node)

The Agent project serves as the **Data Generation Microservice** of the overall architecture. Its core responsibilities are exclusively constrained to compiling instructions with LLMs, querying backend SQL databases, and rendering parallel visual charts or Pandas structures. 

It has been entirely stripped of previously bloated ML prediction architectures, scaling back to a high-cohesion **Django** backend logic framework.

## Directory Structure

### Environment Setup (Conda)
For an expedited setup experience, we recommend using Conda to provision a dedicated python environment and installing the dependencies via `requirement.txt`:
```bash
# 1. Create a conda environment named "dqs" (Python 3.9)
conda create -n dqs python=3.9 -y

# 2. Activate the environment
conda activate dqs

# 3. Install the dependencies
pip install -r requirement.txt
```

- `agent_backend/`: Django project settings (`settings.py`, `wsgi`, etc.).
- `api/`: Django Rest Framework (DRF) application. This App exposes exclusively generation workflows or verification logics (does not contain any predictive load-balancing features).
- `ask_ai/`, `data_access/`, `llm_access/`, `pgv/`: Inner cores to dictate LLM prompt mechanics, database pools and context-enhancement mechanisms.

## Running the Application

For a more streamlined execution aesthetic, the standard `manage.py` entrypoint has been renamed strictly to `main.py`.

You must harness your current pre-configured Conda environment `dqs`. To spin up:
```bash
conda activate dqs
python main.py runserver 8000
```
The node binds seamlessly to `http://127.0.0.1:8000`, awaiting intensive visual generation or AI bridging payloads from the decoupled frontend service.
