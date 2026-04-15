# CA6125: LLM and RAG

## Installation

> Install **ninja** to speed up the installation of **flash-attention**.

```bash
sudo apt-get install ninja-build
```

> Install **uv** for easier Python package and environment management.
```
pip install uv
```

Create and activate the virtual environment:
```
conda create -n LLMCourse python=3.12 -y
conda activate LLMCourse
```
Install the required packages:
```
uv pip install -U vllm --torch-backend=auto
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install openai
```

## Serving LLMs Locally with vLLM
Before running ```./Lecture1/offline_serving.py```, start the vLLM server:
```
vllm serve Qwen/Qwen3.5-2B --port 8000 --tensor-parallel-size 1 --max-model-len 262144
```

