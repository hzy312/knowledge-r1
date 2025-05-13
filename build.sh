conda create -n ikea python=3.9
conda activate ikea
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 vllm==0.6.3

pip install -e .

pip3 install flash-attn --no-build-isolation
pip install wandb



conda create -n retriever python=3.10
conda activate retriever

pip install torch==2.4.0 transformers datasets

conda install -c pytorch -c nvidia faiss-gpu=1.8.0


pip install uvicorn fastapi