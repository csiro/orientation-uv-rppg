rm -rf orientation-uv-rppg
uv venv orientation-uv-rppg --python 3.8.10
source .venv/bin/activate # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt