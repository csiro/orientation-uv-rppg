uv pip install -e .[dev]
uv pip install -U pip build twine
uv build          # creates dist/*.whl and dist/*.tar.gz
twine check dist/*       # metadata sanity