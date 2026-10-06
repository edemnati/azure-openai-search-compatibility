# Contributing

## Development setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

Run the local quality checks:

```powershell
python -m pytest -m "not environment and not evaluation"
python -m ruff check .
python -m mypy src scripts
python -m build
```

## Pull requests

- Keep changes focused and include tests for behavior changes.
- Do not commit credentials, customer data, endpoints, resource IDs, generated live-test
  output, or evaluation results.
- Document compatibility changes and intentional behavior differences from Azure OpenAI
  On Your Data.
- Treat environment and evaluation tests as opt-in because they access Azure resources and
  can incur charges.
