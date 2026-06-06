# pino_research

A Python research project boilerplate configured with:
- **[uv](https://github.com/astral-sh/uv)** for fast, reliable package management.
- **[Hydra](https://hydra.cc/)** for composing hierarchical configurations.
- **[Pydantic](https://docs.pydantic.dev/)** for robust config data validation.
- **[Pyrefly](https://pyrefly.org/)** for fast, incremental static type checking.

## Installation

This project uses `uv`. To install the dependencies:

```bash
uv sync
```

## Running the Code

The main entry point is `src/main.py`.

```bash
uv run python src/main.py
```

You can override Hydra configurations from the command line:

```bash
uv run python src/main.py hparams.learning_rate=0.005
```

## Type Checking

To run Pyrefly type checking:

```bash
uv run pyrefly check
```

## Configuration Structure

Configurations are stored in `conf/`.
- `conf/config.yaml`: The master config file.
- `conf/paths/`: Directory paths.
- `conf/hparams/`: Hyperparameters.

The configurations are strictly validated in `src/main.py` using Pydantic models defined in `src/config_schema.py`.
