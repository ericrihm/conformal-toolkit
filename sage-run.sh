#!/usr/bin/env bash
# Run SageMath commands for conformal_toolkit development.
# Detects whether to use micromamba (native ARM64) or Docker (x86 emulation).
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Option 1: Native micromamba sage environment (fastest on Apple Silicon)
if command -v micromamba &>/dev/null || [ -x "$HOME/.local/bin/micromamba" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    if micromamba env list 2>/dev/null | grep -q "sage"; then
        eval "$(micromamba shell hook -s bash)"
        micromamba activate sage
        if [ $# -eq 0 ]; then
            sage --python3
        elif [ "$1" = "test" ]; then
            sage -python -m pytest tests/ -v "${@:2}"
        elif [ "$1" = "pytest" ]; then
            sage -python -m pytest "${@:2}"
        else
            sage "$@"
        fi
        exit 0
    fi
fi

# Option 2: Docker with Rosetta emulation
if command -v docker &>/dev/null; then
    echo "Using Docker (x86 emulation via Rosetta)..."
    if [ $# -eq 0 ]; then
        docker compose run --rm sage-shell
    elif [ "$1" = "test" ]; then
        docker compose run --rm sage
    elif [ "$1" = "pytest" ]; then
        docker compose run --rm sage sage -python -m pytest "${@:2}"
    else
        docker compose run --rm sage-shell sage "$@"
    fi
    exit 0
fi

echo "Error: Neither micromamba (sage env) nor Docker found."
echo "Install micromamba: curl -Ls https://micro.mamba.pm/api/micromamba/osx-arm64/latest | tar -xvj -C ~/.local/bin --strip-components=1 bin/micromamba"
echo "Then: micromamba create -n sage -c conda-forge sage python=3.11 -y"
exit 1
