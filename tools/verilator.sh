#!/bin/sh
# Run Verilator: the one on PATH, or the binary bundled in the `verilator` pip package.
if command -v verilator >/dev/null 2>&1 && [ -z "$MGPU_FORCE_PIP_VERILATOR" ]; then
    exec verilator "$@"
fi
root=$(python3 -c "import os, verilator; print(os.path.dirname(verilator.__file__))" 2>/dev/null)
if [ -z "$root" ] || [ ! -x "$root/bin/verilator" ]; then
    echo "verilator not found: install it (pip install verilator, or your package manager)" >&2
    exit 127
fi
export VERILATOR_ROOT="$root"
exec "$root/bin/verilator" "$@"
