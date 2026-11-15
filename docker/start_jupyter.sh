#!/usr/bin/env bash
set -euo pipefail
cd /code
exec jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
