#!/usr/bin/env bash
set -euo pipefail
cd /code
exec python -m lifemate.predict
