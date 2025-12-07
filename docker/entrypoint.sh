#!/bin/bash
set -e

# Ensure directories exist
mkdir -p /data /output /crawled_data

# Log environment info
echo "[Entrypoint] DATA_INPUT_DIR: $DATA_INPUT_DIR (test input files)"
echo "[Entrypoint] DATA_OUTPUT_DIR: $DATA_OUTPUT_DIR (output files)"
echo "[Entrypoint] KB_DATA_DIR: $KB_DATA_DIR (knowledge base data)"
echo "[Entrypoint] Python path: $(which python)"
echo "[Entrypoint] Starting application..."

# Execute the command
exec "$@"