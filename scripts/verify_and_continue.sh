#!/bin/bash
set -e
source /etc/network_turbo
source <PROJECT_ROOT>/miniconda3/etc/profile.d/conda.sh
conda activate base
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

echo '=== Step 1: Verify shard 1 integrity ==='
python3 -c 
