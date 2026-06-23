#!/usr/bin/env bash
# Deploy VectorBridge Transatlantic Benchmark to Azure Container Instances
# US East (Virginia) + Germany West Central (Frankfurt)
#
# Prerequisites:
#   az login
#   az account set --subscription YOUR_SUBSCRIPTION_ID
#
# Usage:
#   chmod +x deploy_azure.sh
#   ./deploy_azure.sh
#   # Then run: ./run_benchmark_remote.sh

set -e

RG="vb-benchmark"
LOCATION_US="eastus"
LOCATION_DE="germanywestcentral"
CONTAINER_TARGET="vb-target"
CONTAINER_SOURCE="vb-source"
LICENSE_KEY="${LICENSE_KEY:-vb_test_benchmark}"

echo "=== VectorBridge Azure Benchmark Deployment ==="
echo ""

# Create resource group
echo "1. Creating resource group $RG ..."
az group create --name "$RG" --location "$LOCATION_US" --output table

# ── Deploy Germany (target) first — we need its IP ──────────────────────────

echo ""
echo "2. Deploying Qdrant target in Germany West Central ..."
az container create \
  --resource-group "$RG" \
  --name "$CONTAINER_TARGET" \
  --location "$LOCATION_DE" \
  --image qdrant/qdrant:latest \
  --cpu 2 --memory 4 \
  --ports 6333 6334 \
  --ip-address Public \
  --environment-variables \
    QDRANT__SERVICE__HTTP_PORT=6333 \
    QDRANT__SERVICE__GRPC_PORT=6334 \
  --output table

echo ""
echo "3. Getting Germany container IP ..."
TARGET_IP=$(az container show \
  --resource-group "$RG" \
  --name "$CONTAINER_TARGET" \
  --query ipAddress.ip -o tsv)

echo "   Germany Qdrant IP: $TARGET_IP"
echo "   Waiting 30s for Qdrant to be ready ..."
sleep 30

# Verify Qdrant is up
curl -s "http://$TARGET_IP:6333/healthz" || {
  echo "   Qdrant not ready yet, waiting 30 more seconds ..."
  sleep 30
}
echo "   Qdrant is up!"

# ── Deploy US source + benchmark runner ─────────────────────────────────────

echo ""
echo "4. Deploying ChromaDB + benchmark runner in US East ..."
echo "   Target host: $TARGET_IP"

# We run benchmark from local machine (not from ACI) for better log visibility
# But ChromaDB runs on ACI
az container create \
  --resource-group "$RG" \
  --name "$CONTAINER_SOURCE" \
  --location "$LOCATION_US" \
  --image chromadb/chroma:latest \
  --cpu 2 --memory 4 \
  --ports 8000 \
  --ip-address Public \
  --environment-variables ANONYMIZED_TELEMETRY=False \
  --output table

echo ""
echo "5. Getting US container IP ..."
SOURCE_IP=$(az container show \
  --resource-group "$RG" \
  --name "$CONTAINER_SOURCE" \
  --query ipAddress.ip -o tsv)

echo "   US ChromaDB IP: $SOURCE_IP"

echo ""
echo "=== Deployment Complete ==="
echo ""
echo "Container IPs:"
echo "  US ChromaDB (source):  $SOURCE_IP:8000"
echo "  DE Qdrant   (target):  $TARGET_IP:6333"
echo ""
echo "Next steps (run locally):"
echo ""
echo "  # 1. Install dependencies"
echo "  pip install numpy chromadb qdrant-client"
echo ""
echo "  # 2. Generate + load 100K vectors into ChromaDB"
echo "  python generate_vectors.py"
echo "  python load_source.py --host $SOURCE_IP --port 8000"
echo ""
echo "  # 3. Run the benchmark"
echo "  python run_benchmark.py \\"
echo "    --source-host $SOURCE_IP --source-port 8000 \\"
echo "    --target-host $TARGET_IP --target-port 6333 \\"
echo "    --license-key $LICENSE_KEY"
echo ""
echo "  # 4. View results"
echo "  cat results/benchmark_*.json | python -m json.tool"
echo ""

# Save IPs for convenience
cat > .env.benchmark <<EOF
SOURCE_IP=$SOURCE_IP
TARGET_IP=$TARGET_IP
LICENSE_KEY=$LICENSE_KEY
EOF
echo "IPs saved to .env.benchmark"

# ── Cleanup helper ──────────────────────────────────────────────────────────
echo ""
echo "To destroy everything when done:"
echo "  az group delete --name $RG --yes --no-wait"
