#!/bin/bash
# VERTEX — convenience run scripts
# Make executable: chmod +x run.sh
# Usage: ./run.sh [api|ui|test-sec|test-github|test-all|full]

set -e
cd "$(dirname "$0")"

# Load .env if it exists
if [ -f .env ]; then
  export $(grep -v '^#' .env | xargs)
fi

case "$1" in

  api)
    echo "Starting VERTEX FastAPI server on :${FASTAPI_PORT:-8000}..."
    uvicorn vertex.api.main:app \
      --host 0.0.0.0 \
      --port "${FASTAPI_PORT:-8000}" \
      --reload \
      --log-level info
    ;;

  ui)
    echo "Starting VERTEX Streamlit UI..."
    streamlit run vertex/ui/app.py \
      --server.port 8501 \
      --server.address 0.0.0.0
    ;;

  full)
    echo "Starting full VERTEX stack (API + UI)..."
    # Start API in background
    uvicorn vertex.api.main:app \
      --host 0.0.0.0 \
      --port "${FASTAPI_PORT:-8000}" \
      --log-level warning &
    API_PID=$!
    echo "API started (PID $API_PID)"
    sleep 2
    # Start UI in foreground
    streamlit run vertex/ui/app.py --server.port 8501
    # Cleanup on exit
    kill $API_PID 2>/dev/null
    ;;

  test-sec)
    python vertex/tests/test_agents.py sec
    ;;

  test-github)
    python vertex/tests/test_agents.py github
    ;;

  test-market)
    python vertex/tests/test_agents.py market
    ;;

  test-news)
    python vertex/tests/test_agents.py news
    ;;

  test-memory)
    python vertex/tests/test_agents.py memory
    ;;

  test-rag)
    python vertex/tests/test_agents.py rag
    ;;

  test-all)
    python vertex/tests/test_agents.py all
    ;;

  test-full)
    python vertex/tests/test_agents.py orchestrator
    ;;

  *)
    echo "VERTEX run script"
    echo ""
    echo "Usage: ./run.sh <command>"
    echo ""
    echo "  api          Start FastAPI server (port 8000)"
    echo "  ui           Start Streamlit dashboard (port 8501)"
    echo "  full         Start both API + UI"
    echo ""
    echo "  test-sec     Test SEC EDGAR agent (no key needed)"
    echo "  test-github  Test GitHub signal agent (no key needed)"
    echo "  test-market  Test market agent (needs ALPHA_VANTAGE_API_KEY)"
    echo "  test-news    Test news agent (Yahoo RSS fallback works)"
    echo "  test-memory  Test graph memory store"
    echo "  test-rag     Test ChromaDB vector store"
    echo "  test-all     Run all individual agent tests"
    echo "  test-full    Run full orchestrator test (needs OPENAI_API_KEY)"
    echo ""
    echo "Prerequisites:"
    echo "  pip install -r requirements.txt"
    echo "  cp .env.example .env  # then fill in your API keys"
    ;;
esac
