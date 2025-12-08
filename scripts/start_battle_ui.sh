#!/bin/bash
# Start both backend and frontend for the battle UI

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "==================================="
echo " Pokemon Battle vs ReBeL AI"
echo "==================================="
echo ""

# Check if ports are available
check_port() {
    if lsof -i :$1 > /dev/null 2>&1; then
        echo "Warning: Port $1 is already in use"
        return 1
    fi
    return 0
}

# Start backend
echo "Starting backend API server on port 8001..."
cd "$PROJECT_DIR"
uv run python -m src.battle_api.main &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

# Wait for backend to start
sleep 3

# Check if backend started
if ! kill -0 $BACKEND_PID 2>/dev/null; then
    echo "Error: Backend failed to start"
    exit 1
fi

# Start frontend
echo ""
echo "Starting frontend on port 3000..."
cd "$PROJECT_DIR/frontend"
pnpm run dev &
FRONTEND_PID=$!
echo "Frontend PID: $FRONTEND_PID"

echo ""
echo "==================================="
echo " Servers started!"
echo "==================================="
echo ""
echo " Frontend: http://localhost:3000"
echo " Backend:  http://localhost:8001"
echo " API Docs: http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop both servers"
echo ""

# Cleanup on exit
cleanup() {
    echo ""
    echo "Stopping servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    echo "Done."
}

trap cleanup EXIT

# Wait for either process to exit
wait
