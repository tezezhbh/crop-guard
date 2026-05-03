#!/bin/bash
# start-dev.sh — Start all CropGuard AI services in development mode
# Usage: ./start-dev.sh

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

echo ""
echo "  ╔══════════════════════════════════════════╗"
echo "  ║      CropGuard AI — Dev Startup          ║"
echo "  ╚══════════════════════════════════════════╝"
echo ""

# Check dependencies
command -v node  >/dev/null 2>&1 || { echo -e "${RED}❌ node not found${NC}"; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo -e "${RED}❌ python3 not found${NC}"; exit 1; }
command -v psql >/dev/null 2>&1 || echo -e "${YELLOW}⚠  psql not found — ensure PostgreSQL is running${NC}"

# Check env files
[ -f backend/.env ] || { echo -e "${RED}❌ backend/.env missing — copy backend/.env.example${NC}"; exit 1; }
[ -f frontend/.env ] || cp frontend/.env.example frontend/.env

echo -e "${GREEN}▶ Starting ML server (port 8000)...${NC}"
cd ml && uvicorn serve:app --host 127.0.0.1 --port 8000 &
ML_PID=$!
cd ..

sleep 2

echo -e "${GREEN}▶ Starting Backend (port 3001)...${NC}"
cd backend && npm start &
BACKEND_PID=$!
cd ..

sleep 2

echo -e "${GREEN}▶ Starting Frontend (port 3000)...${NC}"
cd frontend && npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}✅ All services started!${NC}"
echo "  Frontend  →  http://localhost:3000"
echo "  Backend   →  http://localhost:3001"
echo "  ML Server →  http://127.0.0.1:8000"
echo ""
echo "  Press Ctrl+C to stop all services"

trap "kill $ML_PID $BACKEND_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
