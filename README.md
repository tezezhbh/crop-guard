# CropGuard AI

AI-powered plant disease detection platform for farmers. Built at Mekelle Institute of Technology 2026.

## Architecture

```
React (3000)  →  Node/Express (3001)  →  Python FastAPI ML (8000)
                                      →  PostgreSQL (5432)
```

## Quick Start (Development)

### Prerequisites
- Node.js >= 18
- Python 3.11
- PostgreSQL 15

### 1. Backend
```bash
cd backend
cp .env.example .env   # Fill in DATABASE_URL, JWT_SECRET, GROQ_API_KEY
npm install
npm start
```

### 2. ML Server
```bash
cd ml
pip install -r requirements.txt
uvicorn serve:app --host 127.0.0.1 --port 8000
```

### 3. Frontend
```bash
cd frontend
cp .env.example .env   # Set VITE_API_URL if needed
npm install
npm run dev
```

Visit http://localhost:3000

---

## Production Deployment (Docker Compose)

```bash
cp .env.example .env
# Fill in: DB_PASSWORD, JWT_SECRET, GROQ_API_KEY, FRONTEND_URL, VITE_API_URL

docker compose up -d
```

---

## Environment Variables

### Backend (`backend/.env`)
| Variable | Required | Description |
|---|---|---|
| `DATABASE_URL` | ✅ | PostgreSQL connection string |
| `JWT_SECRET` | ✅ | 64-char random string for signing JWTs |
| `GROQ_API_KEY` | ✅ | Free key from console.groq.com |
| `AI_SERVER_URL` | ✅ | ML server URL (default: http://localhost:8000) |
| `FRONTEND_URL` | ✅ | Frontend origin for CORS whitelist |
| `PORT` | optional | API port (default: 3001) |

### Frontend (`frontend/.env`)
| Variable | Required | Description |
|---|---|---|
| `VITE_API_URL` | optional | Backend URL (default: http://localhost:3001) |

### ML (`ml/.env`)
| Variable | Required | Description |
|---|---|---|
| `MODEL_PATH` | optional | Path to SavedModel directory |
| `MIN_CONFIDENCE` | optional | Minimum confidence % (default: 60) |

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | /api/auth/register | — | Create account |
| POST | /api/auth/login | — | Sign in |
| POST | /api/auth/refresh | — | Refresh access token |
| POST | /api/auth/logout | — | Sign out |
| GET | /api/auth/me | ✅ | Get profile |
| PUT | /api/auth/me | ✅ | Update profile |
| POST | /api/predict | ✅ | Scan plant image |
| GET | /api/history | ✅ | Scan history |
| GET | /api/history/stats | ✅ | Dashboard stats |
| POST | /api/chat | — | AI chat assistant |
| POST | /api/feedback | ✅ | Submit scan feedback |
| GET | /api/health | — | Health check |
| GET | /api/ai-health | — | ML server health |

---

## Security Notes

- Never commit `.env` files — they are in `.gitignore`
- JWT_SECRET must be a strong random string (64+ chars)
- ML server is never exposed publicly — only reachable by backend
- All file uploads are validated (type, size, blur, resolution)
- SQL injection protected via parameterized queries
- Rate limiting on all endpoints
