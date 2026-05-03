/**
 * server.js — CropGuard AI Backend v3
 * Mekelle Institute of Technology 2026
 *
 * Architecture:
 *   React (3000)  →  Node/Express (3001)  →  Python FastAPI (8000)
 *                                          →  PostgreSQL (users + predictions)
 */
require("dotenv").config();
const axios = require("axios");
const express = require("express");
const cors = require("cors");
const helmet = require("helmet");
const morgan = require("morgan");
const path = require("path");

let compression;
try {
  compression = require("compression");
} catch {
  compression = null;
}

const rateLimit = require("express-rate-limit");
const authRouter = require("./routes/auth");
const chatRouter = require("./routes/chat");
const predictRouter = require("./routes/predict");
const historyRouter = require("./routes/history");
const feedbackRouter = require("./routes/feedback");
const { initDB, checkDB } = require("./db");

const app = express();
const PORT = process.env.PORT || 3001;
const ENV = process.env.NODE_ENV;

// ── Security
app.use(
  helmet({
    crossOriginResourcePolicy: { policy: "cross-origin" },
    contentSecurityPolicy: false,
  }),
);

// ── CORS
// NOTE: Wildcard "*" cannot be used with credentials:true (CORS spec).
const ALLOWED_ORIGINS = new Set([
  process.env.FRONTEND_URL || "http://localhost:3000",
  "http://localhost:3000",
  "http://localhost:5173",
  "http://127.0.0.1:3000",
  "http://127.0.0.1:5173",
]);
app.use(
  cors({
    origin: (origin, cb) => {
      if (!origin) return cb(null, true);
      if (ALLOWED_ORIGINS.has(origin) || ENV === "development")
        return cb(null, true);
      cb(new Error("CORS: origin " + origin + " not allowed"));
    },
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
    credentials: true,
  }),
);

// Middleware
if (compression) app.use(compression());
app.use(morgan(ENV === "production" ? "combined" : "dev"));
app.use(express.json({ limit: "1mb" }));
app.use(express.urlencoded({ extended: true }));

// Static: uploaded images
app.use(
  "/uploads",
  express.static(path.join(__dirname, "uploads"), {
    maxAge: "7d",
    etag: true,
  }),
);

// Rate limiting
const predictLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 30,
  message: { error: "Too many scan requests. Please wait a moment." },
  standardHeaders: true,
  legacyHeaders: false,
});
const authLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 20,
  message: {
    error: "Too many authentication attempts. Please wait 15 minutes.",
  },
  standardHeaders: true,
  legacyHeaders: false,
});
const generalLimiter = rateLimit({
  windowMs: 60 * 1000,
  max: 200,
  message: { error: "Too many requests." },
});
app.use(generalLimiter);

// Routes
app.use("/api/auth", authLimiter, authRouter);
app.use(
  "/api/chat",
  rateLimit({
    windowMs: 60 * 1000,
    max: 30,
    message: { error: "Too many chat requests." },
    standardHeaders: true,
    legacyHeaders: false,
  }),
  chatRouter,
);
app.use("/api/predict", predictLimiter, predictRouter);
app.use("/api/history", historyRouter);
app.use("/api/feedback", feedbackRouter);

// Health
app.get("/api/health", async (_req, res) => {
  try {
    const db = await checkDB();
    res.json({
      status: "ok",
      env: ENV,
      version: "3.0.0",
      timestamp: new Date().toISOString(),
      database: db,
    });
  } catch {
    res
      .status(503)
      .json({ status: "error", message: "Database not reachable" });
  }
});

// AI server health proxy
app.get("/api/ai-health", async (_req, res) => {
  const AI = process.env.AI_SERVER_URL || "http://localhost:8000";
  try {
    const { data } = await axios.get(`${AI}/health`, { timeout: 5000 });
    res.json({ status: "ok", ai: data });
  } catch {
    res
      .status(503)
      .json({ status: "offline", message: `AI server not reachable at ${AI}` });
  }
});

// 404
app.use((_req, res) => res.status(404).json({ error: "Endpoint not found." }));

// Error handler
app.use((err, _req, res, _next) => {
  console.error("[server] Error:", err.message);
  if (err.code === "LIMIT_FILE_SIZE")
    return res.status(400).json({ error: "File too large. Max 10 MB." });
  if (err.message?.includes("Only"))
    return res.status(400).json({ error: err.message });
  res.status(500).json({ error: "Internal server error." });
});

// Start

// Graceful shutdown
process.on("SIGTERM", () => {
  console.log("[server] SIGTERM received — shutting down gracefully");
  process.exit(0);
});
process.on("SIGINT", () => {
  console.log("[server] SIGINT received — shutting down gracefully");
  process.exit(0);
});

async function start() {
  // Production safety guards
  if (ENV === "production") {
    if (!process.env.JWT_SECRET || process.env.JWT_SECRET.length < 32) {
      console.error(
        "[startup] FATAL: JWT_SECRET must be set and >= 32 chars in production",
      );
      process.exit(1);
    }
    if (
      !process.env.DATABASE_URL ||
      process.env.DATABASE_URL.includes("changeme")
    ) {
      console.error(
        "[startup] FATAL: DATABASE_URL must be configured in production",
      );
      process.exit(1);
    }
    if (!process.env.GROQ_API_KEY) {
      console.warn(
        "[startup] ⚠ GROQ_API_KEY not set — AI chat will be unavailable",
      );
    }
  }

  try {
    await initDB();
    app.listen(PORT, () => {
      console.log("");
      console.log("  ╔══════════════════════════════════════════╗");
      console.log("  ║      CropGuard AI — Backend v3           ║");
      console.log("  ╠══════════════════════════════════════════╣");
      console.log(`  ║  API    →  http://localhost:${PORT}          ║`);
      console.log(`  ║  Auth   →  /api/auth/register|login      ║`);
      console.log(
        `  ║  AI     →  ${(process.env.AI_SERVER_URL || "http://localhost:8000").padEnd(34)}║`,
      );
      console.log("  ╚══════════════════════════════════════════╝");
      console.log("");
    });
  } catch (err) {
    console.error("[startup] Fatal:", err.message);
    console.error(
      "→ Check DATABASE_URL in .env and ensure PostgreSQL is running",
    );
    process.exit(1);
  }
}
start();
