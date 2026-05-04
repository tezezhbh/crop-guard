/**
 * middleware/auth.js
 *
 * SECURITY FIX: JWT_SECRET fallback removed. App fails hard if secret is missing.
 */
const jwt = require("jsonwebtoken");

const SECRET = process.env.JWT_SECRET;
if (!SECRET) {
  console.error("[auth] FATAL: JWT_SECRET environment variable is required. Set it in .env and restart.");
  process.exit(1);
}

function requireAuth(req, res, next) {
  const token = extractToken(req);
  if (!token) {
    return res.status(401).json({ error: "Authentication required. Please sign in." });
  }
  try {
    req.user = jwt.verify(token, SECRET);
    next();
  } catch (err) {
    if (err.name === "TokenExpiredError") {
      return res.status(401).json({ error: "Session expired. Please sign in again.", expired: true });
    }
    return res.status(401).json({ error: "Invalid token. Please sign in again." });
  }
}

function optionalAuth(req, res, next) {
  const token = extractToken(req);
  if (token) {
    try { req.user = jwt.verify(token, SECRET); } catch {}
  }
  next();
}

function extractToken(req) {
  const header = req.headers.authorization || "";
  if (header.startsWith("Bearer ")) return header.slice(7);
  return null;
}

function signAccessToken(user) {
  return jwt.sign(
    { id: user.id, email: user.email, name: user.name, plan: user.plan },
    SECRET,
    { expiresIn: "24h" }
  );
}

function signRefreshToken(userId) {
  return jwt.sign(
    { sub: userId, type: "refresh" },
    SECRET,
    { expiresIn: "30d" }
  );
}

module.exports = { requireAuth, optionalAuth, signAccessToken, signRefreshToken };
