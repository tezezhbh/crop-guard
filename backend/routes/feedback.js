/**
 * routes/feedback.js
 *
 * POST /api/feedback  — save feedback (requires auth)
 * GET  /api/feedback  — list user's own feedback
 *
 * SECURITY FIXES:
 *   - note field length-capped at 1000 chars
 *   - disease field length-capped at 200 chars
 *   - scan_id verified to belong to requesting user (prevents IDOR)
 */
const express  = require("express");
const { pool } = require("../db");
const { requireAuth } = require("../middleware/auth");
const router   = express.Router();

router.post("/", requireAuth, async (req, res) => {
  const { scan_id, disease, rating, note, treatment_result } = req.body;
  if (!disease || !rating) return res.status(400).json({ error: "disease and rating required." });
  if (!["correct","wrong","unsure"].includes(rating)) return res.status(400).json({ error: "Invalid rating." });

  // Input length validation
  if (disease.length > 200) return res.status(400).json({ error: "Disease field too long (max 200 chars)." });
  if (note && note.length > 1000) return res.status(400).json({ error: "Note too long (max 1000 chars)." });
  if (treatment_result && treatment_result.length > 200) {
    return res.status(400).json({ error: "treatment_result too long (max 200 chars)." });
  }

  // SECURITY FIX: verify scan_id belongs to the requesting user (IDOR prevention)
  let resolvedScanId = null;
  if (scan_id) {
    const { rows: scanRows } = await pool.query(
      "SELECT id FROM predictions WHERE id=$1 AND user_id=$2",
      [scan_id, req.user.id]
    );
    if (!scanRows.length) {
      return res.status(403).json({ error: "Scan not found or access denied." });
    }
    resolvedScanId = scan_id;
  }

  try {
    const { rows } = await pool.query(
      `INSERT INTO feedback (user_id,scan_id,disease,rating,note,treatment_result)
       VALUES ($1,$2,$3,$4,$5,$6) RETURNING id,created_at`,
      [req.user.id, resolvedScanId, disease, rating, note||null, treatment_result||null]
    );
    res.status(201).json(rows[0]);
  } catch (err) {
    console.error("[feedback]", err.message);
    res.status(500).json({ error: "Could not save feedback." });
  }
});

router.get("/", requireAuth, async (req, res) => {
  try {
    const { rows } = await pool.query(
      "SELECT * FROM feedback WHERE user_id=$1 ORDER BY created_at DESC LIMIT 200",
      [req.user.id]
    );
    res.json(rows);
  } catch (err) {
    res.status(500).json({ error: "Could not load feedback." });
  }
});

module.exports = router;
