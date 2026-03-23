#!/usr/bin/env bash
# run_review.sh — Run Codex + Qwen reviews in parallel
#
# If .review_sessions exists (from review_init.sh), resumes those sessions so
# reviewers retain their full gap-study context. Otherwise falls back to
# embedding key files directly in the prompt.
#
# Usage: ./scripts/run_review.sh "your review question"
#        ./scripts/run_review.sh              (uses default question)
#
# Output: docs/REVIEW_CODEX.md and docs/REVIEW_QWEN.md

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCS_DIR="$REPO_DIR/docs"
ENV_DIR="$REPO_DIR/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging"
AGENTS_DIR="$ENV_DIR/agents"
SESSIONS_FILE="$REPO_DIR/.review_sessions"

REVIEW_QUESTION="${1:-Review the updated docs/TRAINING_PLAN.md. Focus on: (1) is the reframing of Run 47 Change 1 (from ref_joint_pos to default_joint_pos tuning) correct and well-motivated? (2) is the Run 48 thin PPO subclass approach sound? (3) is the Run 49 staged breakdown the right order? (4) are there any remaining gaps or inconsistencies between the plan and the live env code?}"

# ── Load sessions if available ────────────────────────────────────────────────
USE_SESSIONS=false
CODEX_SESSION_ID=""
QWEN_SESSION_ID=""

if [ -f "$SESSIONS_FILE" ]; then
  source "$SESSIONS_FILE"
  # Check if init was done within 24h (sessions valid ~24h for Codex)
  INIT_EPOCH=$(date -d "$INIT_DATE" +%s 2>/dev/null || echo 0)
  NOW_EPOCH=$(date +%s)
  AGE_HOURS=$(( (NOW_EPOCH - INIT_EPOCH) / 3600 ))
  if [ "${CODEX_SESSION_ID:-none}" != "none" ] && [ "${QWEN_SESSION_ID:-none}" != "none" ] && [ "$AGE_HOURS" -lt 24 ]; then
    USE_SESSIONS=true
    echo "=== Resuming review sessions (init was ${AGE_HOURS}h ago) ==="
  else
    echo "=== Sessions stale (${AGE_HOURS}h old) — using full-embed fallback ==="
    echo "    Run 'make review-init' to refresh sessions."
  fi
else
  echo "=== No sessions found — using full-embed fallback ==="
  echo "    Run 'make review-init' first for better gap-study context."
fi

echo "Question: $REVIEW_QUESTION"
echo ""

# ── Helper ────────────────────────────────────────────────────────────────────
embed_file() {
  local label="$1"
  local path="$2"
  echo "=== FILE: $label ==="
  if [ -f "$path" ]; then
    cat "$path"
  else
    echo "(file not found: $path)"
  fi
  echo "=== END: $label ==="
  echo ""
}

# ── Build prompt ──────────────────────────────────────────────────────────────
prompt_file=$(mktemp)

if [ "$USE_SESSIONS" = true ]; then
  # Sessions exist: just send the question + latest file diffs
  # Include only files that may have changed since briefing
  cat > "$prompt_file" <<PROMPT_EOF
The following files may have been updated since your briefing — review the latest versions:

$(embed_file "docs/TRAINING_PLAN.md (latest)" "$DOCS_DIR/TRAINING_PLAN.md")
$(embed_file "docs/ENGINEAI_VS_ISAACLAB.md (latest)" "$DOCS_DIR/ENGINEAI_VS_ISAACLAB.md")
$(embed_file "armrobotlegging_env_cfg.py (latest)" "$ENV_DIR/armrobotlegging_env_cfg.py")

Now answer this review question:

$REVIEW_QUESTION
PROMPT_EOF

else
  # No sessions: embed full context as fallback
  cat > "$prompt_file" <<PROMPT_EOF
You are a technical reviewer for a bipedal walking RL project (PM01 robot, IsaacLab + RSL-RL PPO).

Key context:
- Currently on Run 46 (15-frame compact history, 334-dim obs, ~6000+ iters done)
- force_balance reward is permanently banned — causes stepping-in-place exploit at any weight
- ref_joint_pos at env.py line ~711 already uses exp(-2*norm) - 0.2*clamp form (EngineAI-style)
- Main remaining gaps vs EngineAI: orientation formula (single signal vs dual), no symmetry loss, no domain randomization
- Run 47 goal: fix orientation formula + add hip-splay regularization via default_joint_pos
- Run 48 goal: add symmetry loss via thin PPO subclass (NOT full RSL-RL copy)
- Run 49 goal: staged domain randomization (obs noise → latency → friction → mass/COM/strength)
- Run 50 goal: command velocity curriculum

$(embed_file "CLAUDE.md" "$REPO_DIR/CLAUDE.md")
$(embed_file "docs/TRAINING_PLAN.md" "$DOCS_DIR/TRAINING_PLAN.md")
$(embed_file "docs/ENGINEAI_VS_ISAACLAB.md" "$DOCS_DIR/ENGINEAI_VS_ISAACLAB.md")
$(embed_file "armrobotlegging_env_cfg.py" "$ENV_DIR/armrobotlegging_env_cfg.py")
$(embed_file "agents/rsl_rl_ppo_cfg.py" "$AGENTS_DIR/rsl_rl_ppo_cfg.py")
$(embed_file "armrobotlegging_env.py (reward grep)" <(grep -n "def rew_\|ref_joint_pos\|default_joint_pos\|rew_orientation\|rew_tracking\|rew_default" "$ENV_DIR/armrobotlegging_env.py" | head -120))

NOTE: For a full gap study including EngineAI reference code, run 'make review-init' first.

Now answer this review question:

$REVIEW_QUESTION
PROMPT_EOF
fi

echo "Prompt size: $(wc -c < "$prompt_file") bytes"

# ── Run both reviews ──────────────────────────────────────────────────────────
codex_out=$(mktemp)
qwen_out=$(mktemp)

if [ "$USE_SESSIONS" = true ]; then
  (
    cd "$REPO_DIR"
    codex exec \
      -c 'sandbox_permissions=["disk-full-read-access","network-full-access"]' \
      resume "$CODEX_SESSION_ID" \
      - < "$prompt_file" > "$codex_out" 2>&1
  ) &
  PID_CODEX=$!

  (
    cd "$REPO_DIR"
    qwen --approval-mode yolo \
      --chat-recording \
      --resume "$QWEN_SESSION_ID" \
      < "$prompt_file" > "$qwen_out" 2>&1
  ) &
  PID_QWEN=$!
else
  (
    cd "$REPO_DIR"
    codex exec \
      -c 'sandbox_permissions=["disk-full-read-access","network-full-access"]' \
      - < "$prompt_file" > "$codex_out" 2>&1
  ) &
  PID_CODEX=$!

  (
    cd "$REPO_DIR"
    qwen --approval-mode yolo \
      < "$prompt_file" > "$qwen_out" 2>&1
  ) &
  PID_QWEN=$!
fi

wait $PID_CODEX || true; STATUS_CODEX=$?
wait $PID_QWEN  || true; STATUS_QWEN=$?

TIMESTAMP="$(date '+%Y-%m-%d %H:%M')"

{
  echo "# Codex Review — $TIMESTAMP"
  echo ""
  echo "> **Question:** $REVIEW_QUESTION"
  if [ "$USE_SESSIONS" = true ]; then
    echo "> **Session:** \`$CODEX_SESSION_ID\` (resumed — has full gap-study context)"
  else
    echo "> **Session:** none (full-embed fallback — run \`make review-init\` for persistent sessions)"
  fi
  echo ""
  echo "---"
  echo ""
  cat "$codex_out"
} > "$DOCS_DIR/REVIEW_CODEX.md"

{
  echo "# Qwen Review — $TIMESTAMP"
  echo ""
  echo "> **Question:** $REVIEW_QUESTION"
  if [ "$USE_SESSIONS" = true ]; then
    echo "> **Session:** \`$QWEN_SESSION_ID\` (resumed — has full gap-study context)"
  else
    echo "> **Session:** none (full-embed fallback — run \`make review-init\` for persistent sessions)"
  fi
  echo ""
  echo "---"
  echo ""
  cat "$qwen_out"
} > "$DOCS_DIR/REVIEW_QWEN.md"

rm -f "$codex_out" "$qwen_out" "$prompt_file"

echo ""
echo "=== Done ==="
echo "  Codex: $DOCS_DIR/REVIEW_CODEX.md (exit $STATUS_CODEX)"
echo "  Qwen:  $DOCS_DIR/REVIEW_QWEN.md  (exit $STATUS_QWEN)"
