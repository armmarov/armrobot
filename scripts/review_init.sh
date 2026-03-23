#!/usr/bin/env bash
# review_init.sh — One-time briefing session for Codex and Qwen reviewers
#
# Feeds them ALL reference material including EngineAI source so they can do
# a proper gap study. Saves session IDs to .review_sessions so run_review.sh
# can resume without re-reading everything.
#
# Usage: ./scripts/review_init.sh
# Output: .review_sessions (session ID file)

set -uo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DOCS_DIR="$REPO_DIR/docs"
ENV_DIR="$REPO_DIR/source/ArmRobotLegging/ArmRobotLegging/tasks/direct/armrobotlegging"
AGENTS_DIR="$ENV_DIR/agents"
SESSIONS_FILE="$REPO_DIR/.review_sessions"

ENGINEAI_LEGGED="$HOME/work/robot/engineai/engineai_legged_gym"
ENGINEAI_RL="$HOME/work/robot/engineai/engineai_rl_workspace"

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

echo "=== Building briefing prompt (this may be large) ==="

# ── Build full briefing prompt ────────────────────────────────────────────────
BRIEFING="$(cat <<BRIEF_EOF
You are a technical reviewer for a bipedal walking RL project. Your job across multiple
review sessions is to act as a peer reviewer — checking that planned training changes are
correct, well-motivated, and consistent with both the live code and the EngineAI reference.

Read ALL files below carefully. After reading, confirm you have understood:
1. What EngineAI does in their reference implementation (reward formulas, PPO modifications, training config)
2. What our current IsaacLab implementation does (reward functions, obs space, PPO config)
3. What the known gaps are (from ENGINEAI_VS_ISAACLAB.md)
4. What the current training plan is (Run 47/48/49/50 from TRAINING_PLAN.md)
5. What past runs failed and why (from TRAINING_HISTORY.md)

This is your briefing — you will be asked specific review questions in follow-up messages.
For now just read everything and confirm your understanding with a brief summary of each gap
between EngineAI and our current code.

--- OUR PROJECT FILES ---

$(embed_file "CLAUDE.md (standing orders + key lessons)" "$REPO_DIR/CLAUDE.md")
$(embed_file "docs/TRAINING_HISTORY.md" "$DOCS_DIR/TRAINING_HISTORY.md")
$(embed_file "docs/TRAINING_PLAN.md" "$DOCS_DIR/TRAINING_PLAN.md")
$(embed_file "docs/ENGINEAI_VS_ISAACLAB.md" "$DOCS_DIR/ENGINEAI_VS_ISAACLAB.md")
$(embed_file "armrobotlegging_env_cfg.py" "$ENV_DIR/armrobotlegging_env_cfg.py")
$(embed_file "agents/rsl_rl_ppo_cfg.py" "$AGENTS_DIR/rsl_rl_ppo_cfg.py")
$(embed_file "armrobotlegging_env.py (full)" "$ENV_DIR/armrobotlegging_env.py")

--- ENGINEAI REFERENCE CODE (compare against our code above) ---

$(embed_file "EngineAI ZqSA01 env rewards (zqsa01.py)" "$ENGINEAI_LEGGED/legged_gym/envs/zqsa01/zqsa01.py")
$(embed_file "EngineAI RSL-RL PPO with symmetry loss (ppo.py)" "$ENGINEAI_LEGGED/rsl_rl/rsl_rl/algorithms/ppo.py")
$(embed_file "EngineAI RSL-RL on_policy_runner.py" "$ENGINEAI_LEGGED/rsl_rl/rsl_rl/runners/on_policy_runner.py")

--- END OF BRIEFING ---

Now summarize the gaps you found between EngineAI and our current implementation.
Be specific: cite file names and line numbers where the implementations differ.
BRIEF_EOF
)"

# Write to temp file
prompt_file=$(mktemp)
printf '%s' "$BRIEFING" > "$prompt_file"
echo "Prompt size: $(wc -c < "$prompt_file") bytes"
echo ""

# ── Run both init sessions ────────────────────────────────────────────────────
codex_out=$(mktemp)
qwen_out=$(mktemp)

echo "Starting Codex briefing session..."
(
  cd "$REPO_DIR"
  codex exec \
    -c 'sandbox_permissions=["disk-full-read-access","network-full-access"]' \
    - < "$prompt_file" > "$codex_out" 2>&1
) &
PID_CODEX=$!

echo "Starting Qwen briefing session (chat-recording enabled)..."
(
  cd "$REPO_DIR"
  # --chat-recording saves history to disk; --continue in run_review.sh will resume it
  qwen --approval-mode yolo \
    --chat-recording \
    < "$prompt_file" > "$qwen_out" 2>&1
) &
PID_QWEN=$!

wait $PID_CODEX || true
wait $PID_QWEN || true

# ── Extract Codex session ID ──────────────────────────────────────────────────
CODEX_SESSION_ID=$(grep "^session id:" "$codex_out" | head -1 | awk '{print $3}')
if [ -z "$CODEX_SESSION_ID" ]; then
  echo "WARNING: Could not extract Codex session ID. Will use full-embed fallback in reviews."
  CODEX_SESSION_ID="none"
fi

# ── Extract Qwen session ID (newest .jsonl in project chats dir) ──────────────
QWEN_CHATS_DIR="$HOME/.qwen/projects/-home-armmarov-work-robot-isaac-workspace-armrobot-ArmRobotLegging/chats"
QWEN_SESSION_ID=$(ls -t "$QWEN_CHATS_DIR"/*.jsonl 2>/dev/null | head -1 | xargs basename 2>/dev/null | sed 's/\.jsonl//')
if [ -z "$QWEN_SESSION_ID" ]; then
  echo "WARNING: Could not find Qwen session file."
  QWEN_SESSION_ID="none"
fi

# ── Save session IDs ──────────────────────────────────────────────────────────
cat > "$SESSIONS_FILE" <<EOF
CODEX_SESSION_ID=$CODEX_SESSION_ID
QWEN_SESSION_ID=$QWEN_SESSION_ID
INIT_DATE=$(date +%Y-%m-%d)
EOF

echo ""
echo "=== Briefing complete ==="
echo "  Codex session: $CODEX_SESSION_ID"
echo "  Qwen session:  $QWEN_SESSION_ID"
echo "  Saved to: $SESSIONS_FILE"
echo ""
echo "--- Codex gap summary ---"
tail -40 "$codex_out"
echo ""
echo "--- Qwen gap summary ---"
tail -40 "$qwen_out"

# Save full briefing responses to docs
TIMESTAMP="$(date '+%Y-%m-%d %H:%M')"
{
  echo "# Codex Briefing — $TIMESTAMP"
  echo ""
  echo "---"
  echo ""
  cat "$codex_out"
} > "$DOCS_DIR/REVIEW_CODEX.md"

{
  echo "# Qwen Briefing — $TIMESTAMP"
  echo ""
  echo "---"
  echo ""
  cat "$qwen_out"
} > "$DOCS_DIR/REVIEW_QWEN.md"

rm -f "$codex_out" "$qwen_out" "$prompt_file"
