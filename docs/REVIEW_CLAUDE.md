# Claude Review — Run 47 Sign-off

**Date:** 2026-03-23
**Run:** 47
**Changes:** Orientation dual-signal formula + Hip splay default_joint_pos fix
**Status:** WAITING FOR USER SIGN-OFF — training is NOT running

---

## What was changed

### Change 1 — Orientation formula (`armrobotlegging_env.py:751–761`)

**Before:**
```python
roll_pitch_error = torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)
rew_orient = cfg.rew_orientation * torch.exp(-roll_pitch_error * 10.0)
```

**After:**
```python
roll, pitch, _ = euler_xyz_from_quat(base_quat)  # tuple unpack — NOT [:, :2]
quat_mismatch = torch.exp(-(torch.abs(roll) + torch.abs(pitch)) * 10.0)
orientation = torch.exp(-torch.norm(projected_gravity[:, :2], dim=1) * 20.0)
rew_orient = cfg.rew_orientation * (quat_mismatch + orientation) / 2.0
```

**Why:** Single gravity signal at scale 10 was too weak — robot has persistent forward trunk lean across all Run 46 screenshots. EngineAI uses two complementary signals (euler + gravity) at stronger scale.

---

### Change 2 — Hip splay indices (`armrobotlegging_env.py:802–814`)

**Before:**
```python
yaw_roll_dev = torch.sum(torch.abs(joint_pos_rel[:, [0, 1, 6, 7]]), dim=-1)
# [0,1,6,7] = hip_pitch_l, hip_roll_l, hip_pitch_r, hip_roll_r
rew_default_pos = cfg.rew_default_joint_pos * (
    torch.exp(-yaw_roll_dev * 100.0) - 0.01 * torch.norm(joint_pos_rel, dim=1)
)
```

**After:**
```python
left_splay  = joint_pos_rel[:, [1, 2]]   # hip_roll_l, hip_yaw_l
right_splay = joint_pos_rel[:, [7, 8]]   # hip_roll_r, hip_yaw_r
splay_norm  = torch.norm(left_splay, dim=1) + torch.norm(right_splay, dim=1)
splay_norm  = torch.clamp(splay_norm - 0.1, min=0.0, max=0.5)  # 0.1 rad deadband
rew_default_pos = cfg.rew_default_joint_pos * (
    torch.exp(-splay_norm * 100.0) - 0.01 * torch.norm(joint_pos_rel, dim=1)
)
```

**Why:** Old indices included hip_pitch (0,6) which needs freedom for gait, and missed hip_yaw (2,8) which is the primary splay joint. Run 46 screenshots show wide lateral leg spread — directly caused by missing hip_yaw penalty.

---

## Review synthesis

### Codex findings
1. **BUG FOUND:** `euler_xyz_from_quat` returns `tuple[Tensor, Tensor, Tensor]` not `[N,3]`. Original code `base_euler[:, :2]` would crash on first reward computation. **Fixed** by unpacking `roll, pitch, _ = euler_xyz_from_quat(base_quat)`.
2. Splay indices `[1,2,7,8]` — confirmed correct.
3. Norm + deadband formula — confirmed correct (pairwise L/R norms, single deadband).
4. `-0.01 * torch.norm(joint_pos_rel)` global term — confirmed must be kept.
5. Noted Codex could not read the actual file (not embedded in session message) — reviewed from description only.

### Qwen findings
1. Change 1 (orientation): ✅ CORRECT — all signals, scales, and averaging match EngineAI exactly.
2. Change 2 (splay): ✅ CORRECT — indices, norm formula, deadband all verified against EngineAI `zqsa01.py`.
3. No bugs found from Qwen's side (Qwen did have the actual code embedded).

### Agreement / conflict
- Both agree on correctness of the approach and target indices.
- Codex caught the tuple bug that Qwen missed (Qwen had the embedded code but didn't flag it; Codex reasoned about the API without seeing code and still caught it).
- Both confirmed `rew_orientation=1.0` is correct — no weight change needed.

---

## Bug fixes applied
- [x] `euler_xyz_from_quat` tuple unpacking fixed before commit

## GO / NO-GO recommendation

**GO** — both changes are correct, bug is fixed, implementation matches EngineAI reference exactly.

**Watch in first 500 iters:**
- `orientation` reward term will likely drop initially (stricter formula) — this is expected, not a regression
- Episode length should stay near ~999 or climb (not drop significantly)
- Hip splay visible in screenshots should narrow
- No value_loss spikes above ~50

---

## Awaiting user sign-off

**Please confirm:** ✅ GO — run training / ❌ NO-GO — hold
