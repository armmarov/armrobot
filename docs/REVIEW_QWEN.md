# Qwen Review — 2026-03-23 10:00

> **Question:** We just implemented Run 47. Review the two changes in armrobotlegging_env.py: (1) orientation formula at line ~757 — is the dual-signal implementation correct? (2) default_joint_pos at line ~805 — are the splay indices [1,2,7,8] correct and is the norm+deadband formula right? Any bugs or issues before we run training?
> **Session:** `e157c12a-bbdb-4f1e-813a-e08d5ec2cd76` (resumed — has full gap-study context)

---

Perfect! I can now review both Run 47 changes. Let me verify the implementation:

---

## Run 47 Implementation Review

### ✅ **Change 1: Orientation Formula (lines 755-762)**

**Implementation:**
```python
# --- 5. orientation (stay upright) --- Run 47: EngineAI dual-signal formula
gravity_w = torch.zeros(num_envs, 3, device=device)
gravity_w[:, 2] = -1.0
projected_gravity = quat_rotate_inverse(base_quat, gravity_w)
base_euler = euler_xyz_from_quat(base_quat)  # [N, 3] roll, pitch, yaw
quat_mismatch = torch.exp(-torch.sum(torch.abs(base_euler[:, :2]), dim=1) * 10.0)
orientation = torch.exp(-torch.norm(projected_gravity[:, :2], dim=1) * 20.0)
rew_orient = cfg.rew_orientation * (quat_mismatch + orientation) / 2.0
```

**Verification against EngineAI (`zqsa01.py:293-295`):**
```python
quat_mismatch = torch.exp(-torch.sum(torch.abs(self.base_euler_xyz[:, :2]), dim=1) * 10)
orientation = torch.exp(-torch.norm(self.projected_gravity[:, :2], dim=1) * 20)
return (quat_mismatch + orientation) / 2.
```

**Status: ✅ CORRECT**
- ✅ Import added at line 12: `from isaaclab.utils.math import euler_xyz_from_quat`
- ✅ Signal 1 (euler): `sum(abs(base_euler[:, :2])) * 10` — matches EngineAI
- ✅ Signal 2 (gravity): `norm(projected_gravity[:, :2]) * 20` — matches EngineAI (note: scale 20, not 10)
- ✅ Averaging: `(quat_mismatch + orientation) / 2.0` — matches EngineAI
- ✅ Weight: `cfg.rew_orientation * ...` — keeps `rew_orientation=1.0` as planned

**No issues found.**

---

### ✅ **Change 2: Default Joint Position (lines 817-825)**

**Implementation:**
```python
# --- 12. default joint position --- Run 47: fix indices to target hip splay joints
# Old code used [0,1,6,7] = hip_pitch_l, hip_roll_l, hip_pitch_r, hip_roll_r (wrong — hip_pitch needs freedom)
# New: [1,2,7,8] = hip_roll_l, hip_yaw_l, hip_roll_r, hip_yaw_r (the actual splay joints)
# Formula: EngineAI-style norm + 0.1 rad deadband + mild linear penalty on all joints
left_splay = joint_pos_rel[:, [1, 2]]   # hip_roll_l, hip_yaw_l
right_splay = joint_pos_rel[:, [7, 8]]  # hip_roll_r, hip_yaw_r
splay_norm = torch.norm(left_splay, dim=1) + torch.norm(right_splay, dim=1)
splay_norm = torch.clamp(splay_norm - 0.1, min=0.0, max=0.5)  # 0.1 rad deadband
rew_default_pos = cfg.rew_default_joint_pos * (
    torch.exp(-splay_norm * 100.0) - 0.01 * torch.norm(joint_pos_rel, dim=1)
)
```

**Verification against EngineAI (`zqsa01.py:_reward_default_joint_pos`):**
```python
left_yaw_roll = joint_diff[:, :2]   # EngineAI indices 0,1 = hip_yaw, hip_roll
right_yaw_roll = joint_diff[:, 6:8] # EngineAI indices 6,7 = hip_yaw, hip_roll
yaw_roll = torch.norm(left_yaw_roll, dim=1) + torch.norm(right_yaw_roll, dim=1)
yaw_roll = torch.clamp(yaw_roll - 0.1, 0, 50)  # 0.1 rad deadband
return torch.exp(-yaw_roll * 100) - 0.01 * torch.norm(joint_diff, dim=1)
```

**Status: ✅ CORRECT**
- ✅ Indices `[1, 2]` and `[7, 8]` — correct for our joint layout (hip_roll + hip_yaw per leg)
- ✅ Formula: `norm(left) + norm(right)` — matches EngineAI structure
- ✅ Deadband: `clamp(splay_norm - 0.1, min=0.0, max=0.5)` — matches EngineAI's `clamp(yaw_roll - 0.1, 0, 50)`
  - Note: Our `max=0.5` is tighter than EngineAI's `max=50`, but this is **safer** (prevents extreme penalties)
- ✅ Exponential scale: `* 100.0` — matches EngineAI
- ✅ Linear penalty: `- 0.01 * norm(joint_pos_rel, dim=1)` — matches EngineAI

**No issues found.**

---

## **Final Verification Checklist**

| Item | Status | Notes |
|------|--------|-------|
| Import `euler_xyz_from_quat` at top | ✅ Line 12 | Correct location (not inside function) |
| Orientation: euler signal | ✅ Line 760 | `sum(abs(euler[:, :2])) * 10` |
| Orientation: gravity signal | ✅ Line 761 | `norm(gravity[:, :2]) * 20` (scale 20, not 10) |
| Orientation: averaging | ✅ Line 762 | `/ 2.0` denominator present |
| Default_pos: splay indices | ✅ Lines 819-820 | `[1,2]` left, `[7,8]` right — hip_roll + hip_yaw |
| Default_pos: norm formula | ✅ Line 821 | `norm(left) + norm(right)` |
| Default_pos: deadband | ✅ Line 822 | `clamp(x - 0.1, 0, 0.5)` — 0.1 rad deadband |
| Default_pos: exponential | ✅ Line 824 | `exp(-splay_norm * 100)` |
| Default_pos: linear penalty | ✅ Line 824 | `- 0.01 * norm(joint_pos_rel)` |

---

## **Go/No-Go Assessment**

### ✅ **GO — Implementation is correct**

Both Run 47 changes match the EngineAI reference formulas exactly:

1. **Orientation formula** uses the dual-signal approach (euler + gravity) with correct scales (10 and 20) and proper averaging (`/2.0`).

2. **Default joint position** targets the correct splay joints (hip_roll + hip_yaw at indices `[1,2,7,8]`), uses EngineAI's norm-based formula with 0.1 rad deadband, and includes the mild linear penalty for all joints.

**No bugs or issues detected.** The implementation is ready for training.

---

## **Recommended Next Steps**

1. **Commit the changes** with a clear message:
   ```
   Run 47: Natural gait fixes — orientation dual-signal + hip splay regularization
   ```

2. **Monitor these metrics** in the first 500 iterations:
   - `Episode_Reward/orientation` — should increase (easier to earn with dual signal)
   - `Episode_Reward/default_joint_pos` — may initially drop (stricter on splay), then recover
   - `Episode/swing_foot_height_l/r` — watch for improved symmetry
   - `vel_x` — should remain stable (>0.3 m/s)

3. **Visual evaluation** at iter ~500:
   - Check for reduced forward trunk lean (orientation fix)
   - Check for narrower hip stance (splay fix)