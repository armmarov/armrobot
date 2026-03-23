# Qwen Review — 2026-03-23 12:05

> **Question:** Review the reward functions in armrobotlegging_env.py. Compare against the EngineAI zqsa01.py reference you read during briefing. List every formula difference, missing term, or incorrect scale factor.
> **Session:** `e157c12a-bbdb-4f1e-813a-e08d5ec2cd76` (resumed — has full gap-study context)

---

Now let me compare against the EngineAI `zqsa01.py` reference. I'll create a comprehensive comparison table:

---

## Reward Function Comparison: Our Implementation vs EngineAI zqsa01.py

### **Summary Table**

| # | Reward Term | Our Formula | EngineAI Formula | Status |
|---|-------------|-------------|------------------|--------|
| 1 | tracking_lin_vel | ✅ Match | `exp(-sum_sq_error / sigma)` | ✅ CORRECT |
| 2 | tracking_ang_vel | ✅ Match | `exp(-sq_error / sigma)` | ✅ CORRECT |
| 3 | ref_joint_pos | ✅ Match | `exp(-2*norm) - 0.2*clamp(norm,0,0.5)` | ✅ CORRECT |
| 4 | feet_air_time | ✅ Match (biped) | `clamp(air_time, 0, 0.5) * first_contact` | ✅ CORRECT |
| 5 | feet_contact_number | ⚠️ Diff | `where(contact == stance_mask, +1.0, -0.3)` | ⚠️ FORMULA DIFFERS |
| 6 | orientation | ✅ Match (Run 47) | `(exp(-sum_abs_euler*10) + exp(-norm_grav*20)) / 2` | ✅ CORRECT |
| 7 | base_height | ✅ Match | `exp(-abs(height_error) * 100)` | ✅ CORRECT |
| 8 | vel_mismatch | ⚠️ Diff | `(exp(-sq_z_lin*10) + exp(-norm_xy_ang*5)) / 2` | ⚠️ FORMULA DIFFERS |
| 9 | action_smoothness | ✅ Match | `term1 + term2 + 0.05*abs(actions)` | ✅ CORRECT |
| 10 | energy | ❌ Diff | `sum(sq(torques))` | ❌ MISSING TORQUES |
| 11 | alive | ❌ Extra | NOT in EngineAI | ❌ SHOULD REMOVE |
| 12 | feet_clearance | ⚠️ Diff | `norm((swing_curve*target - heights) * swing_mask)` | ⚠️ FORMULA DIFFERS |
| 13 | default_joint_pos | ✅ Match | `exp(-norm_yaw_roll*100) - 0.01*norm(all_joints)` | ✅ CORRECT |
| 14 | feet_distance | ✅ Match | `(exp(-abs(d_min)*100) + exp(-abs(d_max)*100)) / 2` | ✅ CORRECT |
| 15 | foot_slip | ✅ Match | `sqrt(foot_speed) * contact` | ✅ CORRECT |
| 16 | track_vel_hard | ✅ Match | `(exp(-lin_err*10) + exp(-ang_err*10)) / 2 - 0.2*penalty` | ✅ CORRECT |
| 17 | low_speed | ✅ Match | Discrete: -1/+2/-2 based on speed ratio | ✅ CORRECT |
| 18 | dof_vel | ✅ Match | `sum(sq(joint_vel))` | ✅ CORRECT |
| 19 | dof_acc | ✅ Match | `sum(sq((last_vel - curr_vel) / dt))` | ✅ CORRECT |
| 20 | collision | ❌ Missing | `sum(norm(contact_forces) > 0.1)` | ❌ MISSING |
| 21 | torques | ❌ Missing | `sum(sq(torques))` | ❌ MISSING |
| 22 | feet_contact_forces | ❌ Missing | `sum(norm(contact_forces) - max_force).clip(0, 350)` | ❌ MISSING |
| 23 | termination | ❌ Extra | NOT in EngineAI (uses 0.0) | ❌ SHOULD REMOVE |
| 24 | swing_phase_ground | ❌ Extra | NOT in EngineAI | ❌ SHOULD REMOVE |
| 25 | feet_height_max | ❌ Extra | NOT in EngineAI | ❌ SHOULD REMOVE |
| 26 | lat_vel | ❌ Extra | NOT in EngineAI | ❌ SHOULD REMOVE |
| 27 | base_acc | ✅ Match | `exp(-norm(base_acc) * 3)` | ✅ CORRECT |
| 28 | knee_distance | ✅ Match | `exp(-abs(knee_dist - target) * 20)` | ✅ CORRECT |
| 29 | force_balance | ❌ Extra | NOT in EngineAI | ❌ SHOULD REMOVE |

---

## **Detailed Analysis**

### **✅ CORRECT Implementations (15 terms)**

These match EngineAI formulas exactly:

1. **tracking_lin_vel** (line 706-707)
   - Our: `exp(-sum_sq_error / sigma)`
   - EngineAI: Same
   - ✅ Scale: 1.4 ✓

2. **tracking_ang_vel** (line 709-710)
   - Our: `exp(-sq_error / sigma)`
   - EngineAI: Same
   - ✅ Scale: 1.1 ✓

3. **ref_joint_pos** (line 717-719)
   - Our: `exp(-2*norm) - 0.2*clamp(norm,0,0.5)`
   - EngineAI: Same (zqsa01.py:_reward_joint_pos)
   - ✅ Scale: 1.47 ✓

4. **feet_air_time** (line 726-728)
   - Our: `clamp(air_time, 0, 0.5) * first_contact` (biped formula)
   - EngineAI biped: Same (rewards_biped.py)
   - ✅ Scale: 1.0 ✓

5. **orientation** (line 743-749) — Run 47 fix
   - Our: `(exp(-sum_abs_euler*10) + exp(-norm_grav*20)) / 2`
   - EngineAI: Same (zqsa01.py:_reward_orientation)
   - ✅ Scale: 1.0 ✓

6. **base_height** (line 752-753)
   - Our: `exp(-abs(height_error) * 100)`
   - EngineAI: Same
   - ✅ Scale: 0.2 ✓

7. **action_smoothness** (line 760-766)
   - Our: `term1 + term2 + 0.05*abs(actions)` with scale -0.003
   - EngineAI: Same (zqsa01.py:_reward_action_smoothness)
   - ✅ Scale: -0.003 ✓

8. **default_joint_pos** (line 785-792) — Run 47 fix
   - Our: `exp(-norm_yaw_roll*100) - 0.01*norm(all_joints)` with indices [1,2,7,8]
   - EngineAI: Same formula (zqsa01.py:_reward_default_joint_pos)
   - ✅ Scale: 0.53 ✓

9. **feet_distance** (line 795-799)
   - Our: `0.5 * (exp(-abs(d_min)*100) + exp(-abs(d_max)*100))`
   - EngineAI: Same (zqsa01.py:_reward_feet_distance)
   - ✅ Scale: 0.13 ✓

10. **foot_slip** (line 802-804)
    - Our: `sqrt(foot_speed) * contact`
    - EngineAI: Same
    - ✅ Scale: -0.067 ✓

11. **track_vel_hard** (line 807-813)
    - Our: `(exp(-lin_err*10) + exp(-ang_err*10)) / 2 - 0.2*penalty`
    - EngineAI: Same
    - ✅ Scale: 0.5 ✓

12. **low_speed** (line 816-826)
    - Our: Discrete -1.0/+2.0/-2.0 based on speed ratio
    - EngineAI: Same (zqsa01.py:_reward_low_speed)
    - ✅ Scale: 0.2 ✓

13. **dof_vel** (line 829-830)
    - Our: `sum(sq(joint_vel))`
    - EngineAI: Same
    - ✅ Scale: -6.7e-6 ✓

14. **dof_acc** (line 833-834)
    - Our: `sum(sq((last_vel - curr_vel) / dt))`
    - EngineAI: Same
    - ✅ Scale: -3.3e-9 ✓

15. **base_acc** (line 843-846)
    - Our: `exp(-norm(base_acc) * 3)`
    - EngineAI: Same (zqsa01.py:_reward_base_acc)
    - ✅ Scale: 0.2 ✓

16. **knee_distance** (line 849-852)
    - Our: `exp(-abs(knee_dist - target) * 20)`
    - EngineAI: Same (zqsa01.py:_reward_knee_distance)
    - ✅ Scale: 0.2 ✓

---

### **⚠️ FORMULA DIFFERS (3 terms)**

These have structural formula differences:

#### **1. feet_contact_number** (line 738-741)

**Our implementation:**
```python
contact_reward = torch.where(foot_contact == desired_contact, 1.0, -0.3)
rew_contact_pattern = cfg.rew_feet_contact_number * torch.mean(contact_reward, dim=-1)
```

**EngineAI (zqsa01.py:_reward_feet_contact_number):**
```python
reward = torch.where(contact == stance_mask, 1.0, -0.3)
return torch.mean(reward, dim=1)
```

**Difference:** ✅ **MATCHES** — I initially flagged this but upon re-reading, our formula IS correct. Both use `where(match, +1.0, -0.3)` and average over both feet.

**Status:** ✅ CORRECT (false alarm)

---

#### **2. vel_mismatch** (line 756-759)

**Our implementation:**
```python
rew_vel_mismatch = cfg.rew_vel_mismatch * 0.5 * (
    torch.exp(-torch.square(lin_vel_b[:, 2]) * 10.0)
    + torch.exp(-torch.sum(torch.square(ang_vel_b[:, :2]), dim=-1) * 5.0)
)
```

**EngineAI (zqsa01.py:_reward_vel_mismatch_exp):**
```python
lin_mismatch = torch.exp(-torch.square(self.base_lin_vel[:, 2]) * 10)
ang_mismatch = torch.exp(-torch.norm(self.base_ang_vel[:, :2], dim=1) * 5.)
return (lin_mismatch + ang_mismatch) / 2.
```

**Difference:**
- Our: `sum(square(ang_vel_xy))` — sum of squares
- EngineAI: `norm(ang_vel_xy)` — L2 norm (sqrt of sum of squares)

**Impact:** Minor. For small errors, `norm(x) ≈ sum_sq(x)` but for larger errors, norm gives smaller penalty.

**Recommendation:** Change to `torch.norm(ang_vel_b[:, :2], dim=1)` to match exactly.

---

#### **3. feet_clearance** (line 772-779)

**Our implementation:**
```python
swing_curve[:, 0] = torch.clamp(-sin_phase, min=0.0)  # left: -sin when sin < 0
swing_curve[:, 1] = torch.clamp(sin_phase, min=0.0)   # right: sin when sin > 0
target_h = swing_curve * cfg.target_feet_height
clearance_error = (target_h - feet_heights) * swing_mask
rew_clearance = cfg.rew_feet_clearance * torch.norm(clearance_error, dim=1)
```

**EngineAI (zqsa01.py:_reward_feet_clearance):**
```python
feet_z = self.rigid_state[:, self.feet_indices, 2] - 0.05
delta_z = feet_z - self.last_feet_z
self.feet_height += delta_z
self.feet_height *= ~contact

swing_mask = 1 - self._get_gait_phase()
rew_pos = (self.feet_height > self.cfg.rewards.target_feet_height)
rew_pos = torch.sum(rew_pos * swing_mask, dim=1)
return rew_pos  # scale 1.6
```

**Difference:**
- **Our:** Penalizes deviation from swing-curve-scaled target (continuous penalty based on norm of error)
- **EngineAI:** Binary reward — counts how many feet exceeded target height at peak (discrete count)

**Impact:** EngineAI's formula is simpler and gives a clearer signal: "did the foot clear the target?" Ours gives gradient throughout the swing but may be less direct.

**Recommendation:** Consider switching to EngineAI's binary formula for clearer stepping signal.

---

### **❌ MISSING IN ENGINEAI (6 extra terms we added)**

These are our custom additions not in EngineAI:

#### **1. alive** (line 769-770)
```python
rew_alive = cfg.rew_alive * torch.ones(num_envs, device=device)
```
- **EngineAI:** NOT PRESENT (uses 0.0)
- **Our scale:** 0.0
- **Status:** ✅ Already disabled — no action needed

#### **2. termination** (line 869-870)
```python
rew_term = cfg.rew_termination * reset_terminated.float()
```
- **EngineAI:** Uses 0.0 (zqsa01.py:_reward_termination returns `self.reset_buf * ~self.time_out_buf` with scale 0.0)
- **Our scale:** -0.0
- **Status:** ✅ Already disabled — no action needed

#### **3. swing_phase_ground** (line 731-737)
```python
rew_swing_ground = swing_scale * torch.sum(swing_ground_violation, dim=-1)
```
- **EngineAI:** NOT PRESENT
- **Our scale:** 0.0
- **Status:** ✅ Already disabled — no action needed

#### **4. feet_height_max** (line 782-783)
```python
rew_feet_height_max = cfg.rew_feet_height_max * torch.sum(over_height, dim=-1)
```
- **EngineAI:** NOT PRESENT
- **Our scale:** -0.6
- **Status:** ⚠️ Our addition to prevent over-lifting (Run 19). May conflict with EngineAI's simpler clearance formula.

#### **5. lat_vel** (line 837-839)
```python
lat_vel_error = torch.square(commands[:, 1] - lin_vel_b[:, 1])
rew_lat_vel = cfg.rew_lat_vel * torch.exp(-lat_vel_error * 10.0)
```
- **EngineAI:** NOT PRESENT
- **Our scale:** 0.04
- **Status:** ⚠️ Our addition for anti-drift. May not be needed with symmetry loss.

#### **6. force_balance** (line 855-862)
```python
force_ratio = torch.min(force_l, force_r) / (force_l + force_r + 1e-6)
rew_force_balance = cfg.rew_force_balance * torch.exp(-torch.square(force_ratio - 0.5) * 50.0)
```
- **EngineAI:** NOT PRESENT
- **Our scale:** 0.0
- **Status:** ✅ Already disabled (Run 42 caused stepping-in-place) — no action needed

---

### **❌ MISSING REWARD TERMS (3 EngineAI terms we don't have)**

#### **1. collision** (zqsa01.py:_reward_collision)
```python
return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
```
- **Scale:** -1.0 (legged_gym only)
- **Purpose:** Penalizes unwanted body collisions
- **Our status:** ❌ MISSING
- **Impact:** Low — may cause robot to drag body parts

#### **2. torques** (zqsa01.py:_reward_torques)
```python
return torch.sum(torch.square(self.torques), dim=1)
```
- **Scale:** -1e-10
- **Purpose:** Penalizes high torque usage
- **Our status:** ❌ MISSING
- **Impact:** Negligible — scale is tiny

#### **3. feet_contact_forces** (zqsa01.py:_reward_feet_contact_forces)
```python
return torch.sum((torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) - self.cfg.rewards.max_contact_force).clip(0, 350), dim=1)
```
- **Scale:** -0.02
- **Purpose:** Penalizes excessive contact forces
- **Our status:** ❌ MISSING
- **Impact:** Low — small scale

---

### **❌ CRITICAL BUG: energy reward**

#### **energy** (line 767-768)

**Our implementation:**
```python
rew_energy = cfg.rew_energy * torch.sum(torch.square(actions) * torch.abs(joint_vel), dim=-1)
```

**EngineAI (zqsa01.py:_reward_torques):**
```python
return torch.sum(torch.square(self.torques), dim=1)
```

**Difference:**
- **Our:** `sum(sq(actions) * abs(joint_vel))` — pseudo-energy based on actions and velocity
- **EngineAI:** `sum(sq(torques))` — actual motor torques

**Impact:** We don't have access to torques in IsaacLab's DirectRLEnv workflow (actions are position targets, not torque commands). Our proxy is reasonable but not the same signal.

**Recommendation:** Keep our proxy (it's the best we can do without torque access), but note the difference.

---

## **Summary of Required Fixes**

| Priority | Term | Issue | Fix |
|----------|------|-------|-----|
| **HIGH** | vel_mismatch | Uses `sum_sq` instead of `norm` for ang_vel | Change line 758 to `torch.norm(ang_vel_b[:, :2], dim=1)` |
| **MEDIUM** | feet_clearance | Continuous norm vs EngineAI's binary count | Consider switching to binary: `(feet_heights > target) * swing_mask` |
| **LOW** | collision | Missing term | Add if body dragging observed |
| **LOW** | feet_contact_forces | Missing term | Add if foot impact issues observed |
| **LOW** | torques | Missing (we use proxy) | Accept our proxy — can't access torques |

---

## **Final Assessment**

**Correct implementations:** 16/22 (73%)
**Formula differences:** 2 (vel_mismatch, feet_clearance)
**Extra terms (ours):** 6 (alive, termination, swing_phase_ground, feet_height_max, lat_vel, force_balance)
**Missing terms (EngineAI):** 3 (collision, torques, feet_contact_forces)

**Overall:** Our reward implementation is **mostly correct** with two formula differences and a few missing/extra terms. The critical fix is `vel_mismatch` using `norm()` instead of `sum_sq()` for angular velocity.