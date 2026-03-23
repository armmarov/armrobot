# PM01 Walking — Target Gait Reference

Reference analysis of the original EngineAI PM01 walking gait, extracted from
`/home/armmarov/Videos/Screencasts/ori.webm` (16 frames at 4fps, ~4 seconds).

This document defines the **visual and quantitative target** for our RL training.

---

## Visual Analysis (Frame-by-Frame)

### Posture
- Torso stays **very vertical** throughout the gait cycle. Minimal lean or sway.
- Head remains stable — no bobbing or tilting.
- Arms hang naturally, slightly forward, providing passive balance. No wild swinging.

### Gait Cycle
- **Clear alternating stepping** with smooth transitions:
  - Phase 1 (frames 1-3): Left leg in swing (forward), right leg planted (stance)
  - Phase 2 (frames 4-6): Transition, both feet near ground (double support)
  - Phase 3 (frames 7-9): Right leg in swing, left leg planted
  - Phase 4 (frames 10-12): Transition back to double support
  - Phase 5 (frames 13-16): Cycle repeats
- Gait cycle duration: ~0.8 seconds (matching our `cycle_time=0.8`)

### Foot Motion
- **Low foot clearance**: ~3-5 cm off the ground during swing. No exaggerated lifting.
- Feet stay close to the ground — conservative, energy-efficient steps.
- Smooth arc during swing — no jerky or abrupt foot movements.
- Feet land flat, no toe-dragging or heel-striking visible.

### Knee Bend
- **Moderate, natural knee flexion** during swing phase.
- Knee bends are **symmetric** between left and right legs.
- Not stiff-legged, but not exaggerated either.

### Symmetry
- **Perfect left-right symmetry** — both legs move identically but 180 degrees out of phase.
- No wobble, oscillation, or jerkiness in either leg.
- No visible asymmetry in foot force or step height.

### Forward Progression
- Robot clearly progresses forward across frames (moves away from camera).
- Steady forward velocity — no stalling, drifting sideways, or walking backwards.
- Estimated forward speed: ~0.3-0.5 m/s (moderate walking pace).

---

## Quantitative Targets

Based on the visual analysis and EngineAI reference code, these are the target
metrics our RL policy should achieve:

| Metric | Target | Notes |
|--------|--------|-------|
| Forward velocity (vel_x) | 0.3 - 0.5 m/s | Steady, not too fast |
| Foot clearance (swing height) | 0.03 - 0.06 m | Conservative, low steps |
| Swing ratio (L/R) | ~0.50 / 0.50 | Perfect symmetry |
| Foot force balance (L/R) | ~1.0 ratio | Equal weight distribution |
| Episode length | > 900 steps | Full 20s episode survival |
| Gait cycle time | 0.8 s | Matching EngineAI |
| Posture (orientation reward) | High | Upright, minimal lean |

---

## Key Characteristics to Match

1. **Smoothness over speed** — The original gait prioritizes smooth, stable motion
   over fast walking. Our policy should not sacrifice stability for velocity.

2. **Low foot lifts** — The original barely lifts feet off the ground. Our
   `target_feet_height=0.12m` may be too high. Consider 0.05-0.06m.

3. **Symmetric stepping** — Both legs must behave identically. Any wobble or
   asymmetry (like Run 38's left leg wobble) is a failure mode.

4. **Conservative actions** — Small, controlled joint movements. The original
   does not use large action magnitudes.

5. **Steady forward progression** — The robot must actually walk forward, not
   just step in place (Run 39's current issue).

---

## Comparison with Training Runs

| Feature | Original (EngineAI) | Run 37 | Run 38 | Run 39 (early) |
|---------|---------------------|--------|--------|-----------------|
| Forward walking | Yes, steady 0.3-0.5 m/s | Yes (0.45) | Yes (0.606) | No (~0 m/s) |
| Foot clearance | ~3-5 cm | ~6 cm | 6-7 cm | 5-6 cm |
| Left-right symmetry | Perfect | Variable | Left wobble | Good (stepping in place) |
| Knee bend | Natural, moderate | Present | Present | Present |
| Posture | Very upright | Good | Good | Good |
| Overall quality | Target | Good walk, some shuffle | Fast but wobble | Smooth but no forward motion |

---

## Implications for Future Runs

- **target_feet_height** should be reduced from 0.12m to ~0.05-0.06m to match
  the original's low foot clearance. High foot lifts are unnecessary and may
  cause instability.
- **action_scale** of 0.5 (Run 37) produced more natural motion than 0.6 (Run 38).
  The original uses conservative actions.
- **Anti-wobble rewards** (base_acc, knee_distance) are good ideas but may need
  to be balanced against forward velocity rewards to prevent stepping-in-place.
- The original proves that **smooth, small steps with steady forward motion**
  is the goal — not dramatic, high-stepping gaits.

---

*Video source: `/home/armmarov/Videos/Screencasts/ori.webm`*
*Analysis date: 2026-03-11*
