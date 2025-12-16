# How to Monitor Training Progress

## Method 1: TensorBoard (Visual Monitoring)

### Step 1: Start TensorBoard

Open a **NEW terminal** (don't close the training one):

```bash
# Navigate to project folder
cd C:\Users\Admin\Projects\Machine_Learning_Bot

# Activate environment
venv\Scripts\activate

# Start TensorBoard
tensorboard --logdir=logs/
```

You'll see:
```
TensorBoard 2.20.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

### Step 2: Open in Browser

Open your browser and go to: **http://localhost:6006**

---

## What to Look For in TensorBoard

### 📊 Main Graph: "rollout/ep_len_mean"

**This is the KEY indicator!**

- **X-axis**: Shows timesteps (0 to 500,000)
- **Y-axis**: Shows episode length
- **Line**: Updates in real-time as training progresses

**Training is DONE when:**
- ✅ X-axis reaches **500,000** (or your total_timesteps)
- ✅ Graph stops updating for 2+ minutes
- ✅ No new data points appear

---

## Key Metrics to Watch

### 1. **time/total_timesteps** (Most Important!)

Location: `SCALARS` tab → Search "total_timesteps"

**What it shows:**
- Current progress: e.g., "125000 / 500000"
- Updates every few minutes

**Training is done when: Value = 500,000**

### 2. **rollout/ep_rew_mean** (Episode Reward)

Shows: Average reward per episode
- **Negative at start** = Agent is learning
- **Increasing trend** = Agent is improving
- **Stabilizing** = Agent has learned

### 3. **train/loss** (Training Loss)

Shows: How well the agent is learning
- **Decreasing** = Good learning
- **Stable** = Converged

### 4. **time/fps** (Frames Per Second)

Shows: Training speed
- Typical: 50-200 FPS on CPU
- If **FPS = 0** for 5+ minutes → Training might be stuck

---

## Visual Guide

```
TensorBoard Interface:

┌─────────────────────────────────────────────┐
│ [SCALARS] [GRAPHS] [DISTRIBUTIONS] [IMAGES]│
├─────────────────────────────────────────────┤
│                                             │
│  time/total_timesteps                       │
│  ┌────────────────────────────────────┐    │
│  │                              ___   │    │
│  │                         ____/      │    │
│  │                    ____/           │    │
│  │            ___────/                │    │
│  │    _______/                        │    │
│  │___/                                │    │
│  └────────────────────────────────────┘    │
│     0      100k    200k    300k   500k ✓   │
│                                             │
│  rollout/ep_rew_mean (Episode Rewards)     │
│  ┌────────────────────────────────────┐    │
│  │           _____                    │    │
│  │       ___/     \___                │    │
│  │   ___/            \___             │    │
│  │__/                    \___         │    │
│  └────────────────────────────────────┘    │
│                                             │
└─────────────────────────────────────────────┘

When X-axis reaches 500k → TRAINING DONE! ✓
```

---

## Method 2: Check Terminal Output

### If training is in foreground:

You'll see:
```
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 245      |
|    ep_rew_mean     | -0.125   |
| time/              |          |
|    total_timesteps | 125000   | ← Watch this number!
---------------------------------
```

**Training is done when you see:**
```
================================================================================
TRAINING COMPLETE
================================================================================

[6] Saving final model to: models/ppo_forex_v1

================================================================================
STEP 5: EVALUATE AGENT
================================================================================
```

### If training is in background:

Check the output file:
```bash
# In your terminal
type C:\Users\Admin\AppData\Local\Temp\claude\tasks\<task_id>.output
```

Or check if process is running:
```bash
# Check if Python is running
tasklist | findstr python
```

---

## Method 3: Check Files (Simplest!)

### Training creates checkpoints every 50,000 steps:

```bash
# Check checkpoints folder
dir models\checkpoints\
```

**You'll see:**
```
ppo_forex_50000_steps.zip   ← After 50k steps
ppo_forex_100000_steps.zip  ← After 100k steps
ppo_forex_150000_steps.zip  ← After 150k steps
...
ppo_forex_500000_steps.zip  ← TRAINING COMPLETE!
```

**Training is done when:**
- ✅ You see `ppo_forex_500000_steps.zip`
- ✅ File `models/ppo_forex_v1.zip` exists (final model)

---

## Quick Status Check Commands

### Check if still training:

```bash
# Check Python processes
tasklist | findstr python

# If you see "python.exe" → Still training
# If nothing → Training finished (or crashed)
```

### Check latest checkpoint:

```bash
dir /O-D models\checkpoints\*.zip | more

# Shows newest checkpoint first
# Last number = current progress
```

### Check logs timestamp:

```bash
dir /O-D logs\PPO_*\events.out.tfevents*

# If timestamp is recent → Still training
# If timestamp is old (30+ min ago) → Likely done or stuck
```

---

## Estimated Time Markers

| Timesteps | Time Elapsed | % Complete | What to Expect |
|-----------|--------------|------------|----------------|
| 50,000    | ~10 min      | 10%        | Initial learning |
| 100,000   | ~20 min      | 20%        | Strategy forming |
| 250,000   | ~50 min      | 50%        | Halfway! |
| 400,000   | ~80 min      | 80%        | Refining strategy |
| 500,000   | ~100 min     | 100%       | **DONE!** |

*(Times vary based on CPU speed)*

---

## Troubleshooting

### Graph stopped updating but timesteps < 500k?

**Possible issues:**
1. Training crashed → Check terminal for errors
2. Computer went to sleep → Wake it up
3. Process killed → Restart training

**Solution:**
```bash
# Check if Python is running
tasklist | findstr python

# If not running, restart training from latest checkpoint
python src/training/train_ppo.py
```

### TensorBoard shows "No data"?

**Solution:**
1. Wait 2-3 minutes (data buffers before showing)
2. Refresh browser (F5)
3. Check logs folder exists: `dir logs\`

---

## When Training is DONE

You'll see in terminal:
```
================================================================================
TRAINING COMPLETE
================================================================================

EVALUATION SUMMARY
================================================================================
Mean Return: -2.34% ± 5.67%
Mean Drawdown: 3.21%
Max Drawdown: 4.89%
Win Rate: 40.0%
================================================================================

Next steps:
1. View training logs: tensorboard --logdir=logs/
2. Test model: Load with PPO.load('models/ppo_forex_v1')
================================================================================
```

**Final files created:**
- ✅ `models/ppo_forex_v1.zip` - Final trained model
- ✅ `models/best/best_model.zip` - Best model on validation
- ✅ `logs/PPO_2/` - Complete training logs

---

## Summary: 3 Ways to Know Training is Done

1. **TensorBoard**: X-axis reaches 500,000 timesteps ✓
2. **Files**: See `ppo_forex_500000_steps.zip` in checkpoints ✓
3. **Terminal**: See "TRAINING COMPLETE" message ✓

**Any ONE of these = Training is finished!**
