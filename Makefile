# PM01 Walking — IsaacLab Training
# Usage: make install && make train

SHELL  := /bin/bash
PYTHON := /workspace/isaaclab/_isaac_sim/python.sh
PIP    := $(PYTHON) -m pip

# ─── Configuration ───────────────────────────────────────────
TASK        := Isaac-PM01-Walking-Direct-v0
NUM_ENVS    := 4096
ISAACLAB    := /workspace/isaaclab
TRAIN_SCRIPT := $(ISAACLAB)/scripts/reinforcement_learning/rsl_rl/train.py
PLAY_SCRIPT  := $(ISAACLAB)/scripts/reinforcement_learning/rsl_rl/play.py
LOG_DIR     := /workspace/armrobot/ArmRobotLegging/logs/rsl_rl/pm01_walking

# ─── Setup ───────────────────────────────────────────────────
.PHONY: install uninstall

install:  ## Install the ArmRobotLegging package (editable)
	$(PIP) install -e source/ArmRobotLegging

uninstall:  ## Uninstall the package
	$(PIP) uninstall -y ArmRobotLegging

# ─── Training ────────────────────────────────────────────────
.PHONY: train train-headless train-small

train:  ## Train with GUI (4096 envs)
	@mkdir -p /workspace/armrobot/logs
	$(PYTHON) $(TRAIN_SCRIPT) \
		--task $(TASK) \
		--num_envs $(NUM_ENVS) 2>&1 | tee $(TRAIN_LOG)

TRAIN_LOG   := /workspace/armrobot/logs/train_$$(date +%Y-%m-%d_%H-%M-%S).log

train-headless:  ## Train without GUI (faster, for remote/SSH)
	@mkdir -p /workspace/armrobot/logs
	$(PYTHON) $(TRAIN_SCRIPT) \
		--task $(TASK) \
		--num_envs $(NUM_ENVS) \
		--headless 2>&1 | tee $(TRAIN_LOG)

train-small:  ## Train with fewer envs (for testing/debugging)
	$(PYTHON) $(TRAIN_SCRIPT) \
		--task $(TASK) \
		--num_envs 64

# ─── Resume Training ─────────────────────────────────────────
.PHONY: resume

resume:  ## Resume training from latest checkpoint
	$(PYTHON) $(TRAIN_SCRIPT) \
		--task $(TASK) \
		--num_envs $(NUM_ENVS) \
		--resume

# ─── Visualization / Play ────────────────────────────────────
.PHONY: play play-latest

play:  ## Play a trained policy (set CHECKPOINT=path/to/model.pt)
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Usage: make play CHECKPOINT=/path/to/model.pt"; \
		exit 1; \
	fi
	$(PYTHON) $(PLAY_SCRIPT) \
		--task $(TASK) \
		--num_envs 32 \
		--checkpoint $(CHECKPOINT)

play-latest:  ## Play the latest checkpoint from logs
	$(PYTHON) $(PLAY_SCRIPT) \
		--task $(TASK) \
		--num_envs 32 \
		--checkpoint $$(find $(LOG_DIR) -name "model_*.pt" | sort | tail -1)

# ─── Utilities ───────────────────────────────────────────────
.PHONY: list-envs logs clean-logs help

list-envs:  ## List all registered IsaacLab environments
	$(PYTHON) $(ISAACLAB)/scripts/environments/list_envs.py

logs:  ## Show training log directory
	@echo "Log directory: $(LOG_DIR)"
	@ls -lht $(LOG_DIR)/ 2>/dev/null || echo "No logs yet. Run 'make train' first."

clean-logs:  ## Remove all training logs (use with caution)
	@echo "This will delete all logs in $(LOG_DIR)"
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] && rm -rf $(LOG_DIR) && echo "Cleaned." || echo "Cancelled."

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
