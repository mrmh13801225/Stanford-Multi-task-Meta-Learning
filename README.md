# Stanford Multi-task Meta-Learning

This repository contains solutions and experiments for multiple Stanford CS330 homework assignments on multitask learning and meta-learning.

## Repository Layout

Canonical homework folders:

- `HW0/` — MovieLens multitask recommendation homework (`main.py`, supporting modules, and report notebook/PDF).
- `HW1/` — Memory-augmented neural network (MANN) homework on Omniglot (`hw1.py`, notebook, and handout).
- `HW2/` — Meta-learning homework with MAML and ProtoNet implementations (`maml.py`, `protonet.py`, notebooks, and handout).

Comprehensive directory aliases are provided for readability:

- `homework_0/` → `HW0/`
- `homework_1/` → `HW1/`
- `homework_2/` → `HW2/`

## Setup

Use Python 3.10+ and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies per homework directory (requirements are separate):

```bash
pip install -r /home/runner/work/Stanford-Multi-task-Meta-Learning/Stanford-Multi-task-Meta-Learning/homework_0/requirements.txt
pip install -r /home/runner/work/Stanford-Multi-task-Meta-Learning/Stanford-Multi-task-Meta-Learning/homework_1/requirements.txt
pip install -r /home/runner/work/Stanford-Multi-task-Meta-Learning/Stanford-Multi-task-Meta-Learning/homework_2/requirements.txt
```

## Running

Run each homework from its own directory so relative paths resolve correctly.

### Homework 0

```bash
cd /home/runner/work/Stanford-Multi-task-Meta-Learning/Stanford-Multi-task-Meta-Learning/homework_0
python main.py
```

### Homework 1

```bash
cd /home/runner/work/Stanford-Multi-task-Meta-Learning/Stanford-Multi-task-Meta-Learning/homework_1
python hw1.py
```

### Homework 2

```bash
cd /home/runner/work/Stanford-Multi-task-Meta-Learning/Stanford-Multi-task-Meta-Learning/homework_2
python maml.py
python protonet.py
```

## Notes

- Some scripts download datasets/checkpoints automatically if missing.
- TensorBoard logs and model checkpoints are stored inside homework folders (for example, `run/` and `logs/`).
