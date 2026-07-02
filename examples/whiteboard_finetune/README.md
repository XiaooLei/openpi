# OpenPI Whiteboard Finetune

Lightweight code workspace for pi05-DROID whiteboard fine-tuning. This repo
contains the training wrappers, OpenPI config overlay, resume scripts, and
checkpoint selection notes. It intentionally does not contain datasets,
checkpoints, TensorBoard logs, or packaged model tarballs.

## Directory Roles

- `src/openpi/training/config.py` - this branch contains the whiteboard
  training configs directly in OpenPI.
- `src/openpi/policies/droid_policy.py` - this branch adds optional
  `extra_state_keys` support for appending force/torque observations.
- `run_job.sh` - one-line job entry point.
- `run_pi05_droid_zed196_ft.sh` - 196-episode vision/joint-state finetune.
- `run_pi05_droid_zed196_force_ft.sh` - 196-episode force-enabled finetune.
- `resume_complete196_joint_to30k.sh` - resume vision/joint run to 30k total steps.
- `resume_complete196_force_to30k.sh` - resume force run to 30k total steps.
- `packages/SELECTED_CHECKPOINTS_24000.txt` - selected checkpoint metrics and
  force deployment input contract.

## External Dependencies

This example is intended to run from inside this OpenPI fork branch. By default,
the launch scripts infer the OpenPI repo root as `examples/whiteboard_finetune/../..`.
You can still override it with:

```bash
BASE_OPENPI_DIR=/path/to/openpi
```

Base pi05-DROID weights and tokenizer are expected under:

```bash
/inspire/qb-ilm/project/gjjproject/public/xl/openpi-baseline/.cache/openpi
```

Runtime data, assets, logs, and checkpoints are kept outside this code repo.
On this machine the default runtime directory is:

```bash
/inspire/qb-ilm/project/gjjproject/public/xl/wipeboard/pi05_zed145_finetune
```

Override it with:

```bash
OPENPI_WHITEBOARD_RUN_DIR=/path/to/runtime/workdir
```

The launch scripts resolve Python in this order:

```text
OPENPI_PYTHON
PYTHON_BIN
<this repo>/.venv/bin/python3
/inspire/qb-ilm/project/gjjproject/public/xl/projects/droid-whiteboard/openpi/.venv/bin/python3
python3 from PATH
```

TensorBoard is resolved similarly with `OPENPI_TENSORBOARD` or
`TENSORBOARD_BIN`. This lets the fork branch run either with its own venv or
with the previously prepared OpenPI environment.

## Current 196-Episode Runs

Vision/joint-state run:

```bash
cd /path/to/openpi/examples/whiteboard_finetune && TRAIN_VARIANT=complete196 ./run_job.sh complete196_joint_position
```

Force-enabled run:

```bash
cd /path/to/openpi/examples/whiteboard_finetune && TRAIN_VARIANT=complete196_force ./run_job.sh complete196_force
```

Resume to 30k total steps without overwrite:

```bash
cd /path/to/openpi/examples/whiteboard_finetune && ./resume_complete196_joint_to30k.sh
cd /path/to/openpi/examples/whiteboard_finetune && ./resume_complete196_force_to30k.sh
```

The wrappers expose per-device batch size:

```bash
OPENPI_PER_DEVICE_BATCH_SIZE=32
```

The OpenPI global batch size is computed internally as:

```text
per_device_batch_size * jax.device_count()
```

## Train / Eval Split

The 196-episode configs use a fixed random split:

```text
train episodes: 186
eval episodes: 10
eval episode ids: [2, 12, 13, 42, 69, 79, 90, 109, 113, 155]
```

Eval runs every `500` steps. Checkpoints save every `2000` steps.

## Force Input Contract

The force config uses:

```python
extra_state_keys=("observation/force_torque_wrench",)
```

The force vector is:

```text
[fx, fy, fz, tx, ty, tz]
```

The force model state layout before pi05 tokenization is:

```text
dims 00-06: joint_position, 7-D
dim  07:    gripper_position, 1-D
dims 08-13: force_torque_wrench, 6-D
dims 14-31: padding to pi05 32-D state/action interface
```

Do not append force values to the natural-language instruction. The task prompt
stays `wipe the whiteboard`; force is passed as structured observation state and
then normalized/discretized into the pi05 `State:` token field.

## Selected Checkpoints

Selected saved checkpoint step for both 196 variants:

```text
24000
```

See:

```text
packages/SELECTED_CHECKPOINTS_24000.txt
```

The packaged model tarball is intentionally not tracked by Git.
