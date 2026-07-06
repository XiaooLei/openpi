import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def make_droid_example() -> dict:
    """Creates a random input example for the Droid policy."""
    return {
        "observation/exterior_image_1_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.random.rand(7),
        "observation/gripper_position": np.random.rand(1),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class DroidInputs(transforms.DataTransformFn):
    # Determines which model will be used.
    model_type: _model.ModelType
    # Optional extra observation keys to append to the proprioceptive state.
    extra_state_keys: tuple[str, ...] = ()

    def __call__(self, data: dict) -> dict:
        gripper_pos = np.asarray(data["observation/gripper_position"])
        if gripper_pos.ndim == 0:
            # Ensure gripper position is a 1D array, not a scalar, so we can concatenate with joint positions
            gripper_pos = gripper_pos[np.newaxis]
        state_parts = [data["observation/joint_position"], gripper_pos]
        for key in self.extra_state_keys:
            val = np.asarray(data[key])
            if val.ndim == 0:
                val = val[np.newaxis]
            state_parts.append(val)
        state = np.concatenate(state_parts)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference
        base_image = _parse_image(data["observation/exterior_image_1_left"])
        wrist_image = _parse_image(data["observation/wrist_image_left"])

        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
                images = (base_image, wrist_image, np.zeros_like(base_image))
                image_masks = (np.True_, np.True_, np.False_)
            case _model.ModelType.PI0_FAST:
                names = ("base_0_rgb", "base_1_rgb", "wrist_0_rgb")
                # We don't mask out padding images for FAST models.
                images = (base_image, np.zeros_like(base_image), wrist_image)
                image_masks = (np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        inputs = {
            "state": state,
            "image": dict(zip(names, images, strict=True)),
            "image_mask": dict(zip(names, image_masks, strict=True)),
        }

        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])

        if "prompt" in data:
            if isinstance(data["prompt"], bytes):
                data["prompt"] = data["prompt"].decode("utf-8")
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DroidSparseActionHistory(transforms.DataTransformFn):
    """Build sparse past delta-action features from LeRobot delta timestamp queries."""

    lags: tuple[int, ...]
    output_key: str = "action_history_delta_sparse"

    def __call__(self, data: dict) -> dict:
        num_lags = len(self.lags)
        if num_lags == 0:
            return data

        actions = np.asarray(data["actions"])
        joint_position = np.asarray(data["joint_position"])
        gripper_position = np.asarray(data["gripper_position"])

        if actions.ndim != 2 or actions.shape[0] <= num_lags or actions.shape[-1] < 8:
            raise ValueError(
                f"Expected actions with shape [history+future, >=8], got {actions.shape} for lags={self.lags}"
            )
        if joint_position.ndim != 2 or joint_position.shape[0] != num_lags + 1 or joint_position.shape[-1] < 7:
            raise ValueError(
                "Expected joint_position with shape [history+current, >=7], "
                f"got {joint_position.shape} for lags={self.lags}"
            )
        if gripper_position.shape[0] != num_lags + 1:
            raise ValueError(
                "Expected gripper_position with shape [history+current, ...], "
                f"got {gripper_position.shape} for lags={self.lags}"
            )

        history_actions = actions[:num_lags]
        history_joints = joint_position[:num_lags]
        joint_deltas = history_actions[:, :7] - history_joints[:, :7]
        gripper_actions = history_actions[:, 7:8]
        history = np.concatenate([joint_deltas, gripper_actions], axis=-1).reshape(-1)

        updated = dict(data)
        updated[self.output_key] = history.astype(actions.dtype, copy=False)
        updated["joint_position"] = joint_position[-1]
        updated["gripper_position"] = np.asarray(gripper_position[-1])
        updated["actions"] = actions[num_lags:]
        return updated


@dataclasses.dataclass(frozen=True)
class DroidOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # Only return the first 8 dims.
        return {"actions": np.asarray(data["actions"][..., :8])}
