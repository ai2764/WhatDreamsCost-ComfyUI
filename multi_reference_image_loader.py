import torch
import torch.nn.functional as F

import comfy.utils


class MultiReferenceImageLoader:
    """Combine up to 4 IMAGE inputs into a single batched IMAGE tensor for LTXDirector global references.

    When width/height > 0, each input is scaled (preserving aspect ratio) to fit inside
    the target canvas, then center-padded with black to exact (height, width). Set both
    to 0 to auto-size to the max H/W across inputs. All dimensions are snapped to
    `multiple_of`.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "multiple_of": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1}),
                "interpolation": (["lanczos", "bilinear", "bicubic", "nearest"],),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("reference_batch",)
    FUNCTION = "combine"
    CATEGORY = "WhatDreamsCost"

    @staticmethod
    def _take_first(image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError(f"Expected IMAGE tensor of shape [B,H,W,3], got {tuple(image.shape)}")
        return image[:1]

    @staticmethod
    def _snap_down(value: int, multiple: int) -> int:
        if multiple <= 1:
            return max(1, value)
        return max(multiple, (value // multiple) * multiple)

    @staticmethod
    def _snap_up(value: int, multiple: int) -> int:
        if multiple <= 1:
            return max(1, value)
        return max(multiple, ((value + multiple - 1) // multiple) * multiple)

    @staticmethod
    def _resize(image: torch.Tensor, new_w: int, new_h: int, interpolation: str) -> torch.Tensor:
        chw = image.permute(0, 3, 1, 2)
        if interpolation == "lanczos":
            chw = comfy.utils.lanczos(chw, new_w, new_h)
        else:
            chw = F.interpolate(chw, size=(new_h, new_w), mode=interpolation)
        return chw.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def _pad_center(image: torch.Tensor, target_h: int, target_w: int) -> torch.Tensor:
        _, h, w, _ = image.shape
        if h == target_h and w == target_w:
            return image
        pad_left = (target_w - w) // 2
        pad_right = target_w - w - pad_left
        pad_top = (target_h - h) // 2
        pad_bottom = target_h - h - pad_top
        chw = image.permute(0, 3, 1, 2)
        padded = F.pad(chw, (pad_left, pad_right, pad_top, pad_bottom), value=0.0)
        return padded.permute(0, 2, 3, 1).contiguous()

    def combine(self, width, height, multiple_of, interpolation,
                image_1=None, image_2=None, image_3=None, image_4=None):
        inputs = [img for img in (image_1, image_2, image_3, image_4) if img is not None]
        if not inputs:
            side = max(multiple_of, 64)
            return (torch.zeros((1, side, side, 3), dtype=torch.float32),)

        frames = [self._take_first(img) for img in inputs]

        if width > 0 and height > 0:
            target_w = self._snap_up(width, multiple_of)
            target_h = self._snap_up(height, multiple_of)
        else:
            max_h = max(f.shape[1] for f in frames)
            max_w = max(f.shape[2] for f in frames)
            target_w = self._snap_up(max_w, multiple_of)
            target_h = self._snap_up(max_h, multiple_of)

        out = []
        for f in frames:
            _, src_h, src_w, _ = f.shape
            # Scale down to fit when input exceeds target on any axis (preserve AR), snapped to multiple.
            ratio = min(target_w / src_w, target_h / src_h, 1.0)
            new_w = self._snap_down(max(multiple_of, int(round(src_w * ratio))), multiple_of)
            new_h = self._snap_down(max(multiple_of, int(round(src_h * ratio))), multiple_of)
            # Clamp in case snap pushed past target.
            new_w = min(new_w, target_w)
            new_h = min(new_h, target_h)
            if (new_w, new_h) != (src_w, src_h):
                f = self._resize(f, new_w, new_h, interpolation)
            f = self._pad_center(f, target_h, target_w)
            out.append(f)

        batch = torch.cat(out, dim=0)
        return (batch,)
