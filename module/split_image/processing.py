"""High level helpers for automatic image splitting workflow."""
from __future__ import annotations

import os
from pathlib import Path
from threading import Event

import cv2
import numpy as np

from ..image_utils import (
    crop_to_content,
    iter_image_files,
    load_image,
    save_with_dpi,
    split_spread,
)


def initSplitImage(worker) -> None:
    source_dir = Path(worker.fileurl)
    destination_dir = Path(worker.directoryName)
    destination_dir.mkdir(parents=True, exist_ok=True)

    files = iter_image_files(source_dir)
    total = len(files)
    processed = 0

    worker._manual_split_entries = []

    if not files:
        return

    print("__СТАРТ РАЗДЕЛЕНИЕ ПО СТРАНИЦАМ__")

    for file_path in files:
        result = worker.parseImage(file_path)
        if result:
            processed += 1
            worker.proc.emit(int(processed * 100 / total))

    manual_entries = getattr(worker, "_manual_split_entries", [])
    if getattr(worker, "isManualSplitAdjust", False) and manual_entries:
        if hasattr(worker, "log"):
            worker.log.emit("Ожидание ручной корректировки середины")
        print(f"[INFO] Найдено {len(manual_entries)} файлов для ручной корректировки")

        wait_event = Event()
        result_holder = {
            "accepted": False,
            "width_img": worker.width_img,
            "height_img": worker.height_img,
        }
        payload = {
            "entries": manual_entries,
            "wait_event": wait_event,
            "result": result_holder,
            "thread": worker,
        }
        worker.manualAdjustmentRequested.emit(payload)
        wait_event.wait()
        accepted = bool(result_holder.get("accepted"))
        if "width_img" in result_holder:
            try:
                worker.width_img = int(result_holder["width_img"] or 0)
            except Exception:
                pass
        if "height_img" in result_holder:
            try:
                worker.height_img = int(result_holder["height_img"] or 0)
            except Exception:
                pass
        print("[INFO] Слот ручной корректировки завершил работу")

        if hasattr(worker, "log"):
            status = "принята" if accepted else "отменена"
            worker.log.emit(f"Ручная корректировка {status}, продолжаем обработку")
            worker.log.emit("Сохранение результатов ручной корректировки")
        print(f"[INFO] Ручная корректировка { 'принята' if accepted else 'отменена' }")

        worker._manual_split_entries = []


def _fit_page_to_canvas(page: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize *page* proportionally to fit inside the target canvas."""

    if page.ndim == 2:
        page = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    if page.shape[0] == target_height and page.shape[1] == target_width:
        return page

    scale = min(target_width / page.shape[1], target_height / page.shape[0])
    scale = max(scale, 1e-6)

    new_width = max(1, int(round(page.shape[1] * scale)))
    new_height = max(1, int(round(page.shape[0] * scale)))

    interpolation = cv2.INTER_LANCZOS4 if scale >= 1 else cv2.INTER_AREA
    resized = cv2.resize(page, (new_width, new_height), interpolation=interpolation)

    channels = 1 if resized.ndim == 2 else resized.shape[2]
    canvas = np.zeros((target_height, target_width, channels), dtype=resized.dtype)

    y_offset = (target_height - new_height) // 2
    x_offset = (target_width - new_width) // 2
    canvas[y_offset : y_offset + new_height, x_offset : x_offset + new_width] = resized
    return canvas


def trim_page_to_resolution(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    *,
    anchor_horizontal: str = "center",
    anchor_vertical: str = "center",
) -> np.ndarray:
    """Trim *image* so that it matches the requested resolution.

    The function crops the page when it exceeds the target dimensions and pads it
    with transparent/black pixels when it falls short. This guarantees that all
    resulting images share the same size when a resolution is provided.
    """

    if image is None or image.ndim < 2:
        return image

    target_width = int(target_width or 0)
    target_height = int(target_height or 0)
    height, width = image.shape[:2]

    if target_width <= 0 and target_height <= 0:
        return image

    left = 0
    top = 0
    right = width
    bottom = height

    if target_width > 0 and width > target_width:
        excess = width - target_width
        if anchor_horizontal == "left":
            right = target_width
            left = 0
        elif anchor_horizontal == "right":
            left = width - target_width
            right = width
        else:
            left = excess // 2
            right = left + target_width
        left = max(0, left)
        right = min(width, right)

    if target_height > 0 and height > target_height:
        excess = height - target_height
        if anchor_vertical == "top":
            bottom = target_height
            top = 0
        elif anchor_vertical == "bottom":
            top = height - target_height
            bottom = height
        else:
            top = excess // 2
            bottom = top + target_height
        top = max(0, top)
        bottom = min(height, bottom)

    if target_width > 0 and right - left > target_width:
        adjustment = (right - left) - target_width
        if anchor_horizontal == "right":
            left += adjustment
        elif anchor_horizontal == "left":
            right -= adjustment
        else:
            left += adjustment // 2
            right = left + target_width

    if target_height > 0 and bottom - top > target_height:
        adjustment = (bottom - top) - target_height
        if anchor_vertical == "bottom":
            top += adjustment
        elif anchor_vertical == "top":
            bottom -= adjustment
        else:
            top += adjustment // 2
            bottom = top + target_height

    left = max(0, min(left, width - 1 if width > 0 else 0))
    top = max(0, min(top, height - 1 if height > 0 else 0))
    right = max(left + 1, min(right, width))
    bottom = max(top + 1, min(bottom, height))

    cropped = image[top:bottom, left:right]
    if cropped is None or cropped.ndim < 2:
        return cropped

    cropped_height, cropped_width = cropped.shape[:2]
    desired_width = int(target_width) if target_width > 0 else cropped_width
    desired_height = int(target_height) if target_height > 0 else cropped_height

    desired_width = max(desired_width, cropped_width)
    desired_height = max(desired_height, cropped_height)

    if desired_width == cropped_width and desired_height == cropped_height:
        return cropped

    def _offset(total: int, size: int, anchor: str) -> int:
        if total <= size:
            return 0
        anchor = (anchor or "center").lower()
        if anchor in {"right", "bottom"}:
            return total - size
        if anchor in {"left", "top"}:
            return 0
        return max(0, (total - size) // 2)

    channels = 1 if cropped.ndim == 2 else cropped.shape[2]
    if cropped.ndim == 2:
        canvas = np.zeros((desired_height, desired_width), dtype=cropped.dtype)
    else:
        canvas = np.zeros((desired_height, desired_width, channels), dtype=cropped.dtype)

    x_offset = _offset(desired_width, cropped_width, anchor_horizontal)
    y_offset = _offset(desired_height, cropped_height, anchor_vertical)

    if cropped.ndim == 2:
        canvas[y_offset : y_offset + cropped_height, x_offset : x_offset + cropped_width] = cropped
    else:
        canvas[
            y_offset : y_offset + cropped_height,
            x_offset : x_offset + cropped_width,
            :,
        ] = cropped

    return canvas


def parseImage(worker, file_path: Path) -> str | None:
    file_path = Path(file_path)
    relative = Path(os.path.relpath(file_path, worker.fileurl))
    target_dir = Path(worker.directoryName) / relative.parent
    target_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(file_path)

    original_root = getattr(worker, "original_fileurl", worker.fileurl)
    original_candidate = Path(original_root) / relative
    if original_candidate.exists():
        original_path = original_candidate
    else:
        original_path = file_path

    if original_path == file_path:
        original_image = image
    else:
        try:
            original_image = load_image(original_path)
        except ValueError:
            original_image = image
            original_path = file_path

    manual_mode = bool(getattr(worker, "isManualSplitAdjust", False))

    original_height, original_width = original_image.shape[:2]
    manual_crop_left = 0
    manual_crop_top = 0
    manual_crop_right = original_width
    manual_crop_bottom = original_height

    analysis_image = image

    def _extract_manual_view(src: np.ndarray) -> np.ndarray:
        if src is None or not src.size:
            return src

        left = max(0, min(manual_crop_left, src.shape[1] - 1))
        right = max(left + 1, min(manual_crop_right, src.shape[1]))
        top = max(0, min(manual_crop_top, src.shape[0] - 1))
        bottom = max(top + 1, min(manual_crop_bottom, src.shape[0]))

        view = src[top:bottom, left:right]
        return view if view.size else src

    if getattr(worker, "isRemoveBorder", False):
        pad = worker.border_px if getattr(worker, "isAddBorder", False) else 0
        cropped, bounds = crop_to_content(original_image, pad_x=pad, pad_y=pad)
        if cropped is not None and bounds is not None:
            manual_crop_left = int(bounds.left)
            manual_crop_top = int(bounds.top)
            manual_crop_right = int(bounds.right)
            manual_crop_bottom = int(bounds.bottom)
            if manual_mode:
                analysis_image = _extract_manual_view(original_image)
            else:
                image = cropped
                analysis_image = image
        else:
            analysis_image = original_image if manual_mode else image
    else:
        analysis_image = original_image if manual_mode else image

    if analysis_image is None or not analysis_image.size:
        analysis_image = original_image if manual_mode else image

    if manual_mode:
        height, width = analysis_image.shape[:2]
    else:
        height, width = image.shape[:2]

    if height >= width:
        save_path = target_dir / relative.name
        if manual_mode:
            manual_view = _extract_manual_view(original_image)
            save_source = manual_view if manual_view is not None and manual_view.size else original_image
        else:
            save_source = image

        if worker.isPxIdentically and save_source is not None and getattr(save_source, "size", 0):
            target_width = int(worker.width_img or 0)
            target_height = int(worker.height_img or 0)
            if target_width <= 0 or target_height <= 0:
                target_width = int(save_source.shape[1])
                target_height = int(save_source.shape[0])
                worker.width_img = target_width
                worker.height_img = target_height
            save_source = _fit_page_to_canvas(save_source, target_width, target_height)
        save_with_dpi(save_source, save_path, worker.dpi)
        return str(save_path)

    try:
        spread_source = analysis_image if manual_mode else image
        force_centre = bool(getattr(worker, "isSplitByPixels", False))
        spread = split_spread(
            spread_source,
            worker.width_px,
            worker.pxMediumVal,
            force_centre=force_centre,
        )
    except Exception as exc:
        print(f"[WARN] Не удалось разделить {file_path}: {exc}")
        save_path = target_dir / relative.name
        if manual_mode:
            manual_view = _extract_manual_view(original_image)
            save_source = manual_view if manual_view is not None and manual_view.size else original_image
        else:
            save_source = analysis_image if analysis_image is not None and analysis_image.size else image
        save_with_dpi(save_source, save_path, worker.dpi)
        return str(save_path)

    number = relative.stem
    left_name = f"{number}_1.jpg"
    right_name = f"{number}_2.jpg"

    left_path = target_dir / left_name
    right_path = target_dir / right_name

    if getattr(worker, "isManualSplitAdjust", False):
        if not hasattr(worker, "_manual_split_entries"):
            worker._manual_split_entries = []

        entry_data = {
            "source_path": str(original_path),
            "relative": str(relative),
            "target_dir": str(target_dir),
            "left_path": str(left_path),
            "right_path": str(right_path),
            "auto_split_x": int(spread.split_x),
            "split_x": int(spread.split_x),
            "overlap": int(worker.width_px),
            "crop_left": manual_crop_left,
            "crop_top": manual_crop_top,
            "crop_right": manual_crop_right,
            "crop_bottom": manual_crop_bottom,
            "rotation_deg": 0.0,
            "image_width": original_width,
            "image_height": original_height,
            "split_disabled": False,
        }
        worker._manual_split_entries.append(entry_data)
        return str(file_path)

    for page_image, page_path in ((spread.left, left_path), (spread.right, right_path)):
        if worker.isPxIdentically and page_image is not None and getattr(page_image, "size", 0):
            target_width = int(worker.width_img or 0)
            target_height = int(worker.height_img or 0)
            if target_width <= 0 or target_height <= 0:
                target_width = int(page_image.shape[1])
                target_height = int(page_image.shape[0])
                worker.width_img = target_width
                worker.height_img = target_height
            page_image = _fit_page_to_canvas(page_image, target_width, target_height)

        save_with_dpi(page_image, page_path, worker.dpi)

    return str(left_path)


__all__ = [
    "initSplitImage",
    "parseImage",
    "trim_page_to_resolution",
    "_fit_page_to_canvas",
]
