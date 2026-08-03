"""
Backwards-compatible facade over the modules this file was split into.

utils.py had grown to ~1560 lines covering four unrelated concerns (GPIO
hardware, overlay drawing, image encode/save, reject audit logging), which
made it both hard to navigate and a magnet for unrelated changes. The code
now lives in:

    dio.py            — DI/DO pin access (libapmi/ctypes + subprocess fallback)
    visualization.py  — draw_* overlay helpers
    image_io.py       — encode/save frames, AsyncImageSaver, BackgroundResultEmitter
    reject_log.py     — reject action audit log

Everything is re-exported here so existing `from .utils import X` call sites
keep working unchanged — including the private GPIO symbols that ai_services/
test.py reaches for. New code should import from the specific module above
rather than adding to this list.

Deliberately NOT re-exported (nothing outside their own module uses them):
`logger`, and reject_log's `_reject_logger` / `_reject_log_file`. The latter
two are lazily assigned on the first log call, so a `from ... import` here
would snapshot None forever and silently never see the real logger — import
reject_log directly if you ever need them.
"""

from .dio import (  # noqa: F401
    _LIBAPMI_PIN_MAP,
    _apmi_dio_read_input,
    _apmi_dio_read_output,
    _apmi_dio_write_output,
    _gpio_lock,
    _init_libapmi,
    _libapmi,
    _pulse_locks,
    _use_native_gpio,
    _write_do_raw,
    check_trigger_edge,
    read_di_value,
    trigger_reject_pulse,
    write_do_value,
)
from .image_io import (  # noqa: F401
    AsyncImageSaver,
    BackgroundResultEmitter,
    encode_frame_for_display,
    encode_image_to_base64,
    resize_for_display,
    save_and_encode_frame,
)
from .reject_log import (  # noqa: F401
    _init_reject_logger,
    log_reject_cancelled,
    log_reject_end,
    log_reject_start,
)
from .visualization import (  # noqa: F401
    draw_center_points,
    draw_color_match_overlay,
    draw_detected_obb_boxes,
    draw_inference_bboxes,
)

__all__ = [
    # dio
    "read_di_value",
    "check_trigger_edge",
    "write_do_value",
    "trigger_reject_pulse",
    # visualization
    "draw_inference_bboxes",
    "draw_center_points",
    "draw_detected_obb_boxes",
    "draw_color_match_overlay",
    # image_io
    "resize_for_display",
    "encode_image_to_base64",
    "encode_frame_for_display",
    "save_and_encode_frame",
    "AsyncImageSaver",
    "BackgroundResultEmitter",
    # reject_log
    "log_reject_start",
    "log_reject_end",
    "log_reject_cancelled",
]
