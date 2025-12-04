import datetime
import io
import logging
import os
import re
import subprocess
import tempfile
import uuid
import urllib.parse
import base64
from typing import Any, Dict, List, Optional
from io import BytesIO
import PIL.Image as Image
from hmdriver2.driver import Driver as HarmonyDriver
# try:
#     from hmdriver2.driver import Driver as HarmonyDriver
# except ImportError:  # pragma: no cover - optional dependency
#     HarmonyDriver = None


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Process helpers
# ---------------------------------------------------------------------------


def _run(cmd: List[str], timeout: int = 30) -> bytes:
    """Run external command and return raw stdout (raises on non-zero exit)."""
    logger.debug("$ %s", " ".join(cmd))
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=timeout)


def _resolve_hdc_binary() -> List[str]:
    """Return command prefix for hdc (handles ROOT_DIR / HDC_COMMAND overrides)."""
    override = os.environ.get("HDC_COMMAND")
    if override:
        return [override]

    # Prefer a project-local `hdc` binary if present so the repository
    # can include a bundled executable that works out-of-the-box for others.
    default_name = "hdc.exe" if os.name == "nt" else "hdc"
    this_dir = os.path.dirname(__file__)
    project_candidate = os.path.join(this_dir, "hdc", default_name)
    if os.path.exists(project_candidate):
        return [project_candidate]

    # Fall back to ROOT_DIR if provided (legacy behaviour).
    root = os.environ.get("ROOT_DIR")
    if root:
        candidate = os.path.join(root, default_name)
        if os.path.exists(candidate):
            return [candidate]

    # Finally return the default name so the system PATH can resolve it.
    return [default_name]


def _hdc_prefix(serial: Optional[str]) -> List[str]:
    base = _resolve_hdc_binary()
    if serial:
        return base + ["-t", serial]
    return base


def _resize_pillow(origin_img: Image.Image, max_line_res: int = 1120) -> Image.Image:
    """Resize image so that the longest edge is <= max_line_res using Lanczos."""
    w, h = origin_img.size
    if max_line_res is not None:
        max_line = max_line_res
        if h > max_line:
            w = int(w * max_line / h)
            h = max_line
        if w > max_line:
            h = int(h * max_line / w)
            w = max_line
    return origin_img.resize((w, h), resample=Image.Resampling.LANCZOS)


# ---------------------------------------------------------------------------
# HarmonyDevice class
# ---------------------------------------------------------------------------

class HarmonyDevice:
    """Interact with a HarmonyOS device via hdc."""

    _TEXT_DRIVER_UNAVAILABLE_MSG = (
        "Text input requires the optional hmdriver2 package; install it or "
        "extend HarmonyDevice._handle_type()."
    )

    _keycodes: Dict[str, int] = {
        "HOME": 1,
        "BACK": 2,
        "MENU": 3,
        "ENTER": 2054,
    }

    def __init__(self, serial: Optional[str]):
        self.serial: Optional[str] = serial
        self.width: int = 0
        self.height: int = 0
        self.last_req_time: datetime.datetime = datetime.datetime.now()
        self._driver = self._init_text_driver()

    # ---------- internal ----------
    def _hdc(self, *args: str, timeout: int = 30) -> bytes:
        print(args)
        return _run(_hdc_prefix(self.serial) + list(args), timeout)

    def _init_text_driver(self):
        if HarmonyDriver is None:
            logger.debug(
                "hmdriver2 not available; falling back to uitest command for text")
            return None

        try:
            return HarmonyDriver(serial=self.serial)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to initialise Harmony driver: %s", exc)
            return None

    def _send_key(self, keyname: str) -> None:
        if keyname not in self._keycodes:
            raise ValueError(
                f"Unsupported PRESS value for Harmony device: {keyname}")
        code = self._keycodes[keyname]
        code_str = str(code)
        self._hdc("shell", "uinput", "-K", "-d", code_str, "-u", code_str)

    # ---------- public API ----------
    def refresh_resolution(self) -> None:
        """Query snapshot_display metadata to populate width/height."""
        raw = self._hdc("shell", "snapshot_display",
                        "/data/local/tmp/").decode(errors="ignore")
        match = re.search(r"width.*\s(\d+)\s*,\s*height.*\s(\d+)", raw)
        if not match:
            raise RuntimeError(
                f"Failed to parse snapshot_display output: {raw}")
        self.width, self.height = map(int, match.groups())
        logger.info("Device %s resolution: %dx%d",
                    self.serial or "<default>", self.width, self.height)

    def step(self, data: Dict[str, Any]) -> bool:
        """Execute a control step (tap/swipe/key/text/clear). Returns True if STATUS=finish/impossible."""
        logger.debug("Step: %s", data)
        if "POINT" in data:
            self._handle_point(data)
        if "PRESS" in data:
            self._handle_press(data["PRESS"])
        if "TYPE" in data:
            self._handle_type(data["TYPE"])
        if "CLEAR" in data:
            self._handle_clear()
        self.last_req_time = datetime.datetime.now()

        status = data.get("STATUS")
        if status in {"finish", "impossible"}:
            logger.info("Task finished: %s", status)
            return True
        return False

    def state(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "last_req_time": self.last_req_time.isoformat(),
            "screenshot": self.screenshot(),
        }

    def encode_image(image_path: Optional[str] = None,
                     byte_stream: Optional[bytes] = None) -> str:
        if image_path is None and byte_stream is None:
            raise ValueError("args [image_path] and [byte_stream] should not empty for all.")

        if image_path is not None and byte_stream is not None:
            raise ValueError("args [image_path] and [byte_stream] should have values for one.")

        if image_path:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        else:
            return base64.b64encode(byte_stream).decode('utf-8')

    # def screenshot(self, max_side: Optional[int] = None) -> Image.Image:
    #     """Capture screen via snapshot_display; return Pillow Image (optionally downscaled)."""
    #     remote_path = f"/data/local/tmp/{uuid.uuid4().hex}.png"
    #     fd, local_path = tempfile.mkstemp(prefix="hdc_screen_", suffix=".png")
    #     os.close(fd)
    #     try:
    #         self._hdc("shell", "snapshot_display", "-f", remote_path)
    #         self._hdc("file", "recv", remote_path, local_path)
    #         with open(local_path, "rb") as fh:
    #             raw_png = fh.read()
    #     finally:
    #         try:
    #             self._hdc("shell", "rm", remote_path)
    #         except subprocess.CalledProcessError:
    #             logger.debug(
    #                 "Remote screenshot cleanup failed for %s", remote_path)
    #         try:
    #             os.remove(local_path)
    #         except OSError:
    #             logger.debug(
    #                 "Local screenshot cleanup failed for %s", local_path)
    #
    #     img = Image.open(io.BytesIO(raw_png))
    #     if max_side is not None:
    #         img = _resize_pillow(img, max_side)
    #     return img

    # =================== private helpers ===================
    def _handle_point(self, data: Dict[str, Any]) -> None:
        x, y = data["POINT"]
        x = int(x / 1000 * self.width)
        y = int(y / 1000 * self.height)
        if "to" in data:
            x2, y2 = self._compute_swipe_target(data["to"], x, y)
            duration = str(data.get("duration", 150))
            self._hdc("shell", "uinput", "-T", "-m", str(x),
                      str(y), str(x2), str(y2), duration)
        else:
            self._hdc("shell", "uinput", "-T", "-c", str(x), str(y))

    def _compute_swipe_target(self, target: Any, x: int, y: int) -> tuple[int, int]:
        if isinstance(target, list):
            x2, y2 = target
            x2 = int(x2 / 1000 * self.width)
            y2 = int(y2 / 1000 * self.height)
            return x2, y2
        dirs = {
            "up": (0, -0.15),
            "down": (0, 0.15),
            "left": (-0.15, 0),
            "right": (0.15, 0),
        }
        if target not in dirs:
            raise ValueError(f"Invalid swipe direction: {target}")
        dx_ratio, dy_ratio = dirs[target]
        x2 = int(max(min(x + dx_ratio * self.width, self.width), 0))
        y2 = int(max(min(y + dy_ratio * self.height, self.height), 0))
        return x2, y2

    def _handle_press(self, key: str) -> None:
        self._send_key(key)

    def _handle_type(self, raw: str) -> None:
        text = urllib.parse.unquote(raw)
        if self._driver is not None:
            self._driver.input_text(text)
            return

        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        self._hdc("shell", f'uitest uiInput inputText 1 1 "{escaped}"')

        # self.driver.input_text(escaped)
        # self.run_hdc_command(f'hdc shell uinput -K -d 2054 -u 2054', self.device_id)


    def _handle_clear(self) -> None:
        # Harmony does not expose KEYCODE_CLEAR via documented constants; emulate via uitest.
        self._hdc("shell", "uitest uiInput deleteText 0")


# ---------------------------------------------------------------------------
# Public utility
# ---------------------------------------------------------------------------

def setup_device() -> HarmonyDevice:
    """Detect first connected Harmony device and return HarmonyDevice instance."""
    lines = _run(_resolve_hdc_binary() +
                 ["list", "targets"]).decode(errors="ignore").splitlines()
    serials: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("list targets"):
            continue
        if "Empty" in stripped:
            continue
        serials.append(stripped.split()[0])
    if not serials:
        raise RuntimeError(
            "No available Harmony devices found. Connect a device and ensure hdc recognises it.")
    if len(serials) > 1:
        logger.warning(
            "Multiple devices detected; defaulting to the first (%s).", serials[0])
    dev = HarmonyDevice(serials[0])
    dev.refresh_resolution()
    return dev


if __name__ == "__main__":
    device = setup_device()
    logger.info("Device ready: serial=%s (%dx%d)",
                device.serial, device.width, device.height)
    device.step({"POINT": [500, 500]})
    png = device.screenshot()
    target = os.path.join(os.path.dirname(__file__), "screencap_hdc.png")
    png.save(target)
    logger.info("Screenshot saved -> %s", target)
