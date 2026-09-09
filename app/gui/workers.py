"""QThread workers so hardware detection and model pulls never block the UI
thread. Hardware detection itself is fast (a few subprocess calls), but a
model pull can run for minutes, and Qt's event loop must keep processing
paint/input events the whole time or the window looks frozen -- the exact
"is it hung or just working" problem this project already hit once with
mini_agent.py's streaming, solved here by construction instead of by
timeout tuning.
"""
from __future__ import annotations

from PySide6.QtCore import QThread, Signal

from app.core.hardware import HardwareSnapshot, detect_hardware
from app.core.model_runtime import OllamaRuntime


class HardwareDetectWorker(QThread):
    finished_ok = Signal(object)  # HardwareSnapshot

    def run(self) -> None:
        snapshot: HardwareSnapshot = detect_hardware()
        self.finished_ok.emit(snapshot)


class ModelPullWorker(QThread):
    progress = Signal(dict)  # raw Ollama pull chunk
    finished_ok = Signal(bool)

    def __init__(self, model: str, parent=None):
        super().__init__(parent)
        self.model = model

    def run(self) -> None:
        runtime = OllamaRuntime()
        ok = runtime.pull(self.model, on_progress=lambda chunk: self.progress.emit(chunk))
        self.finished_ok.emit(ok)
