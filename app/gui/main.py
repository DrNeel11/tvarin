"""Desktop app entry point. Milestone 1 scope only: hardware detection ->
model selection. Agent UI, project workspace, and everything past model
selection are later milestones -- this window intentionally stops after
picking a model so that stage can be tested and shown on its own.

Run: python -m app.gui.main
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QMainWindow, QStackedWidget

from app.core.hardware import HardwareSnapshot
from app.gui.setup_screen import SetupScreen
from app.gui.model_screen import ModelScreen


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ramer")
        self.resize(560, 640)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.setup_screen = SetupScreen()
        self.model_screen = ModelScreen()
        self.stack.addWidget(self.setup_screen)
        self.stack.addWidget(self.model_screen)

        self.setup_screen.continue_clicked.connect(self._go_to_model_screen)

    def _go_to_model_screen(self, snapshot: HardwareSnapshot) -> None:
        self.model_screen.load_for_hardware(snapshot)
        self.stack.setCurrentWidget(self.model_screen)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
