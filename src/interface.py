import sys
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QGroupBox,
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt


class DexPilotUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dex Pilot")
        self.resize(1200, 700)

        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        # Левая часть (видео)

        left_layout = QVBoxLayout()

        title = QLabel("Dex Pilot")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.video_label = QLabel("Видео с камеры")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel{
                background-color:#1f1f1f;
                color:white;
                border:2px solid #4CAF50;
                border-radius:12px;
                font-size:20px;
            }
        """)

        left_layout.addWidget(title)
        left_layout.addWidget(self.video_label)

        # Правая панель управления

        control_panel = QFrame()
        control_panel.setFixedWidth(300)
        control_panel.setStyleSheet("""
            QFrame{
                background:#2c2c2c;
                border-radius:12px;
            }
        """)

        panel_layout = QVBoxLayout(control_panel)

        group = QGroupBox("Управление")
        group_layout = QVBoxLayout()

        self.start_button = QPushButton("▶ Запустить распознавание")
        self.stop_button = QPushButton("■ Остановить распознавание")
        self.face_mesh_button = QPushButton("👤 Показать карту лица")

        buttons = [
            self.start_button,
            self.stop_button,
            self.face_mesh_button
        ]

        for button in buttons:
            button.setMinimumHeight(45)
            button.setStyleSheet("""
                QPushButton{
                    background:#4CAF50;
                    color:white;
                    border:none;
                    border-radius:8px;
                    font-size:14px;
                    font-weight:bold;
                }

                QPushButton:hover{
                    background:#45a049;
                }

                QPushButton:pressed{
                    background:#2d7d32;
                }
            """)
            group_layout.addWidget(button)

        group.setLayout(group_layout)

        info_group = QGroupBox("Статус")

        info_layout = QVBoxLayout()

        self.camera_status = QLabel("Камера: выключена")
        self.face_status = QLabel("Лицо: не обнаружено")
        self.emotion_status = QLabel("Эмоция: ---")
        self.id_status = QLabel("ID пользователя: ---")

        for lbl in [
            self.camera_status,
            self.face_status,
            self.emotion_status,
            self.id_status,
        ]:
            lbl.setStyleSheet("font-size:14px;")

        info_layout.addWidget(self.camera_status)
        info_layout.addWidget(self.face_status)
        info_layout.addWidget(self.emotion_status)
        info_layout.addWidget(self.id_status)

        info_group.setLayout(info_layout)

        panel_layout.addWidget(group)
        panel_layout.addSpacing(20)
        panel_layout.addWidget(info_group)
        panel_layout.addStretch()

        main_layout.addLayout(left_layout)
        main_layout.addWidget(control_panel)

        # Общий стиль
        
        self.setStyleSheet("""
            QWidget{
                background:#202124;
                color:white;
                font-family:Segoe UI;
            }

            QGroupBox{
                font-size:15px;
                font-weight:bold;
                border:1px solid gray;
                border-radius:8px;
                margin-top:10px;
                padding-top:15px;
            }

            QGroupBox::title{
                subcontrol-origin:margin;
                left:10px;
                padding:0 5px;
            }
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = DexPilotUI()
    window.show()

    sys.exit(app.exec())