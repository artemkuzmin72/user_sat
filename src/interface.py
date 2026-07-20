import sys
import time
import cv2
import mediapipe as mp

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QFrame,
    QGroupBox,
    QProgressBar,
)
from PyQt6.QtGui import QFont, QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

from main import (
    face_landmarker,
    draw_mesh,
    crop_face,
    predict_emotion,
    get_emotion_delta,
)


class DexPilotUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Dex Pilot")
        self.resize(1200, 700)

        self.cap = None
        self.running = False
        self.show_mesh = True
        self.satisfaction = 50.0
        self.positive_total = 0.0
        self.negative_total = 0.0
        self.last_emotion = None
        self.last_analysis_time = 0.0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.init_ui()
        self._update_analysis_ui()
        self.start_button.clicked.connect(self.start_recognition)
        self.stop_button.clicked.connect(self.stop_recognition)
        self.face_mesh_button.clicked.connect(self.toggle_face_mesh)

    def init_ui(self):
        main_layout = QHBoxLayout(self)

        left_layout = QVBoxLayout()

        title = QLabel("Dex Pilot")
        title.setFont(QFont("Arial", 20, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.video_label = QLabel("Нажмите «Запустить распознавание»")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 600)
        self.video_label.setStyleSheet("""
            QLabel{
                background-color:#6BE6C5;
                color:white;
                border:2px solid #7AE79F;
                border-radius:12px;
                font-size:20px;
            }
        """)

        left_layout.addWidget(title)
        left_layout.addWidget(self.video_label)

        control_panel = QFrame()
        control_panel.setFixedWidth(340)
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
        self.face_mesh_button = QPushButton("Скрыть карту лица")

        for button in [self.start_button, self.stop_button, self.face_mesh_button]:
            button.setMinimumHeight(45)
            button.setStyleSheet("""
                QPushButton{
                    background:#7536D2;
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

        analysis_group = QGroupBox("Анализ")
        analysis_layout = QVBoxLayout()

        self.satisfaction_label = QLabel("Удовлетворённость: 50%")
        self.satisfaction_label.setStyleSheet(
            "font-size:16px; font-weight:bold; color:#4CAF50;"
        )
        self.satisfaction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        columns_layout = QHBoxLayout()
        columns_layout.setSpacing(12)

        self.positive_bar, self.positive_value = self._create_column(
            "Позитив", "#4CAF50"
        )
        self.negative_bar, self.negative_value = self._create_column(
            "Негатив", "#f44336"
        )
        self.satisfaction_bar, self.satisfaction_column_value = self._create_column(
            "Итог", "#B121F3"
        )

        columns_layout.addStretch()
        columns_layout.addLayout(self.positive_bar)
        columns_layout.addLayout(self.negative_bar)
        columns_layout.addLayout(self.satisfaction_bar)
        columns_layout.addStretch()

        analysis_layout.addWidget(self.satisfaction_label)
        analysis_layout.addLayout(columns_layout)

        analysis_group.setLayout(analysis_layout)

        panel_layout.addWidget(group)
        panel_layout.addSpacing(20)
        panel_layout.addWidget(info_group)
        panel_layout.addSpacing(20)
        panel_layout.addWidget(analysis_group)
        panel_layout.addStretch()

        main_layout.addLayout(left_layout)
        main_layout.addWidget(control_panel)

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

    def _create_column(self, title, color):
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        label = QLabel(title)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label.setStyleSheet("font-size:12px; font-weight:bold;")

        bar = QProgressBar()
        bar.setOrientation(Qt.Orientation.Vertical)
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        bar.setFixedSize(56, 110)
        bar.setStyleSheet(f"""
            QProgressBar {{
                background: #1a1a1a;
                border: 1px solid #444;
                border-radius: 6px;
            }}
            QProgressBar::chunk {{
                background: {color};
                border-radius: 5px;
            }}
        """)

        value_label = QLabel("0")
        value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        value_label.setStyleSheet("font-size:13px; font-weight:bold;")

        layout.addWidget(label)
        layout.addWidget(bar, alignment=Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(value_label)

        return layout, (bar, value_label)

    def _reset_analysis(self):
        self.satisfaction = 50.0
        self.positive_total = 0.0
        self.negative_total = 0.0
        self.last_emotion = None
        self.last_analysis_time = 0.0
        self._update_analysis_ui()

    def _update_analysis_ui(self):
        satisfaction = int(round(self.satisfaction))
        self.satisfaction_label.setText(f"Удовлетворённость: {satisfaction}%")

        color = "#4CAF50"
        if satisfaction < 40:
            color = "#f44336"
        elif satisfaction < 60:
            color = "#FFC107"
        self.satisfaction_label.setStyleSheet(
            f"font-size:16px; font-weight:bold; color:{color};"
        )

        pos_bar, pos_val = self.positive_value
        neg_bar, neg_val = self.negative_value
        sat_bar, sat_val = self.satisfaction_column_value

        pos_bar.setValue(min(int(self.positive_total), 100))
        neg_bar.setValue(min(int(self.negative_total), 100))
        sat_bar.setValue(satisfaction)

        pos_val.setText(f"+{self.positive_total:.0f}")
        neg_val.setText(f"-{self.negative_total:.0f}")
        sat_val.setText(str(satisfaction))

    def _apply_emotion_to_analysis(self, emotion, confidence):
        now = time.time()
        if emotion == self.last_emotion and now - self.last_analysis_time < 1.0:
            return

        delta = get_emotion_delta(emotion, confidence)
        if delta == 0:
            self.last_emotion = emotion
            self.last_analysis_time = now
            return

        self.satisfaction = max(0, min(100, self.satisfaction + delta))
        if delta > 0:
            self.positive_total += delta
        else:
            self.negative_total += abs(delta)

        self.last_emotion = emotion
        self.last_analysis_time = now
        self._update_analysis_ui()

    def start_recognition(self):
        if self.running:
            return
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.camera_status.setText("Камера: ошибка")
            return
        self._update_analysis_ui()
        self.running = True
        self.camera_status.setText("Камера: включена")
        self.timer.start(33)

    def stop_recognition(self):
        self.running = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_status.setText("Камера: выключена")
        self.face_status.setText("Лицо: не обнаружено")
        self.emotion_status.setText("Эмоция: ---")
        self._update_analysis_ui()
        self.video_label.setText("Нажмите «Запустить распознавание»")
        self.video_label.setPixmap(QPixmap())

    def toggle_face_mesh(self):
        self.show_mesh = not self.show_mesh
        self.face_mesh_button.setText(
            "Скрыть карту лица" if self.show_mesh else "Показать карту лица"
        )

    def update_frame(self):
        if not self.running or self.cap is None:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.stop_recognition()
            return

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        timestamp = int(time.time() * 1000)
        result = face_landmarker.detect_for_video(mp_image, timestamp)

        face_detected = False
        if result.face_landmarks:
            for landmarks in result.face_landmarks:
                face_detected = True
                if self.show_mesh:
                    draw_mesh(frame, landmarks)
                face = crop_face(frame, landmarks)
                if face is not None:
                    emotion, confidence = predict_emotion(face)
                    text = f"{emotion}: {confidence:.1f}%"
                    cv2.putText(
                        frame,
                        text,
                        (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (0, 255, 0),
                        3,
                    )
                    self.emotion_status.setText(
                        f"Эмоция: {emotion}"
                    )
                    self._apply_emotion_to_analysis(emotion, confidence)

        self.face_status.setText(
            "Лицо: обнаружено" if face_detected else "Лицо: не обнаружено"
        )
        if not face_detected:
            self.emotion_status.setText("Эмоция: ---")

        display_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = display_rgb.shape
        qt_image = QImage(
            display_rgb.data,
            w,
            h,
            ch * w,
            QImage.Format.Format_RGB888,
        )
        pixmap = QPixmap.fromImage(qt_image).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.stop_recognition()
        event.accept()
