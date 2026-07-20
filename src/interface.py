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
    compute_frame_analysis,
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
        self.satisfaction_sum = 0.0
        self.sample_count = 0
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
        self._set_video_placeholder_style()

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
                    background:#359f82;
                    color:white;
                    border:none;
                    border-radius:8px;
                    font-size:14px;
                    font-weight:bold;
                }

                QPushButton:hover{
                    background:#87359f;
                }

                QPushButton:pressed{
                    background:#87359f;
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
            "Итог", "#2196F3"
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

    def _set_video_placeholder_style(self):
        self.video_label.setStyleSheet("""
            QLabel {
                background-color: #1f1f1f;
                color: white;
                border: 2px solid #6BE6C5;
                border-radius: 12px;
                font-size: 20px;
            }
        """)

    def _satisfaction_color(self, satisfaction):
        if satisfaction < 40:
            return "#f44336"
        if satisfaction < 60:
            return "#FFC107"
        return "#4CAF50"

    def _reset_analysis(self):
        self.satisfaction = 50.0
        self.positive_total = 0.0
        self.negative_total = 0.0
        self.satisfaction_sum = 0.0
        self.sample_count = 0
        self.last_analysis_time = 0.0
        self._set_video_placeholder_style()
        self._update_analysis_ui()

    def _update_analysis_ui(self, final=False):
        satisfaction = int(round(self.satisfaction))
        prefix = "Результат" if final else "Удовлетворённость"
        color = self._satisfaction_color(satisfaction)
        self.satisfaction_label.setText(f"{prefix}: {satisfaction}%")
        self.satisfaction_label.setStyleSheet(
            f"font-size:16px; font-weight:bold; color:{color};"
        )

        pos_bar, pos_val = self.positive_value
        neg_bar, neg_val = self.negative_value
        sat_bar, sat_val = self.satisfaction_column_value

        avg_positive = self.positive_total / self.sample_count if self.sample_count else 0
        avg_negative = self.negative_total / self.sample_count if self.sample_count else 0

        pos_bar.setValue(min(int(round(avg_positive)), 100))
        neg_bar.setValue(min(int(round(avg_negative)), 100))
        sat_bar.setValue(satisfaction)

        pos_val.setText(f"{avg_positive:.0f}%")
        neg_val.setText(f"{avg_negative:.0f}%")
        sat_val.setText(f"{satisfaction}%")

    def _apply_emotion_to_analysis(self, probs):
        now = time.time()
        if now - self.last_analysis_time < 0.25:
            return

        frame_satisfaction, positive, negative = compute_frame_analysis(probs)
        self.satisfaction_sum += frame_satisfaction
        self.positive_total += positive
        self.negative_total += negative
        self.sample_count += 1
        self.satisfaction = self.satisfaction_sum / self.sample_count
        self.last_analysis_time = now
        self._update_analysis_ui()

    def _show_center_result(self):
        satisfaction = int(round(self.satisfaction))
        color = self._satisfaction_color(satisfaction)

        if self.sample_count == 0:
            text = "Результат\n\nНет данных"
            color = "#9e9e9e"
        else:
            text = f"Результат\n\n{satisfaction}%"

        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(text)
        self.video_label.setStyleSheet(f"""
            QLabel {{
                background-color: #1f1f1f;
                color: {color};
                border: 3px solid {color};
                border-radius: 12px;
                font-size: 56px;
                font-weight: bold;
                padding: 40px;
            }}
        """)

    def start_recognition(self):
        if self.running:
            return
        
        # Шаг 1: Пробуем запуститься в режиме для WSL 2 (сжатие, V4L2)
        print("Попытка запуска камеры в режиме WSL 2...")
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        if self.cap.isOpened():
            # Настраиваем кодек и разрешение для WSL
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(1.0)
            
            # Проверяем, отдаёт ли камера кадры (боремся с select() timeout)
            ret, _ = self.cap.read()
            if not ret:
                print("Режим WSL 2 не отдал кадры. Откатываемся к стандартному режиму...")
                self.cap.release()
                self.cap = cv2.VideoCapture(0)  # Дефолтный запуск
        else:
            # Шаг 2: Если CAP_V4L2 вообще не поддерживается (например, на обычной Windows)
            print("Режим WSL 2 недоступен. Запуск в стандартном режиме...")
            self.cap = cv2.VideoCapture(0)

        # Финальная проверка: открылась ли камера хотя бы в одном из режимов
        if not self.cap.isOpened():
            self.camera_status.setText("Камера: ошибка")
            print("Ошибка: Не удалось открыть камеру ни в одном из режимов.")
            return

        self._reset_analysis()
        self.running = True
        self.camera_status.setText("Камера: включена")
        self.timer.start(40)
        print("Камера успешно запущена!")

    def stop_recognition(self):
        self.running = False
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.camera_status.setText("Камера: выключена")
        self.face_status.setText("Лицо: не обнаружено")
        self.emotion_status.setText("Эмоция: ---")
        self._update_analysis_ui(final=True)
        self._show_center_result()

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
                    emotion, confidence, probs = predict_emotion(face)
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
                    self._apply_emotion_to_analysis(probs)

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
