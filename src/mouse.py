import numpy as np
import matplotlib.pyplot as plt
from PyQt6.QtCore import QThread, pyqtSignal
import time
import os
from datetime import datetime
from pynput import mouse


class MouseTracker(QThread):
    status_updated = pyqtSignal(str)
    data_collected = pyqtSignal(int, int)
    
    def __init__(self):
        super().__init__()
        try:
            import pyautogui
            self.screen_width, self.screen_height = pyautogui.size()
        except:
            self.screen_width, self.screen_height = 1920, 1080
        print(f"Разрешение экрана: {self.screen_width}×{self.screen_height}")

        self.positions = []
        self.clicks = []
        self.running = False
        self.is_tracking = False
        self.start_time = None
        self.last_pos = None
        self.listener = None
        
    def run(self):
        """Запуск трекинга в отдельном потоке"""
        self.running = True
        self.is_tracking = True
        self.start_time = time.time()
        self.positions = []
        self.clicks = []
        self.last_pos = None
        
        self.status_updated.emit("Трекинг мыши запущен. Двигайте мышью и кликайте!")
        print("Трекинг мыши запущен. Двигайте мышью и кликайте!")
        print(f"Сбор данных начат в {datetime.now().strftime('%H:%M:%S')}")
        
        def on_move(x, y):
            if self.is_tracking:
                self.positions.append((x, y))
                if len(self.positions) % 50 == 0:
                    print(f"Собрано точек: {len(self.positions)}")
        
        def on_click(x, y, button, pressed):
            if self.is_tracking and pressed:
                self.clicks.append((x, y))
                print(f"Клик в ({x}, {y})")
        
        self.listener = mouse.Listener(on_move=on_move, on_click=on_click)
        self.listener.start()
        
        while self.running and self.is_tracking:
            time.sleep(0.1)
        
        if self.listener:
            self.listener.stop()
            self.listener = None
        
        self.is_tracking = False
        print(f"ИТОГО собрано позиций: {len(self.positions)}, кликов: {len(self.clicks)}")
        
        if len(self.positions) > 0:
            elapsed = time.time() - self.start_time
            self.status_updated.emit(
                f"Трекинг остановлен. Собрано: {len(self.positions)} позиций, "
                f"{len(self.clicks)} кликов за {elapsed:.1f}с"
            )
            self.data_collected.emit(len(self.positions), len(self.clicks))
            self.generate_heatmap()
        else:
            self.status_updated.emit("Нет данных для построения тепловой карты")
            print("Нет данных для построения тепловой карты")
        
        self.finished.emit()

    def stop_tracking(self):
        """Остановка трекинга"""
        print("Остановка трекинга...")
        self.running = False
        self.is_tracking = False

    def generate_heatmap(self, bins=80):
        """Генерация тепловой карты и сохранение в файл"""
        if not self.positions:
            print("Нет данных для построения тепловой карты")
            self.status_updated.emit("Нет данных для построения тепловой карты")
            return

        print(f"Строим тепловую карту из {len(self.positions)} точек...")

        x_coords = [p[0] for p in self.positions]
        y_coords = [p[1] for p in self.positions]

        print(f"Диапазон X: {min(x_coords)} - {max(x_coords)}")
        print(f"Диапазон Y: {min(y_coords)} - {max(y_coords)}")

        H, xedges, yedges = np.histogram2d(
            x_coords, y_coords,
            bins=bins,
            range=[[0, self.screen_width], [0, self.screen_height]]
        )
        H = H.T

        if np.sum(H) == 0:
            print("Гистограмма пуста!")
            self.status_updated.emit("Ошибка: гистограмма пуста")
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            
            fig, ax = plt.subplots(figsize=(14, 10))
            
            im = ax.imshow(H, origin='upper', 
                          extent=[0, self.screen_width, self.screen_height, 0],
                          cmap='hot', aspect='auto', interpolation='bilinear')
            
            ax.set_title('Тепловая карта активности мыши', fontsize=16, fontweight='bold')
            ax.set_xlabel('X (пиксели)', fontsize=12)
            ax.set_ylabel('Y (пиксели)', fontsize=12)
            
            if self.clicks:
                click_x = [p[0] for p in self.clicks]
                click_y = [p[1] for p in self.clicks]
                ax.scatter(click_x, click_y, color='cyan', s=30, alpha=0.7, 
                          label=f'Клики ({len(self.clicks)})', edgecolors='white', linewidth=0.5)
            
            ax.grid(True, alpha=0.1, linestyle='--')
            
            stats_text = (
                f"Всего точек: {len(self.positions)}\n"
                f"Кликов: {len(self.clicks)}\n"
                f"Время: {time.time() - self.start_time:.1f}с\n"
                f"Разрешение: {self.screen_width}×{self.screen_height}"
            )
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            if self.clicks:
                ax.legend(loc='upper right')
            
            plt.colorbar(im, label='Интенсивность', fraction=0.046, pad=0.04)
            plt.tight_layout()
            
            os.makedirs("heatmaps", exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"heatmap_{timestamp}.png"
            filepath = os.path.join("heatmaps", filename)
            
            plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            
            print(f"Тепловая карта сохранена в: {filepath}")
            print(f"Размер файла: {os.path.getsize(filepath)} байт")
            self.status_updated.emit(f"Тепловая карта сохранена в {filepath}")
            
            try:
                if os.name == 'posix':
                    os.system(f'xdg-open "{filepath}" 2>/dev/null &')
                elif os.name == 'nt':
                    os.startfile(filepath)
                print(f"Открываю файл: {filepath}")
            except:
                print(f"Тепловая карта сохранена в: {filepath}")
            
        except Exception as e:
            print(f"Ошибка при построении тепловой карты: {e}")
            import traceback
            traceback.print_exc()
            self.status_updated.emit(f"Ошибка: {e}")