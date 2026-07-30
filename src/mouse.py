import numpy as np
import matplotlib.pyplot as plt
from pynput import mouse, keyboard
import pyautogui   # для получения разрешения экрана


class MouseTracker:
    def __init__(self):
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Разрешение экрана: {self.screen_width}×{self.screen_height}")

        self.positions = []        # координаты движений
        self.clicks = []           # координаты кликов
        self.running = True
        self.mouse_listener = None
        self.keyboard_listener = None

    def on_move(self, x, y):
        if self.running:
            self.positions.append((x, y))

    def on_click(self, x, y, button, pressed):
        if self.running and pressed:   
            self.clicks.append((x, y))

    def on_press(self, key):
        if key == keyboard.Key.esc:
            self.running = False
            if self.mouse_listener:
                self.mouse_listener.stop()
            if self.keyboard_listener:
                self.keyboard_listener.stop()
            return False

    def start(self):
        self.mouse_listener = mouse.Listener(
            on_move=self.on_move,
            on_click=self.on_click,
        )
        self.mouse_listener.start()

        self.keyboard_listener = keyboard.Listener(on_press=self.on_press)
        self.keyboard_listener.start()

        print("Трекинг мыши запущен. Нажмите ESC для остановки и построения тепловой карты.")

        self.mouse_listener.join()
        self.keyboard_listener.join()

        print("Сбор данных завершён.")
        self.generate_heatmap()

    def generate_heatmap(self, bins=100):
        if not self.positions and not self.clicks:
            print("Нет данных.")
            return

        x_coords = []
        y_coords = []
        weights = []

        for x, y in self.positions:
            x_coords.append(x)
            y_coords.append(y)
            weights.append(1)

        for x, y in self.clicks:
            x_coords.append(x)
            y_coords.append(y)
            weights.append(5)   # клики имеют больший вес

        x_coords = np.array(x_coords)
        y_coords = np.array(y_coords)
        weights = np.array(weights)

        x_min, x_max = 0, self.screen_width
        y_min, y_max = 0, self.screen_height

        H, xedges, yedges = np.histogram2d(
            x_coords, y_coords,
            bins=bins,
            range=[[x_min, x_max], [y_min, y_max]],
            weights=weights
        )
        H = H.T  

        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(H, origin='upper', extent=[0, self.screen_width, self.screen_height, 0],
                       cmap='hot', aspect='auto')
        ax.set_title('Тепловая карта активности мыши', fontsize=14)
        ax.set_xlabel('X (пиксели)', fontsize=12)
        ax.set_ylabel('Y (пиксели)', fontsize=12)
        plt.colorbar(im, label='Интенсивность (вес)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    tracker = MouseTracker()
    tracker.start()
