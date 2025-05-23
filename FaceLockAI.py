import sys
import subprocess
import cv2
import tkinter as tk
from tkinter import simpledialog, messagebox
import numpy as np
import time
import pickle
import os
import platform

# Проверка и установка недостающих библиотек
try:
    import face_recognition
except ImportError:
    print("Установка библиотеки face_recognition...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "face_recognition"])
    import face_recognition

class FaceLockAI:
    def __init__(self, master):
        self.master = master
        master.title("FaceLock AI")

        # Настройка камеры
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            messagebox.showerror("Ошибка", "Не удалось открыть камеру!")
            master.destroy()
            return

        # Файл для сохранения лиц
        self.known_faces_file = "known_faces.dat"
        self.known_face_encodings = []
        self.known_face_names = []

        # Флаги
        self.is_locked = False
        self.show_webcam = False  # Для отладки

        # Загружаем известные лица
        self.load_known_faces()

        # Если нет сохранённых лиц → обучаем
        if not self.known_face_encodings:
            messagebox.showinfo("Обучение", "Смотрите в камеру 5 секунд для регистрации вашего лица.")
            self.train_my_face()

        # Создание кнопок
        self.add_user_button = tk.Button(master, text="Добавить пользователя", command=self.add_new_face)
        self.add_user_button.pack(pady=10)

        self.remove_user_button = tk.Button(master, text="Удалить пользователя", command=self.remove_face)
        self.remove_user_button.pack(pady=10)

        # Запускаем детектор
        self.update()

    def load_known_faces(self):
        if os.path.exists(self.known_faces_file):
            with open(self.known_faces_file, "rb") as f:
                data = pickle.load(f)
                self.known_face_encodings = data["encodings"]
                self.known_face_names = data["names"]

    def save_known_faces(self):
        data = {"encodings": self.known_face_encodings, "names": self.known_face_names}
        with open(self.known_faces_file, "wb") as f:
            pickle.dump(data, f)

    def train_my_face(self):
        my_face_encodings = []
        start_time = time.time()

        while time.time() - start_time < 5:
            ret, frame = self.camera.read()
            if not ret:
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for face_encoding in face_encodings:
                my_face_encodings.append(face_encoding)
                if self.show_webcam:
                    cv2.putText(frame, "Обучение...", (30, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow("Обучение", frame)
                    cv2.waitKey(1)

        if my_face_encodings:
            self.known_face_encodings.append(np.mean(my_face_encodings, axis=0))
            self.known_face_names.append("Я")
            self.save_known_faces()
            messagebox.showinfo("Успех", "Обучение завершено!")
        else:
            messagebox.showerror("Ошибка", "Лицо не найдено!")

    def add_new_face(self):
        name = simpledialog.askstring("Добавить лицо", "Введите имя:")
        if name:
            my_face_encodings = []
            start_time = time.time()

            while time.time() - start_time < 5:
                ret, frame = self.camera.read()
                if not ret:
                    continue

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                for face_encoding in face_encodings:
                    my_face_encodings.append(face_encoding)
                    if self.show_webcam:
                        cv2.putText(frame, "Обучение...", (30, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.imshow("Обучение", frame)
                        cv2.waitKey(1)

            if my_face_encodings:
                self.known_face_encodings.append(np.mean(my_face_encodings, axis=0))
                self.known_face_names.append(name)
                self.save_known_faces()
                messagebox.showinfo("Успех", f"Обучение нового лица {name} завершено!")
            else:
                messagebox.showerror("Ошибка", "Лицо не найдено!")

    def remove_face(self):
        name = simpledialog.askstring("Удалить лицо", "Введите имя пользователя для удаления:")
        if name in self.known_face_names:
            index = self.known_face_names.index(name)
            del self.known_face_encodings[index]
            del self.known_face_names[index]
            self.save_known_faces()
            messagebox.showinfo("Успех", f"Пользователь {name} удалён.")
        else:
            messagebox.showerror("Ошибка", f"Пользователь {name} не найден.")

    def lock_screen(self):
        """Блокирует экран через системную команду"""
        if not self.is_locked:
            self.is_locked = True

            # Windows
            if platform.system() == "Windows":
                subprocess.run(["rundll32.exe", "user32.dll,LockWorkStation"], creationflags=subprocess.CREATE_NO_WINDOW)

            # Linux (gnome)
            elif platform.system() == "Linux":
                subprocess.run(["gnome-screensaver-command", "-l"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def unlock_if_safe(self, recognized_faces):
        """Разблокирует, если в кадре только 'Я'"""
        if self.is_locked:
            if all(name == "Я" for name in recognized_faces):
                self.is_locked = False

    def update(self):
        ret, frame = self.camera.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            recognized_faces = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, face_encoding, tolerance=0.5)
                name = "Неизвестный"

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]

                recognized_faces.append(name)

            # Если есть незнакомцы → блокировка
            if "Неизвестный" in recognized_faces:
                self.lock_screen()
            else:
                self.unlock_if_safe(recognized_faces)

            # Отображение камеры (для отладки)
            if self.show_webcam:
                for (top, right, bottom, left), name in zip(face_locations, recognized_faces):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.imshow("Камера", frame)
                cv2.waitKey(1)

        self.master.after(100, self.update)  # Проверка каждые 100 мс

    def on_closing(self):
        self.camera.release()
        cv2.destroyAllWindows()
        self.save_known_faces()
        self.master.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FaceLockAI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()