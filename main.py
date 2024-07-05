import os
import sys
from io import BytesIO
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QImage
from PyQt6.QtWidgets import (
    QApplication, QStackedWidget, QGraphicsScene, QVBoxLayout, QFileDialog, QDialog, QSizePolicy, QGraphicsView,
    QMessageBox, QWidget
)
from PyQt6.uic import loadUi
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from detector import pipeline


def get_app_dir():
    return getattr(sys, '_MEIPASS', os.path.abspath(os.path.dirname(__file__)))


# Класс, отвечающий за детекцию
class DetectionThread(QThread):
    detection_finished = pyqtSignal(object, list)
    progress_updated = pyqtSignal(int)

    def __init__(self, file_path, model_path, yolo_path):
        super().__init__()
        self.file_path = file_path
        self.model_path = model_path
        self.yolo_path = yolo_path

    def run(self):
        with Image.open(self.file_path) as img:
            buf = BytesIO()
            img.save(buf, format='JPEG')
            byte_im = buf.getvalue()
        reconstructed_img, diams = pipeline(byte_im, self.model_path, self.yolo_path, (640, 640), self.update_progress)
        self.detection_finished.emit(reconstructed_img, diams)

    def update_progress(self, progress):
        self.progress_updated.emit(progress)


class Params(QDialog):
    """Окно для изменения параметров замера"""

    def __init__(self):
        super().__init__()
        loadUi(os.path.join(get_app_dir(), 'params_seed.ui'), self)
        self.setWindowTitle("Параметры замера")


class NewRes(QDialog):
    """Окно для внесения данных для нового замера"""

    def __init__(self, parent=None):
        super().__init__(parent)
        loadUi(os.path.join(get_app_dir(), 'new_res.ui'), self)
        self.setWindowTitle("Новый замер")
        self.apply_btn.clicked.connect(self.apply_data)

    def apply_data(self):
        try:
            fio = self.fio_input.text()
            date = self.date_input.text()
            time = self.time_input.text()
            supplier = self.supplier_input.text()

            if self.parent() is not None:
                self.parent().set_new_res_data(fio, date, time, supplier)
            self.close()
        except Exception as e:
            print(f"Произошла ошибка в apply_data: {e}")


class MainWindow(QDialog):
    """Класс для взаимодействия с основным окном"""

    def __init__(self):
        super().__init__()
        loadUi(os.path.join(get_app_dir(), 'main_form_lay_seed.ui'), self)
        self.choose_file.clicked.connect(self.open_file_dialog)
        self.new_res.clicked.connect(self.open_new_res_dialog)
        self.progress_bar.setVisible(False)
        self.start_btn.clicked.connect(self.run_detection)
        self.save_post.clicked.connect(self.save_post_img)
        self.detection_thread = None

    def resizeEvent(self, event):
        """Обрабатывает событие изменения размера окна для подгонки изображений в QGraphicsView и графика."""
        self.fit_images()
        super().resizeEvent(event)

    def fit_images(self):
        """Подгоняет изображения под размер QGraphicsView."""
        if hasattr(self, 'orig_scene') and self.orig_scene:
            self.orig.fitInView(self.orig_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if hasattr(self, 'post_scene') and self.post_scene:
            self.post.fitInView(self.post_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.draw()
            self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.canvas.updateGeometry()
            if hasattr(self, 'graph_scene') and self.graph_scene:
                self.plot.fitInView(self.graph_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def save_post_img(self):
        if hasattr(self, 'qim'):
            file_path, _ = QFileDialog.getSaveFileName(self, 'Сохранить изображение', '', 'Images (*.png *.jpg *.jpeg)')
            if file_path:
                self.qim.save(file_path)

    def open_new_res_dialog(self):
        self.new_res_dialog = NewRes(self)
        self.new_res_dialog.show()

    def set_new_res_data(self, fio, date, time, supplier):
        try:
            self.fio.setText("ФИО: " + fio)
            self.date.setText("Дата: " + date)
            self.time.setText("Время: " + time)
            self.sup.setText("Поставщик: " + supplier)
        except Exception as e:
            print(f"Произошла ошибка в set_new_res_data: {e}")

    def open_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Выберите изображение', '')
        if fname:
            self.file_path = fname
            self.load_image()

    def load_image(self):
        if self.file_path:
            pixmap = QPixmap(self.file_path)
            self.orig_scene = QGraphicsScene()
            self.orig_scene.addPixmap(pixmap)
            self.orig.setScene(self.orig_scene)
            self.fit_images()

    # Функция предназначена для детектирования семян по классам
    def run_detection(self):
        if not hasattr(self, 'file_path') or not self.file_path:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, загрузите изображение перед запуском.")
            return
        model_path = os.path.join(get_app_dir(), "yolov5/models/modelsweight.pt")
        yolo_path = os.path.join(get_app_dir(), 'yolov5')

        # Запуск детекции в отдельном потоке
        self.detection_thread = DetectionThread(self.file_path, model_path, yolo_path)
        self.detection_thread.detection_finished.connect(self.display_results)
        self.detection_thread.progress_updated.connect(self.update_progress)
        self.progress_bar.setVisible(True)
        self.detection_thread.start()

    def update_progress(self, progress):
        self.progress_bar.setValue(progress)

    def display_results(self, reconstructed_img, diams):
        self.progress_bar.setVisible(False)

        detect_img = Image.fromarray(reconstructed_img, "RGB")
        diams_df = pd.DataFrame(diams)
        self.diams_df = diams_df
        diams_df = diams_df.rename(columns={0: 'diameter'})
        qim = QImage(detect_img.tobytes(), detect_img.size[0], detect_img.size[1], QImage.Format.Format_RGB888)
        self.qim = qim
        pixmap = QPixmap.fromImage(qim)
        self.post_scene = QGraphicsScene()
        self.post_scene.addPixmap(pixmap)
        self.post.setScene(self.post_scene)
        self.fit_images()

        # Настройка стиля Seaborn
        sns.set(style="whitegrid", palette="muted", color_codes=True)

        # Строим график размеров семян
        if not diams_df.empty:
            fig = Figure(figsize=(7, 8))  # Увеличиваем высоту графика
            ax = fig.add_subplot(111)

            # Создание гистограммы и KDE с улучшенным стилем
            sns.histplot(data=diams_df, x='diameter', bins=20, kde=True, ax=ax,
                         line_kws={"linewidth": 2, "linestyle": "--"},
                         color="b", edgecolor="k", alpha=0.7)

            ax.set_xlabel('Диаметр', fontsize=12)
            ax.set_ylabel('Частота', fontsize=12)
            ax.set_title('Распределение диаметров семян', fontsize=12)

            # Устанавливаем размер шрифтов для меток
            ax.tick_params(axis='both', which='major', labelsize=10)

            fig.subplots_adjust(bottom=0.25)  # Увеличиваем нижний отступ

            # Удалить старый canvas если он существует
            if hasattr(self, 'canvas'):
                self.graph_layout.removeWidget(self.canvas)
                self.canvas.deleteLater()
                del self.canvas

            # Добавить новый canvas в layout
            self.canvas = FigureCanvas(fig)
            self.graph_layout = QVBoxLayout(self.plot)  # Обновляем layout для plot
            self.graph_layout.addWidget(self.canvas)
            self.canvas.draw()

            # Установить минимум и максимум размеров для plot
            self.plot.setMinimumSize(400, 300)
            self.plot.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            self.plot.updateGeometry()

            self.fit_images()


app = QApplication(sys.argv)
main_window = MainWindow()
widget = QStackedWidget()
widget.addWidget(main_window)
widget.show()
widget.move(QApplication.primaryScreen().geometry().center() - widget.rect().center())
try:
    app.exec()
except Exception as e:
    print(f"Произошла ошибка: {e}")
finally:
    print("Выход!")
