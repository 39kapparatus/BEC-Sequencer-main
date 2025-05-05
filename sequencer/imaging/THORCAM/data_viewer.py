import sys
import json
import numpy as np
from PyQt5.QtWidgets import QApplication, QMenuBar,QMainWindow, QLabel, QTextEdit, QVBoxLayout, QHBoxLayout, QPushButton, QWidget, QAction, QFileDialog
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QEvent

from sequencer.imaging.THORCAM.imaging_software import DataItem


class DataItemViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.data_item = None
        self.current_image_index = 0

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Data Item Viewer')
        self.setGeometry(100, 100, 800, 600)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFocusPolicy(Qt.StrongFocus)  # Ensure the label can receive focus

        self.json_text = QTextEdit(self)
        self.json_text.setReadOnly(True)
        self.json_text.setFocusPolicy(Qt.NoFocus)  # Prevent the text edit from taking focus

        self.next_button = QPushButton('Next', self)
        self.next_button.clicked.connect(self.next_image)

        self.prev_button = QPushButton('Previous', self)
        self.prev_button.clicked.connect(self.prev_image)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.json_text)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.create_menu()

        self.image_label.setFocus()  # Set initial focus to the label

    def create_menu(self):
        menubar = QMenuBar(self)
        file_menu = menubar.addMenu('File')

        load_action = QAction('Load', self)
        load_action.triggered.connect(self.load_data_item)
        file_menu.addAction(load_action)

        layout = QVBoxLayout()
        layout.setMenuBar(menubar)
        self.setLayout(layout)

    def load_data_item(self):
        options = QFileDialog.Options()
        filename, _ = QFileDialog.getOpenFileName(self, "Load Data Item", "", "NPZ Files (*.npz);;All Files (*)", options=options)
        if filename:
            self.data_item = DataItem.load(filename)
            print("self.data_item.images")
            print(len(self.data_item.images))
            self.json_text.setText(self.data_item.json_str)
            self.current_image_index = 0
            self.show_image(self.current_image_index)

    def show_image(self, index):
        if self.data_item and 0 <= index < len(self.data_item.images):
            image_array = self.data_item.images[index]
            height, width = image_array.shape
            bytes_per_line = width
            q_image = QImage(image_array.data, width, height, bytes_per_line, QImage.Format_Grayscale8)
            pixmap = QPixmap.fromImage(q_image)
            self.image_label.setPixmap(pixmap)

    def next_image(self):
        if self.data_item:
            self.current_image_index = (self.current_image_index + 1) % len(self.data_item.images)
            self.show_image(self.current_image_index)

    def prev_image(self):
        if self.data_item:
            self.current_image_index = (self.current_image_index - 1) % len(self.data_item.images)
            self.show_image(self.current_image_index)

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_F5:
            self.close()
        elif e.key() == Qt.Key_D:
            self.next_button.click()
        elif e.key() == Qt.Key_A:
            self.prev_button.click()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    viewer = DataItemViewer()
    viewer.show()
    sys.exit(app.exec_())
