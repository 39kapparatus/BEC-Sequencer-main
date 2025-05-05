from sequencer.Sequence.sequence import creat_test


from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QFormLayout, QDialogButtonBox,QMessageBox
)

from PyQt5.QtGui import QPainter, QPolygon, QBrush,QDoubleValidator
import numpy as np

import sys

class SweepEventDialog(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Parameter Sweep Dialog')

        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Form layout
        self.form_layout = QFormLayout()
        self.layout.addLayout(self.form_layout)


        # Sweep type combo box
        self.sweep_type_label = QLabel("Sweep Type:")
        self.sweep_type_combo = QComboBox()
        self.sweep_type_combo.addItems([
            "Linear Sweep", "Logarithmic Sweep", "Geometric Sweep", "Custom Sequence Sweep"
        ])
        self.sweep_type_combo.currentIndexChanged.connect(self.update_sweep_settings)
        self.form_layout.addRow(self.sweep_type_label, self.sweep_type_combo)

        # Sweep settings fields
        self.sweep_settings_layout = QFormLayout()
        self.layout.addLayout(self.sweep_settings_layout)
        self.sweep_settings_widgets = {}

        self.update_sweep_settings()

        # Dialog buttons (OK and Cancel)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def clear_layout(self, layout):
        while layout.count():
            child = layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

    def update_sweep_settings(self):
        # Clear current sweep settings
        self.clear_layout(self.sweep_settings_layout)
        self.sweep_settings_widgets.clear()

        # Add new settings fields based on selected sweep type
        sweep_type = self.sweep_type_combo.currentText()

        if sweep_type == "Linear Sweep":
            self.add_sweep_setting("Start:", "start", QDoubleValidator())
            self.add_sweep_setting("End:", "end", QDoubleValidator())
            self.add_sweep_setting("Step Size:", "step_size", QDoubleValidator())
        elif sweep_type == "Logarithmic Sweep":
            self.add_sweep_setting("Start:", "start", QDoubleValidator())
            self.add_sweep_setting("End:", "end", QDoubleValidator())
            self.add_sweep_setting("Base:", "base", QDoubleValidator())
        elif sweep_type == "Geometric Sweep":
            self.add_sweep_setting("Start:", "start", QDoubleValidator())
            self.add_sweep_setting("End:", "end", QDoubleValidator())
            self.add_sweep_setting("Factor:", "factor", QDoubleValidator())
        elif sweep_type == "Custom Sequence Sweep":
            self.add_sweep_setting("Custom Sequence (comma-separated):", "custom_sequence", None)

    def add_sweep_setting(self, label_text, setting_key, validator):
        label = QLabel(label_text)
        line_edit = QLineEdit()
        if validator:
            line_edit.setValidator(validator)
        self.sweep_settings_layout.addRow(label, line_edit)
        self.sweep_settings_widgets[setting_key] = line_edit

    def accept(self):

        sweep_type = self.sweep_type_combo.currentText()
        settings = {key: widget.text() for key, widget in self.sweep_settings_widgets.items()}

        try:
            values = self.generate_sweep_values(sweep_type, settings)
            self.result = values
            super().accept()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))

    def generate_sweep_values(self, sweep_type, settings):
        if sweep_type == "Linear Sweep":
            start = float(settings.get("start", 0))
            end = float(settings.get("end", 0))
            step_size = float(settings.get("step_size", 1))
            return list(np.arange(start, end + step_size, step_size))
        elif sweep_type == "Logarithmic Sweep":
            start = float(settings.get("start", 1))
            end = float(settings.get("end", 1))
            base = float(settings.get("base", 10))
            if start <= 0 or end <= 0:
                raise ValueError("Logarithmic Sweep requires positive start and end values.")
            num_points = int(np.log(end / start) / np.log(base)) + 1
            return list(start * np.power(base, np.arange(num_points)))
        elif sweep_type == "Geometric Sweep":
            start = float(settings.get("start", 1))
            end = float(settings.get("end", 1))
            factor = float(settings.get("factor", 2))
            if start <= 0 or end <= 0 or factor <= 0:
                raise ValueError("Geometric Sweep requires positive start, end, and factor values.")
            values = []
            current = start
            while current <= end:
                values.append(current)
                current *= factor
            return values
        elif sweep_type == "Custom Sequence Sweep":
            sequence = settings.get("custom_sequence", "")
            try:
                values = list(map(float, sequence.split(',')))
            except ValueError:
                raise ValueError("Custom Sequence must contain valid floating point numbers separated by commas.")
            return values
        else:
            raise ValueError("Invalid sweep type.")

    def get_result(self):
        return getattr(self, 'result', None)

if __name__ == '__main__':
    app = QApplication(sys.argv)

    # Example event initialization
    seq= creat_test()
    example_event = seq.get_all_events()[0]
    dialog = SweepEventDialog(example_event)
    if dialog.exec_() == QDialog.Accepted:
        result = dialog.get_result()
        if result:
            values = result
            print(f"Sweep Values: {values}")
        else:
            print("No result")
    else:
        print("Dialog canceled")

    sys.exit(app.exec_())
