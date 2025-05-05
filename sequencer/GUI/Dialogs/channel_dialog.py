from PyQt5.QtWidgets import (
    QApplication, QDialog, QVBoxLayout, QLabel, QLineEdit, QPushButton, QComboBox, QFormLayout, QDialogButtonBox
)
from PyQt5.QtGui import QIntValidator, QDoubleValidator
import sys 

from sequencer.Sequence.channel import Analog_Channel, Digital_Channel

from PyQt5.QtWidgets import (
   QComboBox, QApplication, QHBoxLayout,QToolTip,QGridLayout, QMessageBox,QSizePolicy, QDialog,QLabel,QMenu, QPushButton, QWidget, QVBoxLayout, QScrollArea, QScrollBar,QInputDialog
)


# Add this class to the Dialogs/channel_dialog.py file
class ChannelDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Channel')
        self.layout = QFormLayout(self)

        self.type_combo = QComboBox(self)
        self.type_combo.addItems(['Analog', 'Digital'])
        self.type_combo.currentIndexChanged.connect(self.update_form)

        self.name_edit = QLineEdit(self)
        self.card_number_edit = QLineEdit(self)
        self.card_number_edit.setValidator(QIntValidator())
        self.channel_number_edit = QLineEdit(self)
        self.channel_number_edit.setValidator(QIntValidator())
        self.reset_value_edit = QLineEdit(self)
        self.reset_value_edit.setValidator(QDoubleValidator())

        self.layout.addRow('Type:', self.type_combo)
        self.layout.addRow('Name:', self.name_edit)
        self.layout.addRow('Card Number:', self.card_number_edit)
        self.layout.addRow('Channel Number:', self.channel_number_edit)
        self.layout.addRow('Reset Value:', self.reset_value_edit)

        # Analog-specific fields
        self.max_voltage_edit = QLineEdit(self)
        self.max_voltage_edit.setValidator(QDoubleValidator())
        self.min_voltage_edit = QLineEdit(self)
        self.min_voltage_edit.setValidator(QDoubleValidator())
        self.layout.addRow('Max Voltage:', self.max_voltage_edit)
        self.layout.addRow('Min Voltage:', self.min_voltage_edit)


        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
        self.update_form(self.type_combo.currentIndex())

    def update_form(self, index):
        if index == 0:  # Analog
            self.max_voltage_edit.show()
            self.min_voltage_edit.show()
            self.reset_value_edit.show()
        elif index == 1:  # Digital
            self.max_voltage_edit.hide()
            self.min_voltage_edit.hide()
            self.reset_value_edit.hide()

    def get_data(self):
        data = {
            'type': self.type_combo.currentText(),
            'name': self.name_edit.text(),
            'card_number': int(self.card_number_edit.text()),
            'channel_number': int(self.channel_number_edit.text()),
        }
        if data['type'] == 'Analog':
            data['max_voltage'] = float(self.max_voltage_edit.text())
            data['min_voltage'] = float(self.min_voltage_edit.text())
            data['reset_value'] = float(self.reset_value_edit.text())
        return data

class CustomDialog(QDialog):
    def __init__(self, channels, types, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Select Channel and Type")
        
        layout = QVBoxLayout(self)
        
        # Channel selection
        self.channel_label = QLabel("Channels:")
        self.channel_combo = QComboBox()
        self.channel_combo.addItems(channels)
        
        layout.addWidget(self.channel_label)
        layout.addWidget(self.channel_combo)
        
        # Type selection
        self.type_label = QLabel("Type:")
        self.type_combo = QComboBox()
        self.type_combo.addItems(types)

        # add numeric input for resolution and assert it's numberic 
        self.resolution_label = QLabel("Resolution:")
        self.resolution_edit = QLineEdit(self)
        self.resolution_edit.setValidator(QDoubleValidator())
        layout.addWidget(self.resolution_label)
        layout.addWidget(self.resolution_edit)

        
        layout.addWidget(self.type_label)
        layout.addWidget(self.type_combo)
        
        # OK and Cancel buttons
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        
        layout.addWidget(self.button_box)
    
    def get_values(self):
        return self.channel_combo.currentText(), self.type_combo.currentText(), float(self.resolution_edit.text()) if self.resolution_edit.text() else 0.1




# Edit Analog Channel dialog
class Edit_Analog_Channel(QDialog):
    def __init__(self, channel: Analog_Channel, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Edit Analog Channel')
        self.layout = QFormLayout(self)

        self.name_edit = QLineEdit(self)
        self.name_edit.setText(channel.name)
        self.card_number_edit = QLineEdit(self)
        self.card_number_edit.setValidator(QIntValidator())
        self.card_number_edit.setText(str(channel.card_number))
        self.channel_number_edit = QLineEdit(self)
        self.channel_number_edit.setValidator(QIntValidator())
        self.channel_number_edit.setText(str(channel.channel_number))
        self.reset_value_edit = QLineEdit(self)
        self.reset_value_edit.setValidator(QDoubleValidator())
        self.reset_value_edit.setText(str(channel.reset_value))
        self.max_voltage_edit = QLineEdit(self)
        self.max_voltage_edit.setValidator(QDoubleValidator())
        self.max_voltage_edit.setText(str(channel.max_voltage))
        self.min_voltage_edit = QLineEdit(self)
        self.min_voltage_edit.setValidator(QDoubleValidator())
        self.min_voltage_edit.setText(str(channel.min_voltage))

        self.layout.addRow('Name:', self.name_edit)
        self.layout.addRow('Card Number:', self.card_number_edit)
        self.layout.addRow('Channel Number:', self.channel_number_edit)
        self.layout.addRow('Reset Value:', self.reset_value_edit)
        self.layout.addRow('Max Voltage:', self.max_voltage_edit)
        self.layout.addRow('Min Voltage:', self.min_voltage_edit)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def get_data(self):
        return {
            'name': self.name_edit.text(),
            'card_number': int(self.card_number_edit.text()),
            'channel_number': int(self.channel_number_edit.text()),
            'reset_value': float(self.reset_value_edit.text()),
            'max_voltage': float(self.max_voltage_edit.text()),
            'min_voltage': float(self.min_voltage_edit.text())
        }


# Edit Digital Channel dialog
class Edit_Digital_Channel(QDialog):
    def __init__(self, channel: Digital_Channel, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Edit Digital Channel')
        self.layout = QFormLayout(self)

        self.name_edit = QLineEdit(self)
        self.name_edit.setText(channel.name)
        self.card_number_edit = QLineEdit(self)
        self.card_number_edit.setValidator(QIntValidator())
        self.card_number_edit.setText(str(channel.card_number))
        self.channel_number_edit = QLineEdit(self)
        self.channel_number_edit.setValidator(QIntValidator())
        self.channel_number_edit.setText(str(channel.channel_number))

        self.layout.addRow('Name:', self.name_edit)
        self.layout.addRow('Card Number:', self.card_number_edit)
        self.layout.addRow('Channel Number:', self.channel_number_edit)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)

    def get_data(self):
        return {
            'name': self.name_edit.text(),
            'card_number': int(self.card_number_edit.text()),
            'channel_number': int(self.channel_number_edit.text()),
        }

        


if __name__ == "__main__":
    exit()
    app = QApplication(sys.argv)

    # Test ChannelDialog for adding a new channel
    print("Testing ChannelDialog for adding a new channel")
    add_channel_dialog = ChannelDialog()
    if add_channel_dialog.exec_() == QDialog.Accepted:
        add_channel_data = add_channel_dialog.get_data()
        print(add_channel_data)
    
    # Test Edit_Analog_Channel for editing an existing analog channel
    analog_channel = Analog_Channel('AnalogChannel1', 2, 3, False, 1.0, max_voltage=5.0, min_voltage=-5.0)
    print("Testing Edit_Analog_Channel for editing an existing analog channel")
    edit_analog_dialog = Edit_Analog_Channel(analog_channel)
    if edit_analog_dialog.exec_() == QDialog.Accepted:
        edit_analog_data = edit_analog_dialog.get_data()
        print(edit_analog_data)

    # Test Edit_Digital_Channel for editing an existing digital channel
    digital_channel = Digital_Channel('DigitalChannel1', 4, 5, 'Card1', 8, False, 0.0)
    print("Testing Edit_Digital_Channel for editing an existing digital channel")
    edit_digital_dialog = Edit_Digital_Channel(digital_channel)
    if edit_digital_dialog.exec_() == QDialog.Accepted:
        edit_digital_data = edit_digital_dialog.get_data()
        print(edit_digital_data)

    sys.exit(app.exec_())
