
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QVBoxLayout, QApplication, QScrollArea
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
   QComboBox, QApplication, QTextEdit,QHBoxLayout,QToolTip,QDialogButtonBox,QGridLayout, QMessageBox,QSizePolicy, QDialog,QLabel,QMenu, QPushButton, QWidget, QVBoxLayout, QScrollArea, QScrollBar,QInputDialog
)
from sequencer.Sequence.channel import Channel
from sequencer.Sequence.time_frame import TimeInstance
from PyQt5.QtWidgets import QDialog

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel
from PyQt5.QtGui import QPainter, QPolygon, QBrush,QDoubleValidator
from sequencer.Sequence.event import RampType



class AnalogEventDialog(QDialog):
    def __init__(self,channal:Channel, time_instance:TimeInstance):
        super().__init__()
        
        self.layout = QVBoxLayout()

        # Channel
        self.channel_label = QLabel("Channel:")
        self.channel_label_value = QLabel(channal.name)
        self.layout.addWidget(self.channel_label)
        self.layout.addWidget(self.channel_label_value)

        # Time instance
        self.time_instance_label = QLabel("Time Instance:")
        self.time_instance_label_value = QLabel(time_instance.name)
        self.layout.addWidget(self.time_instance_label)
        self.layout.addWidget(self.time_instance_label_value)


        self.comment_label = QLabel("Description of the event:")
        self.comment_text = QLineEdit()
        self.layout.addWidget(self.comment_label)
        self.layout.addWidget(self.comment_text)

        # Behavior type
        self.behavior_label = QLabel("Behavior Type:")
        self.behavior_combo = QComboBox()
        self.behavior_combo.addItems(["Jump", "Ramp"])
        self.layout.addWidget(self.behavior_label)
        self.layout.addWidget(self.behavior_combo)

        # Target value for Jump
        self.target_value_label = QLabel("Target Value (for Jump):")
        self.target_value_input = QLineEdit()
        self.target_value_input.setValidator(QDoubleValidator())
        self.layout.addWidget(self.target_value_label)
        self.layout.addWidget(self.target_value_input)

        # Ramp duration for Ramp
        # combo box for time instances after me
        time_instances_after_me = time_instance.get_all_time_instances_after_me()
        self.time_instance_after_me_label = QLabel("End Time Instance")
        self.time_instance_after_me_combo = QComboBox()
        self.time_instance_after_me_combo.addItems([time_instance.name for time_instance in time_instances_after_me])
        self.layout.addWidget(self.time_instance_after_me_label)
        self.layout.addWidget(self.time_instance_after_me_combo)

        self.ramp_type_label = QLabel("Ramp Type (for Ramp):")
        self.ramp_type_combo = QComboBox()
        self.ramp_type_combo.addItems([e.value for e in RampType])
        self.layout.addWidget(self.ramp_type_label)
        self.layout.addWidget(self.ramp_type_combo)

        self.start_value_label = QLabel("Start Value (for Ramp):")
        self.start_value_input = QLineEdit()
        self.start_value_input.setValidator(QDoubleValidator())
        self.layout.addWidget(self.start_value_label)
        self.layout.addWidget(self.start_value_input)

        self.end_value_label = QLabel("End Value (for Ramp):")
        self.end_value_input = QLineEdit()
        self.end_value_input.setValidator(QDoubleValidator())
        self.layout.addWidget(self.end_value_label)
        self.layout.addWidget(self.end_value_input)

        # New: Generic function text editor
        self.generic_function_label = QLabel("Generic Function:")
        self.generic_function_input = QTextEdit()
        self.generic_function_input.setPlaceholderText("Enter your custom function here (e.g., 3*x*cos(t))")
        self.layout.addWidget(self.generic_function_label)
        self.layout.addWidget(self.generic_function_input)

        self.resoltion_label = QLabel("Resolution (for Ramp) in ms:")
        self.resoltion_input = QLineEdit()
        self.resoltion_input.setValidator(QDoubleValidator())
        self.layout.addWidget(self.resoltion_label)
        self.layout.addWidget(self.resoltion_input)

        self.update_behavior_fields()
        self.behavior_combo.currentIndexChanged.connect(self.update_behavior_fields)
        self.ramp_type_combo.currentIndexChanged.connect(self.update_ramp_fields)

        self.setLayout(self.layout)
        self.update_behavior_fields()
        self.update_ramp_fields()
        self.update_behavior_fields()



    def add_ok_cancel_buttons(self):
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)


    def update_behavior_fields(self):
        is_jump = self.behavior_combo.currentText() == "Jump"
        self.target_value_label.setVisible(is_jump)
        self.target_value_input.setVisible(is_jump)
        self.start_value_label.setVisible(not is_jump)
        self.start_value_input.setVisible(not is_jump)
        self.end_value_label.setVisible(not is_jump)
        self.end_value_input.setVisible(not is_jump)
        self.ramp_type_label.setVisible(not is_jump)
        self.ramp_type_combo.setVisible(not is_jump)
        self.resoltion_label.setVisible(not is_jump)
        self.resoltion_input.setVisible(not is_jump)
        self.time_instance_after_me_label.setVisible(not is_jump)
        self.time_instance_after_me_combo.setVisible(not is_jump)
        self.generic_function_label.setVisible(not is_jump and self.ramp_type_combo.currentText() == "Generic")
        self.generic_function_input.setVisible(not is_jump and self.ramp_type_combo.currentText() == "Generic")
        self.adjustSize()

    def update_ramp_fields(self):
        is_generic = self.ramp_type_combo.currentText() == "Generic"
        self.start_value_label.setVisible(not is_generic)
        self.start_value_input.setVisible(not is_generic)
        self.end_value_label.setVisible(not is_generic)
        self.end_value_input.setVisible(not is_generic)
        self.generic_function_label.setVisible(is_generic)
        self.generic_function_input.setVisible(is_generic)
        self.adjustSize()

    def get_behavior(self):
        behavior_type = self.behavior_combo.currentText()
        
        if behavior_type == "Jump":
            jump_target_value = self.target_value_input.text()
            behavior_params = {
                'behavior_type': behavior_type,
                'jump_target_value': float(jump_target_value),
                'ramp_duration': None,
                'ramp_type': None,
                'start_value': None,
                'end_value': None,
                'comment': self.comment_text.text()
            }
        else:
            end_time_instance = self.time_instance_after_me_combo.currentText()
            ramp_type = self.ramp_type_combo.currentText()
            resolution = self.resoltion_input.text()
            
            if ramp_type == "Generic":
                behavior_params = {
                    'behavior_type': behavior_type,
                    'jump_target_value': None,
                    'end_time_instance': end_time_instance,
                    'ramp_type': ramp_type,
                    'generic_function': self.generic_function_input.toPlainText(),
                    'resolution': float(resolution),
                    'comment': self.comment_text.text()
                }
            else:
                start_value = self.start_value_input.text()
                end_value = self.end_value_input.text()
                behavior_params = {
                    'behavior_type': behavior_type,
                    'jump_target_value': None,
                    'end_time_instance': end_time_instance,
                    'ramp_type': ramp_type,
                    'start_value': float(start_value),
                    'end_value': float(end_value),
                    'resolution': float(resolution),
                    'comment': self.comment_text.text()
                }
        
        return behavior_params



class DigitalEventDialog(QDialog):
    def __init__(self,channal:Channel, time_instance:TimeInstance):
        super().__init__()
        
        self.layout = QVBoxLayout()

        # Channel
        self.channel_label = QLabel("Channel:")
        self.channel_label_value = QLabel(channal.name)
        self.layout.addWidget(self.channel_label)
        self.layout.addWidget(self.channel_label_value)

        # Time instance
        self.time_instance_label = QLabel("Time Instance:")
        self.time_instance_label_value = QLabel(time_instance.name)
        self.layout.addWidget(self.time_instance_label)
        self.layout.addWidget(self.time_instance_label_value)
        
        self.comment_label = QLabel("Description of the event:")
        self.comment_text = QLineEdit()
        self.layout.addWidget(self.comment_label)
        self.layout.addWidget(self.comment_text)

        # Behavior type
        # Target value for Jump
        self.target_value_label = QLabel("State:") # high or low from combobox
        self.target_value_combo = QComboBox()
        self.target_value_combo.addItems(["high", "low"])
        self.layout.addWidget(self.target_value_label)
        self.layout.addWidget(self.target_value_combo)
        self.setLayout(self.layout)
    
    def add_ok_cancel_buttons(self):
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, self)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        self.layout.addWidget(self.buttons)
    
    def get_behavior(self):
        behavior_type = "Digital"
        state = self.target_value_combo.currentText()
        behavior_params = {
            'behavior_type': behavior_type,
            'state': 0 if state == "low" else 1,
            'comment': self.comment_text.text()
        }
        return behavior_params
