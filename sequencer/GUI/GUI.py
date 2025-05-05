
import copy 
from PyQt5.QtWidgets import (
   QComboBox, QApplication, QHBoxLayout,QToolTip,QGridLayout, QMessageBox,QSizePolicy, QDialog,QLabel,QMenu, QPushButton, QWidget, QVBoxLayout, QScrollArea, QScrollBar,QInputDialog,QFrame
)
from PyQt5.QtGui import QDrag

from PyQt5.QtCore import Qt, QRect, pyqtSignal,QPoint

from sequencer.Sequence.sequence import Sequence, SequenceManager
from sequencer.Sequence.event import  Jump, Ramp, RampType, Digital 
from sequencer.Sequence.utils import exp_to_func
from sequencer.Sequence.channel import Analog_Channel, Digital_Channel, Channel
from sequencer.Sequence.time_frame import TimeInstance


import sys
from typing import List, Optional
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QVBoxLayout, QScrollArea, QScrollBar, QMenuBar,QDialogButtonBox,
    QMenu, QAction, QPushButton, QWidget, QLabel, QDialog, QMessageBox, QFileDialog
)

from sequencer.imaging.THORCAM.imaging_software import ThorCamControlWidget

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton, QLabel, QHBoxLayout


import sys
from PyQt5.QtWidgets import QApplication, QLabel, QSizePolicy, QGridLayout, QWidget
from PyQt5.QtGui import QPainter, QPolygon, QBrush,QDoubleValidator
from PyQt5.QtCore import QPoint, Qt, QSize

from PyQt5.QtWidgets import QWidget, QLabel, QGridLayout, QScrollArea, QVBoxLayout, QSizePolicy
from PyQt5.QtGui import QPainter, QBrush, QPolygon
from PyQt5.QtCore import QPoint, Qt, QSize




from sequencer.GUI.Dialogs.channel_dialog import ChannelDialog, Edit_Digital_Channel, Edit_Analog_Channel,CustomDialog
from sequencer.GUI.Dialogs.events_dialogs import AnalogEventDialog, DigitalEventDialog
from sequencer.GUI.Dialogs.sweep_dialog import SweepEventDialog
from sequencer.GUI.Runner_widget import Runner


import numpy as np
color_maps = {
        'blue red': ([0, 0, 255], [255, 255, 255], [255, 0, 0]),  # Dark Blue, White, Dark Red
        'viridis': ([68, 1, 84], [49, 104, 142], [221, 221, 72], [253, 231, 37]),
        'magma': ([0, 0, 4], [80, 18, 123], [221, 52, 151], [253, 231, 37]),
        'plasma': ([13, 8, 135], [156, 23, 158], [237, 121, 83], [240, 249, 33]),
        'inferno': ([0, 0, 4], [87, 16, 110], [224, 88, 77], [252, 234, 37]),
        'cividis': ([0, 32, 77], [57, 111, 114], [159, 183, 145], [253, 231, 37]),
        'blues': ([247, 251, 255], [198, 219, 239], [107, 174, 214], [33, 113, 181]),
        'greens': ([247, 252, 245], [199, 233, 192], [116, 196, 118], [35, 139, 69]),
        'reds': ([255, 245, 240], [254, 224, 210], [252, 146, 114], [222, 45, 38]),
        'purples': ([252, 251, 253], [239, 237, 245], [218, 218, 235], [188, 189, 220]),
        'oranges': ([255, 245, 235], [254, 230, 206], [253, 174, 107], [230, 85, 13]),
        'coolwarm': ([59, 76, 192], [180, 4, 38], [221, 221, 221], [180, 4, 38]),
        'spectral': ([158, 1, 66], [213, 62, 79], [244, 109, 67], [253, 174, 97]),
        'hsv': ([255, 0, 0], [255, 255, 0], [0, 255, 0], [0, 255, 255]),
        'twilight': ([226, 217, 226], [167, 187, 230], [114, 52, 146], [58, 14, 62]),
        'cubehelix': ([0, 0, 0], [87, 81, 124], [132, 186, 140], [201, 202, 88])
    }
def voltage_to_color(voltage, color_scheme='blue red'):
    """
    Convert a voltage between -10 and 10 to a color using the specified color scheme.
    Returns a tuple of (background_color, text_color)
    """
    voltage = max(-10.0, min(10.0, float(voltage)))
    
    # Normalize voltage to [0, 1] range
    normalized = (voltage + 10) / 20.0
    
    
    
    if color_scheme.lower() not in color_maps:
        raise ValueError(f"Unsupported color scheme: {color_scheme}. Supported schemes are: {', '.join(color_maps.keys())}")
    
    if color_scheme.lower() == 'blue red':
        # Special handling for blue red scheme
        if voltage < 0:
            # Dark Blue to White
            intensity = int(255 * ((1 + voltage / 10) ** 0.5))
            bg_color = f'#{intensity:02x}{intensity:02x}ff'
        else:
            # White to Dark Red
            intensity = int(255 * ((1 - voltage / 10) ** 0.5))
            bg_color = f'#ff{intensity:02x}{intensity:02x}'
    else:
        # For other color schemes
        color_points = color_maps[color_scheme.lower()]
        
        # Interpolate colors
        r = np.interp(normalized, [0, 0.33, 0.66, 1], [p[0] for p in color_points])
        g = np.interp(normalized, [0, 0.33, 0.66, 1], [p[1] for p in color_points])
        b = np.interp(normalized, [0, 0.33, 0.66, 1], [p[2] for p in color_points])
        
        # Convert to hex color for background
        bg_color = f'#{int(r):02x}{int(g):02x}{int(b):02x}'
    
    # Calculate perceived brightness
    r, g, b = int(bg_color[1:3], 16), int(bg_color[3:5], 16), int(bg_color[5:7], 16)
    brightness = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    
    # Choose contrasting text color
    text_color = '#ffffff' if brightness < 0.5 else '#000000'
    return (bg_color, text_color)

class ScrollAreaWithShiftScroll(QScrollArea):
    def __init__(self, parent=None):
        super().__init__(parent)
        
    def wheelEvent(self, event):
        if event.modifiers() == Qt.ShiftModifier:
            # Scroll horizontally if Shift is pressed
            delta = event.angleDelta().y()
            scroll_bar = self.horizontalScrollBar()
        else:
            # Scroll vertically by default
            delta = event.angleDelta().y()
            scroll_bar = self.verticalScrollBar()
        
        # Perform the scroll
        scroll_bar.setValue(scroll_bar.value() - delta)

class QArrowWidget(QLabel):
    """Responsible for drawing arrows which show parent-child relationship."""
    def __init__(self, arrow_list, grid, start_pos=(0, 0), parent_widget:'TimeInstanceWidget'=None):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.grid = grid
        self.start_pos = start_pos
        self.grid.addWidget(self, self.start_pos[0], self.start_pos[1], 1, -1)
        self.figureOutHowToDrawArrows(arrow_list)
        # self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)

    def sizeHint(self):
        return QSize(5000, (self.max_height + 1) * 10 + 5)

    def figureOutHowToDrawArrows(self, arrow_list):
        n_cols = 0
        for arrow in arrow_list:
            n_cols = max(max(n_cols, arrow[0]), arrow[1])
        self.n_cols = n_cols + 1

        self.arrow_list = arrow_list

        availability = []
        self.height_list = []
        for arrow in arrow_list:
            left_index = min(arrow[0], arrow[1])
            right_index = max(arrow[0], arrow[1])
            used = [False] * (right_index - left_index)
            for i, avail in enumerate(availability):
                if all(avail[left_index:right_index]):
                    height = i
                    break
            else:
                height = len(availability)
                availability.append([True] * self.n_cols)
            availability[height][left_index:right_index] = used
            self.height_list.append(height)
        self.max_height = len(availability)

    def paintEvent(self, event):
        centers = []
        r = self.grid.cellRect(self.start_pos[0], self.start_pos[1])
        top_left = r.topLeft()
        for i in range(self.n_cols):
            r = self.grid.cellRect(self.start_pos[0], i + self.start_pos[1])
            c = r.bottomRight() - top_left
            centers.append(c)

        qp = QPainter()
        qp.begin(self)
        for arrow, height in zip(self.arrow_list, self.height_list):
            center_left = centers[arrow[0]]
            center_right = centers[arrow[1]]
            
            hp = 10 * height + 5

            def drawArrow(qp, x1, x2, height):
                length = 5
                qp.drawLine(x1, height, x2, height)
                qp.drawLine(x1, height - length, x1, height + length)
                brush = QBrush(Qt.SolidPattern)
                qp.setBrush(brush)
                if x1 > x2:
                    length = -5
                points = [QPoint(x2, height),
                          QPoint(x2 - length, height - length),
                          QPoint(x2 - length, height + length)]
                arrowHead = QPolygon(points)
                qp.drawPolygon(arrowHead)
            
            drawArrow(qp, center_left.x(), center_right.x(), hp)
        qp.end()
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel
from PyQt5.QtGui import QIntValidator

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel, QShortcut
from PyQt5.QtGui import QIntValidator, QKeySequence
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel, QShortcut
from PyQt5.QtGui import QIntValidator, QKeySequence
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLineEdit, QLabel
from PyQt5.QtGui import QIntValidator, QKeyEvent
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QLineEdit, QLabel, QPushButton, QFormLayout, QMessageBox)
from PyQt5.QtGui import QIntValidator

class AddChildTimeInstanceDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Child Time Instance")
        self.initUI()

    def initUI(self):
        layout = QFormLayout()

        self.name_edit = QLineEdit()
        layout.addRow("Name:", self.name_edit)

        self.relative_time_edit = QLineEdit()
        self.relative_time_edit.setValidator(QIntValidator())
        layout.addRow('Relative Time (us):', self.relative_time_edit)

        self.add_button = QPushButton("Add")
        self.add_button.clicked.connect(self.add_child)
        layout.addWidget(self.add_button)

        self.setLayout(layout)

    def add_child(self):
        name = self.name_edit.text()
        relative_time = self.relative_time_edit.text()

        if not name or not relative_time:
            QMessageBox.warning(self, "Input Error", "All fields are required.")
            return

        try:
            relative_time = int(relative_time)
            self.accept()
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Relative time must be an integer.")
            return

        self.name = name
        self.relative_time = relative_time

class EditParentTimeInstanceDialog(QDialog):
    def __init__(self, parent_widget: 'TimeInstanceLabel'):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.all_children = [child.name for child in self.parent_widget.time_instance.get_root().get_all_children() if child.name != self.parent_widget.time_instance.name]
        self.initUI()

    def initUI(self):
        layout = QFormLayout()

        self.parent_time_instance_combo = QComboBox()
        self.parent_time_instance_combo.addItems(self.all_children)
        layout.addRow("Parent Time Instance:", self.parent_time_instance_combo)

        self.edit_button = QPushButton("Edit")
        self.edit_button.clicked.connect(self.edit_parent)
        layout.addWidget(self.edit_button)

        self.setLayout(layout)

    def edit_parent(self):
        parent_time_instance_name = self.parent_time_instance_combo.currentText()

        if not parent_time_instance_name:
            QMessageBox.warning(self, "Input Error", "Parent time instance is required.")
            return

        self.parent_time_instance_name = parent_time_instance_name
        self.accept()



from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLineEdit, QLabel, QMenu, QAction, QDialog, QFormLayout, QPushButton, QMessageBox, QComboBox)
from PyQt5.QtGui import QIntValidator
from PyQt5.QtCore import Qt

class TimeInstanceLabel(QWidget):
    def __init__(self, time_instance: 'TimeInstance', parent_widget: 'TimeInstanceWidget' = None):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.time_instance = time_instance
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Editable name
        self.name_edit = QLineEdit(self.time_instance.name)
        self.name_edit.setContextMenuPolicy(Qt.CustomContextMenu)
        self.name_edit.customContextMenuRequested.connect(self.show_context_menu)
        if self.time_instance.is_sweept:
            self.name_edit.setStyleSheet("background-color: yellow")
        layout.addWidget(self.name_edit)

        # Editable relative time with numeric validation
        self.relative_time_edit = QLineEdit(str(self.time_instance.relative_time))
        self.relative_time_edit.setValidator(QDoubleValidator())
        layout.addWidget(self.relative_time_edit)

        # Non-editable absolute time
        self.absolute_time_label = QLabel(str(self.time_instance.get_absolute_time()))
        layout.addWidget(self.absolute_time_label)

        # Connect signals
        self.name_edit.editingFinished.connect(self.change_name)
        self.relative_time_edit.editingFinished.connect(self.change_relative_time)

        self.setLayout(layout)
    
    def change_name(self):
        try:
            self.time_instance.edit_name(self.name_edit.text())
            self.parent_widget.parent_widget.refresh_time_axis()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            # unconnect 
            self.name_edit.editingFinished.disconnect(self.change_name)
            self.name_edit.setText(self.time_instance.name)
            self.name_edit.editingFinished.connect(self.change_name)
    def change_relative_time(self):
        self.time_instance.edit_relative_time(int(self.relative_time_edit.text()))
        self.parent_widget.parent_widget.refresh_time_axis()
        self.parent_widget.parent_widget.event_table.order_UI()

    def show_context_menu(self, position):
        context_menu = QMenu(self)
        
        add_child_action = QAction("Add Child Time Instance", self)
        add_child_action.triggered.connect(self.add_child_time_instance)
        context_menu.addAction(add_child_action)
        
        edit_parent_action = QAction("Edit Parent Time Instance", self)
        edit_parent_action.triggered.connect(self.edit_parent_time_instance)
        context_menu.addAction(edit_parent_action)
        
        delete_action = QAction("Delete", self)
        delete_action.triggered.connect(self.delete_time_instance)
        context_menu.addAction(delete_action)
        sweep_action = QAction("Sweep", self)
        sweep_action.triggered.connect(self.sweep)
        delete_sweep = QAction("Delete Sweep", self)
        delete_sweep.triggered.connect(self.delete_sweep)
        context_menu.addAction(delete_sweep)
        context_menu.addAction(sweep_action)

        
        context_menu.exec_(self.name_edit.mapToGlobal(position))


    def sweep(self):
        dialog = SweepEventDialog()
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_result()
            self.parent_widget.parent_widget.sequence.stack_sweep_paramter(self.time_instance, data,"relative_time")
            self.parent_widget.parent_widget.refresh_time_axis()

    def delete_sweep(self):
        self.parent_widget.parent_widget.sequence.unstack_sweep_parameter(self.time_instance)
        self.parent_widget.parent_widget.refresh_time_axis()

    def add_child_time_instance(self):
        dialog = AddChildTimeInstanceDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # try:
                self.time_instance.add_child_time_instance(dialog.name, dialog.relative_time)
                self.parent_widget.parent_widget.event_table.add_time_instance(
                    self.time_instance.get_child_time_instance_by_name(dialog.name)
                )
                self.parent_widget.parent_widget.refresh_time_axis()
            # except Exception as e:
            #     QMessageBox.critical(self, "Error", str(e))

    def edit_parent_time_instance(self):
        dialog = EditParentTimeInstanceDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            # try:
                self.parent_widget.parent_widget.sequence.edit_time_instance(self.time_instance,new_parent_name=dialog.parent_time_instance_name)
                self.parent_widget.parent_widget.refresh_time_axis()
            # except Exception as e:
            #     QMessageBox.critical(self, "Error", str(e))

    def delete_time_instance(self):
        self.parent_widget.parent_widget.sequence.delete_time_instance(self.time_instance.name)
        self.parent_widget.parent_widget.event_table.delete_time_instance(self.time_instance)
        self.parent_widget.parent_widget.refresh_time_axis()

        # self.parent_widget.parent_widget.event_table()



class TimeInstanceWidget(QWidget):
    def __init__(self, root_time_instance: 'TimeInstance', parent_widget: 'SequenceViewerWdiget' = None):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.root_time_instance = root_time_instance
        
        self.setup_UI()

    def setup_UI(self):
        # Create a container widget to hold the grid layout
        self.inner_widget = QWidget()
        self.grid = QGridLayout(self.inner_widget)
        self.inner_widget.setLayout(self.grid)

        # Create a QScrollArea and set the container widget as its widget
        self.scroll_area = ScrollAreaWithShiftScroll()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.inner_widget)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)


        # Create a layout for TimeInstanceWidget and add the scroll area
        layout = QVBoxLayout()
        layout.addWidget(self.scroll_area)
        self.setLayout(layout)

        # Clear the grid layout
        for i in reversed(range(self.grid.count())):
            widget = self.grid.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

        # Remove the arrow widget if it exists
        if hasattr(self, 'arrow_widget'):
            self.arrow_widget.deleteLater()

        # Get and sort time instances
        self.time_instances = self.root_time_instance.get_all_time_instances()
        self.time_instances.sort(key=lambda ti: ti.get_absolute_time())

        # Add custom labels to grid for each time instance
        self.labels = {}
        for i, time_instance in enumerate(self.time_instances):
            label_widget = TimeInstanceLabel(time_instance, self)
            

            self.grid.addWidget(label_widget, 0, i)
            self.grid.setColumnMinimumWidth(i,150)
            self.labels[time_instance] = (0, i)

        # Add the arrow widget in row 2
        arrow_list = []
        for time_instance in self.time_instances:
            if time_instance.parent:
                parent_index = self.labels[time_instance.parent][1]
                child_index = self.labels[time_instance][1]
                arrow_list.append((parent_index, child_index))

        self.arrow_widget = QArrowWidget(arrow_list, self.grid, start_pos=(1, 0), parent_widget=self)

from PyQt5.QtWidgets import QPushButton

class ChannelButton(QPushButton):
    def __init__(self, channel:Channel, parent_widget:'ChannelLabelListWidget'=None):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.channel = channel
        self.setText(f'{self.channel.name}')
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: { 'gray'  if isinstance(self.channel, Analog_Channel) else 'black'};
                font-weight: bold;
                color:  { 'white'  if isinstance(self.channel, Analog_Channel) else 'white'};
                padding: 20px;
                border-radius: 5px;
            }}
        """)

    def get_channel_index(self):
        return self.parent_widget.sequence.channels.index(self.channel)
    def enterEvent(self,event):
        text = f"Card Number: {self.channel.card_number}\nChannel Number: {self.channel.channel_number}"
        if isinstance(self.channel, Analog_Channel):
            text += f"\nMax Voltage: {self.channel.max_voltage}\nMin Voltage: {self.channel.min_voltage}"
        QToolTip.showText(event.globalPos(),text , self)
        super().enterEvent(event)

    def delete_channel(self):
        self.parent_widget.sequence.delete_channel(self.channel.name)
        self.parent_widget.refresh_UI() # Flag For fix 
        self.parent_widget.parent_widget.event_table.delete_channel(self.channel)
    def add_channel(self):
        # open the add channel dialog
        dialog = ChannelDialog()
        
        # get current index
        index = self.get_channel_index()

        if dialog.exec_() == QDialog.Accepted:
            
            data= dialog.get_data()
            if data['type'] == 'Analog':
                self.parent_widget.sequence.add_analog_channel(name=data['name'], card_number=data['card_number'], channel_number=data['channel_number'], reset_value=data['reset_value'], max_voltage=data['max_voltage'], min_voltage=data['min_voltage'],index=index+1)
            else:
                self.parent_widget.sequence.add_digital_channel(name=data['name'], card_number=data['card_number'], channel_number=data['channel_number'], index=index+1)
            new_channel = self.parent_widget.sequence.find_channel_by_name(data['name'])
            self.parent_widget.refresh_UI()
            self.parent_widget.parent_widget.event_table.add_channel(new_channel)

    def edit_channel(self):
        # open the edit channel dialog
        if isinstance(self.channel, Analog_Channel):
            dialog = Edit_Analog_Channel(self.channel)
        else:
            dialog = Edit_Digital_Channel(self.channel)
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_data()
            if isinstance(self.channel, Digital_Channel):
                self.parent_widget.sequence.edit_digital_channel(name=self.channel.name,new_name=data['name'], card_number=data['card_number'], channel_number=data['channel_number'])
            else: 
                self.parent_widget.sequence.edit_analog_channel(name=self.channel.name,new_name=data['name'], card_number=data['card_number'], channel_number=data['channel_number'], reset_value=data['reset_value'], max_voltage=data['max_voltage'], min_voltage=data['min_voltage'])
            self.parent_widget.refresh_UI()
    
    def change_order(self,index):
        # open input dialog to get the index which is a drop down list of indexes
        len_channels = len(self.parent_widget.sequence.channels)
        current_index = self.get_channel_index()
        index, ok = QInputDialog.getInt(self, "Change Order", "Enter the new index", value=index, min=0, max=len_channels-1)
        if ok:
            self.parent_widget.sequence.change_channel_order(self.channel, index)
            self.parent_widget.refresh_UI()
            self.parent_widget.parent_widget.event_table.change_channel_order(current_index, index)

    def contextMenuEvent(self, event):
        context_menu = QMenu(self)
        delete_action = context_menu.addAction("Delete Channel")
        delete_action.triggered.connect(self.delete_channel)
        add_action = context_menu.addAction("Add Channel")
        add_action.triggered.connect(self.add_channel)
        edit_action = context_menu.addAction("Edit Channel")
        edit_action.triggered.connect(self.edit_channel)
        change_order_action = context_menu.addAction("Change Order")
        change_order_action.triggered.connect(self.change_order)
        context_menu.exec_(event.globalPos())



from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtCore import Qt

class ChannelLabelListWidget(QWidget):
    def __init__(self, sequence:Sequence, parent_widget:'SequenceViewerWdiget'=None):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.sequence = sequence
        self.buttons = []
        self.setup_UI()
        
    def setup_UI(self):
        # Use QGridLayout instead of QVBoxLayout
        self.layout = QVBoxLayout(self)
        
        self.scroll_area = ScrollAreaWithShiftScroll(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.inner_widget = QWidget()
        self.inner_layout = QGridLayout(self.inner_widget)  # Use QGridLayout here as well

        self.scroll_area.setWidget(self.inner_widget)
        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)
        
        self.refresh_UI()
        
    def refresh_UI(self):
        # Clear existing buttons from the layout
        for button in self.buttons:
            self.inner_layout.removeWidget(button)
            button.deleteLater()
        
        self.buttons = []
        
        # Add buttons to the grid layout
        for row, channel in enumerate(self.sequence.channels):
            button = ChannelButton(channel, self)  # Assuming ChannelButton is a QPushButton
            self.buttons.append(button)
            self.inner_layout.addWidget(button, row, 0,alignment=Qt.AlignBottom)  # Add each button in a new row
            self.inner_layout.setRowMinimumHeight(row,150)
        # Add a button to add a new channel
        if len(self.sequence.channels) < 1:
            button = QPushButton("Add Channel")
            button.clicked.connect(self.add_channel)
            self.inner_layout.addWidget(button, 0, 0,alignment=Qt.AlignTop)  # Add the button in the first row
            self.inner_layout.setRowMinimumHeight(0,150)
            self.buttons.append(button)
        
        self.inner_widget.setLayout(self.inner_layout)
    def add_channel(self):
        # open the add channel dialog
        dialog = ChannelDialog()
        if dialog.exec_() == QDialog.Accepted:
            data= dialog.get_data()
            if data['type'] == 'Analog':
                self.sequence.add_analog_channel(name=data['name'], card_number=data['card_number'], channel_number=data['channel_number'], reset_value=data['reset_value'], max_voltage=data['max_voltage'], min_voltage=data['min_voltage'])
            else:
                self.sequence.add_digital_channel(name=data['name'], card_number=data['card_number'], channel_number=data['channel_number'])
            new_channel = self.sequence.find_channel_by_name(data['name'])
            self.refresh_UI()
            self.parent_widget.event_table.add_channel(new_channel)

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QPushButton, QLineEdit, QComboBox, QDoubleSpinBox, QDialog

class QComboBox_NO_mouse(QComboBox):
    def __init__(self) -> None:
        super().__init__()

    def wheelEvent(self, event):
        event.ignore()

class QDouble_NO_mouse_SpinBox(QDoubleSpinBox):
    def __init__(self, text_view="") -> None:
        super().__init__()
        self.text_view = text_view
        self.setDecimals(4)

    def stepBy(self, steps):
        cursor_position = self.lineEdit().cursorPosition()
        # number of characters before the decimal separator including +/- sign
        n_chars_before_sep = len(str(abs(int(self.value())))) + 1
        if cursor_position == 0:
            # set the cursor right of the +/- sign
            self.lineEdit().setCursorPosition(1)
            cursor_position = self.lineEdit().cursorPosition()
        single_step = 10 ** (n_chars_before_sep - cursor_position)
        # Handle decimal separator. Step should be 0.1 if cursor is at `1.|23` or
        # `1.2|3`.
        if cursor_position >= n_chars_before_sep + 2:
            single_step = 10 * single_step
        # Change single step and perform the step
        self.setSingleStep(single_step)
        super().stepBy(steps)
        # Undo selection of the whole text.
        self.lineEdit().deselect()
        # Handle cases where the number of characters before the decimal separator
        # changes. Step size should remain the same.
        new_n_chars_before_sep = len(str(abs(int(self.value())))) + 1
        if new_n_chars_before_sep < n_chars_before_sep:
            cursor_position -= 1
        elif new_n_chars_before_sep > n_chars_before_sep:
            cursor_position += 1
        self.lineEdit().setCursorPosition(cursor_position)


    def wheelEvent(self, event):
        event.ignore()

    def enterEvent(self,event):
        QToolTip.showText(event.globalPos(),self.text_view , self)
        super().enterEvent(event)

class EventButton(QWidget):
    def __init__(self, channel: Channel, time_instance: TimeInstance, parent_widget: 'EventsWidget',color_scheme='blue red'):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.channel = channel
        self.time_instance = time_instance
        self.color_scheme = color_scheme
        # Initialize the layout
        self.layout_button = QVBoxLayout(self)
        self.value = None
        

        # Initial call to set up the UI
        self.refresh_UI()
        self.setLayout(self.layout_button)
        
        
    
    def update_color(self):
        # loop through the layout and update the color of the widgets
        for i in range(self.layout_button.count()-1):
            widget = self.layout_button.itemAt(i).widget()
            if widget is not None:
                # exclude the last widget which is a horizontal line
                if isinstance(self.channel, Digital_Channel):
                    if self.value == 1:
                        self.refresh_color(10,widget)
                    else:  
                        self.refresh_color(0,widget)
                else:
                    self.refresh_color(self.value,widget)
                # prind widget name 




    def get_col(self):
        layout = self.parent_widget.inner_layout
        for col in range(layout.columnCount()):
            try:
                if layout.itemAtPosition(0, col).widget().time_instance.name == self.time_instance.name:
                    return col 
            except Exception as e:
                pass
        return None

    def get_row(self):
        layout = self.parent_widget.inner_layout
        for row in range(layout.rowCount()):
            try:
                if layout.itemAtPosition(row, 0).widget().channel.name == self.channel.name:
                    return row 
            except Exception as e:
                pass 
        return None
    def clear_layout(self):
        for i in reversed(range(self.layout_button.count())):
            widget = self.layout_button.itemAt(i).widget()
            if widget is not None:
                widget.deleteLater()

    def refresh_UI(self):
        # Clear the current layout
        self.clear_layout()
        self.paramter_label = None
        # Check if the time instance contains any events in the channel
        ramp_value = self.channel.detect_a_ramp(self.time_instance)
        if ramp_value is not None:
            ramp = ramp_value[0]
            self.value = ramp_value[1]
            # Add a label to display the value
            value_label = QLabel(f'{ramp.behavior.ramp_type}: {self.value:.3f}')
            self.layout_button.addWidget(value_label)


            self.no_event = True
        
        else:

            event_time = self.channel.get_event_by_time_instance(self.time_instance)
            if event_time is None:
                all_events = self.channel.events
                all_events.sort(key=lambda event: event.start_time_instance.get_absolute_time())
                for i, event in enumerate(all_events):
                    if event.start_time_instance.get_absolute_time() > self.time_instance.get_absolute_time():
                        break
                else:
                    i = len(all_events)
                if i == 0:
                    self.value = self.channel.reset_value
                else:
                    self.value = all_events[i-1].behavior.get_value_at_time(self.time_instance.get_absolute_time())
                if isinstance(self.channel, Digital_Channel):
                    value_button = QPushButton(f'Value: {"High" if self.value == 1 else "Low"}')
                else:
                    value_button = QPushButton(f'Value: {self.value:.3f}')
                value_button.clicked.connect(self.add_event)
                # Add a button to display the value and connect to the create event when clicked 
                self.layout_button.addWidget(value_button)

                self.no_event = True
            else:
                self.no_event = False
                event = event_time[0]
                self.event_instance = event
                time_ref = event_time[1]
                if isinstance(event.behavior, Ramp):
                    if time_ref == 'start':
                        self.value = event.behavior.get_start_value()
                        # Create a vertical layout with a label and numeric value spin box
                        self.paramter_label= "start_value"                        
                        # add a combobox to select the ramp type
                        ramp_combo = QComboBox_NO_mouse()
                        ramp_combo.addItems([e.value for e in RampType])
                        preselected_ramp_type = event.behavior.ramp_type.value
                        ramp_combo.setCurrentText(preselected_ramp_type)
                        ramp_combo.currentIndexChanged.connect(lambda index: self.edit_ramp_type(event))
                        self.layout_button.addWidget(ramp_combo)
                        value_label = QLabel('Start:')
                        value_spinbox = QDouble_NO_mouse_SpinBox(text_view=self.event_instance.comment)
                        # make the range of the spin box between the min and max voltage of the channel
                        value_spinbox.setRange(self.channel.min_voltage, self.channel.max_voltage)
                        value_spinbox.setValue(self.value)

                        value_spinbox.valueChanged.connect(lambda val: self.edit_start_value(event, val))
                        if event.behavior.ramp_type == RampType.GENERIC:
                            value_spinbox.setEnabled(False)

                        self.layout_button.addWidget(value_label)
                        self.layout_button.addWidget(value_spinbox)
                        value_spinbox.valueChanged.connect(self.update_color)


                    else:
                        self.value = event.behavior.get_end_value()
                        self.paramter_label= "end_value" 


                        # Create a vertical layout with a label and numeric value spin box
                        
                        value_label = QLabel('End:')
                        value_spinbox = QDouble_NO_mouse_SpinBox(text_view=self.event_instance.comment)
                        # make the range of the spin box between the min and max voltage of the channel
                        value_spinbox.setRange(self.channel.min_voltage, self.channel.max_voltage)    
                        value_spinbox.setValue(self.value)
                        value_spinbox.valueChanged.connect(lambda val: self.edit_end_value(event, val))
                        if event.behavior.ramp_type == RampType.GENERIC:
                            value_spinbox.setEnabled(False)
                        self.layout_button.addWidget(value_label)
                        self.layout_button.addWidget(value_spinbox)
                        value_spinbox.valueChanged.connect(self.update_color)


                        # if value > event.channel.max_voltage or value < event.channel.min_voltage:
                        #     QMessageBox.warning(self, "Warning", f"Generic Ramp will result in a value outside the channel range, please fix the ramp, the value will be set to the edge of the voltage range {value_spinbox.text()}")

                elif isinstance(event.behavior, Jump):
                    self.value = event.behavior.target_value
                    self.paramter_label= "target_value"
                    # Create a vertical layout with a label and numeric value spin box
                    
                    value_label = QLabel('Jump:')
                    value_spinbox = QDouble_NO_mouse_SpinBox(self.event_instance.comment)
                    value_spinbox.setRange(self.channel.min_voltage, self.channel.max_voltage)
                    value_spinbox.setValue(self.value)
                    value_spinbox.valueChanged.connect(lambda val: self.edit_target_value(event, val))
                    self.layout_button.addWidget(value_label)
                    self.layout_button.addWidget(value_spinbox)
                    value_spinbox.valueChanged.connect(self.update_color)

                elif isinstance(event.behavior, Digital):
                    self.event_instance = event
                    self.paramter_label= "digital_state"
                    
                    self.value = event.behavior.target_value
                    combo_box = QComboBox_NO_mouse()
                    combo_box.addItems([ 'Low','High'])
                    combo_box.setCurrentIndex(1 if self.value == 1 else 0)
                    self.layout_button.addWidget(combo_box)
                    # Create a combo box to display the value and connect to the target value (On =1 , Off = 0)
                    # Connect it to edit target value
                    combo_box.currentIndexChanged.connect(lambda index: self.edit_digtial_state(event))
                    combo_box.currentIndexChanged.connect(self.update_color)

                # Connect to delete event when right clicked
                self.setContextMenuPolicy(Qt.CustomContextMenu)
                self.customContextMenuRequested.connect(self.context_menu)
                # Add a horizontal line at the bottom
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.layout_button.addWidget(line)
        self.update_color()


    def enterEvent(self,event):
        if not self.no_event:
            if isinstance(self.event_instance.behavior, Ramp):
                if self.event_instance.behavior.ramp_type == RampType.GENERIC:
                    func = self.event_instance.behavior.func_text
                    text = self.event_instance.comment +"\nfunc: "+func
                else:
                    text= self.event_instance.comment

            else:
                text = self.event_instance.comment

        else:
            text = "Holding Value: click to add an event"
        QToolTip.showText(event.globalPos(),text , self)
        super().enterEvent(event)

    def context_menu(self, position):
        context_menu = QMenu(self)
        delete_action = context_menu.addAction("Delete Event")
        delete_action.triggered.connect(self.delete_event)
        sweep_action = context_menu.addAction("Sweep")
        sweep_action.triggered.connect(self.sweep)
        delete_sweep_action = context_menu.addAction("Delete Sweep")
        delete_sweep_action.triggered.connect(self.delete_sweep)
        context_menu.exec_(self.mapToGlobal(position))
    
    def delete_sweep(self):
        self.parent_widget.sequence.unstack_sweep_parameter(self.event_instance)
        self.refresh_UI()
        self.refresh_row_after_me()
        self.refresh_row_before_me()


    def sweep(self):
        # open the sweep dialog
        if isinstance(self.channel, Digital_Channel):
            self.parent_widget.sequence.stack_sweep_paramter(self.event_instance,[0,1],parameter=self.paramter_label)
            self.refresh_UI()
            return
        dialog = SweepEventDialog()
        if dialog.exec_() == QDialog.Accepted:
            data = dialog.get_result()
            self.parent_widget.sequence.stack_sweep_paramter(self.event_instance,data,parameter=self.paramter_label)
            self.refresh_UI()
            
    def edit_start_value(self, event, value):
        self.parent_widget.sequence.edit_event_behavior(edited_event=event, start_value=value)
        self.value = value
        self.update_color()
        self.refresh_row_after_me()
        # Make a deep copy of the event behavior

    def edit_end_value(self, event, value):
        self.parent_widget.sequence.edit_event_behavior(edited_event=event, end_value=value)
        self.value = value
        self.update_color()
        self.refresh_row_after_me()
        self.refresh_row_before_me()


    def edit_target_value(self, event, value):
        self.parent_widget.sequence.edit_event_behavior(edited_event=event, target_value=value)
        self.value = value
        self.update_color()
        self.refresh_row_after_me()
        # Make a deep copy of the event behavior
    def edit_digtial_state(self, event):
        if self.sender().currentText() == 'High':
            value = 1
        else:
            value = 0
        self.parent_widget.sequence.edit_event_behavior(edited_event=event, target_value=value)
        self.value = value
        self.update_color()
        self.refresh_row_after_me()
        # Make a deep copy of the event behavior
    def edit_ramp_type(self,event):
        try:
            value = RampType(self.sender().currentText())
            if value == RampType.GENERIC:
                # open a dialog to get the function
                func_text, ok = QInputDialog.getText(self, "Generic Function", "Enter the generic function as a function of time (t):")
                if ok:
                    self.parent_widget.sequence.edit_event_behavior(edited_event=event, ramp_type=value, func_text=func_text)
                    self.refresh_row_after_me()
                    self.refresh_UI()
            else:
                self.parent_widget.sequence.edit_event_behavior(edited_event=event, ramp_type=value)
                self.refresh_row_after_me()
                self.refresh_UI()            
            
        except Exception as e:
            self.sender().setCurrentText(event.behavior.ramp_type.value)
            QMessageBox.critical(self, "Error", f"An error occurred while editing the ramp type: {str(e)}")
            

        # Make a deep copy of the event behavior

    def add_event(self):
        # Add an event to the time instance and channel 
        # try: 
            if isinstance(self.channel, Analog_Channel):
                dialog = AnalogEventDialog(self.channel, self.time_instance)
            else:
                dialog = DigitalEventDialog(self.channel, self.time_instance)
            dialog.add_ok_cancel_buttons()
            if dialog.exec_() == QDialog.Accepted:
                behavior = dialog.get_behavior()
                if isinstance(self.channel, Analog_Channel):
                    if behavior['behavior_type'] == 'Ramp':
                        end_time_instance_name = behavior['end_time_instance']
                        end_time_instance = self.parent_widget.sequence.find_TimeInstance_by_name(end_time_instance_name)
                        ramp_type =RampType(behavior['ramp_type']) 
                        if ramp_type == RampType. GENERIC:
                            behavior_instance = Ramp(self.time_instance,end_time_instance, ramp_type,func= exp_to_func( behavior['generic_function']),resolution=behavior['resolution'])
                        else: 
                            behavior_instance = Ramp(self.time_instance,end_time_instance, ramp_type, behavior['start_value'], behavior['end_value'], resolution=behavior['resolution'])
                        self.parent_widget.sequence.add_event(self.channel.name,behavior_instance, self.time_instance, end_time_instance=end_time_instance,comment= behavior['comment'])
                    elif behavior['behavior_type'] == 'Jump':
                        behavior_instance = Jump(behavior['jump_target_value'])
                        self.parent_widget.sequence.add_event(self.channel.name,behavior_instance, self.time_instance,comment= behavior['comment'])
                else:
                    behavior_instance = Digital(behavior['state'])
                    self.parent_widget.sequence.add_event(self.channel.name,behavior_instance, self.time_instance,comment= behavior['comment'])
                self.refresh_UI()
            self.refresh_row_after_me()
        # except Exception as e:
        #         QMessageBox.critical(self, "Error", f"An error occurred while adding the event: {str(e)}")

    def delete_event(self):
        # Delete the event from the time instance and channel
        self.parent_widget.sequence.delete_event(self.time_instance.name,self.channel.name)
        self.refresh_UI()
        self.refresh_row_after_me()

    def refresh_color(self, value, widget):
        try:
            color,text_color = voltage_to_color(value,self.color_scheme)
            
            common_style = f"""
                background-color: {color};
                font-weight: bold;
                color: {text_color};
                padding: 3px;
                border-radius: 5px;
            """

            specific_styles = {
                "QComboBox": f"""
                """,
                "QDoubleSpinBox": """
                    padding: 3px;
                    border: 1px solid #555;
                    border-radius: 3px;
                """,
                "QPushButton": """
                    border: 2px solid gray;
                """,

            }
            try:
                if self.event_instance.is_sweept:
                    specific_styles["QLabel"] = """
                            border: 3px solid black;
                        """
                    specific_styles['QComboBox_NO_mouse'] = """
                            border: 3px solid black; 
                        """
                else:
                    specific_styles["QLabel"] = """
                            border: 1px solid gray;
                        """
                    specific_styles['QComboBox_NO_mouse'] = """
                            border: 1px solid gray; 
                        """
            except:
                pass
            widget_type = widget.__class__.__name__
            style = common_style + specific_styles.get(widget_type, "")
            
            widget.setStyleSheet(style)
        except ValueError:
            print(f"Invalid value: {value}")
        except AttributeError:
            print(f"Unsupported widget type: {type(widget)}")

    def refresh_row_after_me(self):
        row = self.get_row()
        col = self.get_col()
        # loop over the columns and refresh the UI
        for i in range(col+1, self.parent_widget.inner_layout.columnCount()):
            item = self.parent_widget.inner_layout.itemAtPosition(row, i)
            if item is not None:
                try:
                    widget = item.widget()
                    widget.refresh_UI() 

                except Exception as e:
                    pass
    def refresh_row_before_me(self):
        row = self.get_row()
        col = self.get_col()
        # loop over the columns and refresh the UI
        for i in range(col-1, 0, -1):
            item = self.parent_widget.inner_layout.itemAtPosition(row, i)
            if item is not None:
                try:
                    widget = item.widget()
                    widget.refresh_UI()
                except Exception as e:
                    pass

from typing import Dict, Tuple
class EventsWidget(QWidget):
    def __init__(self, sequence: Sequence, parent_widget: 'SequenceViewerWdiget' = None):
        super().__init__(parent_widget)
        self.parent_widget = parent_widget
        self.sequence = sequence
        self.time_instances = []
        self.setup_UI()

    def setup_UI(self):
        self.layout = QGridLayout(self)
        
        self.scroll_area = ScrollAreaWithShiftScroll(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)

        self.inner_widget = QWidget()
        self.inner_layout = QGridLayout(self.inner_widget)
        
        self.time_instances = self.sequence.root_time_instance.get_all_time_instances()
        self.time_instances.sort(key=lambda ti: ti.get_absolute_time())

        channels = self.sequence.channels
        fixed_width = 150
        fixed_height = 150
        # Calculate the maximum width and height required for the buttons
        for row, channel in enumerate(channels):
            for col, time in enumerate(self.time_instances):
                button = EventButton(channel, time, self)
                self.inner_layout.addWidget(button, row , col,alignment=Qt.AlignBottom )
                self.inner_layout.setColumnMinimumWidth(col,fixed_width)
                self.inner_layout.setRowMinimumHeight(row,fixed_height)

        self.inner_widget.setLayout(self.inner_layout)
        self.scroll_area.setWidget(self.inner_widget)
        self.layout.addWidget(self.scroll_area)
        self.setLayout(self.layout)

    def get_row(self, channel: Channel) -> int:
        for row in range(self.inner_layout.rowCount()):
            try:
                if self.inner_layout.itemAtPosition(row, 0).widget().channel == channel:
                    return row
            except Exception as e:
                pass
        return None
    def get_col(self, time_instance: TimeInstance) -> int:
        for col in range(self.inner_layout.columnCount()): 
            try:
                if self.inner_layout.itemAtPosition(0, col).widget().time_instance == time_instance:
                    return col
            except Exception as e:
                pass
        return None
    def get_shape(self):
        # itereate over the layout and get the shape of the grid
        max_row = 0
        max_col = 0
        for row in range(self.inner_layout.rowCount()):
            if self.inner_layout.itemAtPosition(row, 0) is not None:
                max_row = row+1
        for col in range(self.inner_layout.columnCount()):
            if self.inner_layout.itemAtPosition(0, col) is not None:
                max_col = col+1
            
        return max_row, max_col
    
    def change_channel_order(self, current_index: int, new_index: int):
        n_rows, n_cols = self.get_shape()
        # Ensure indices are within bounds
        current_index = max(0, min(current_index, n_rows - 1))
        new_index = max(0, min(new_index, n_rows - 1))
        
        if current_index == new_index:
            return  # No change needed
        
        # Remove widgets from the current row
        widgets_to_move = []
        for col in range(n_cols):
            item = self.inner_layout.itemAtPosition(current_index, col)
            if item and item.widget():
                widget = item.widget()
                self.inner_layout.removeWidget(widget)
                widgets_to_move.append(widget)
        
        # Shift other widgets
        step = 1 if new_index > current_index else -1
        for row in range(current_index, new_index, step):
            for col in range(n_cols):
                next_row = row + step
                item = self.inner_layout.itemAtPosition(next_row, col)
                if item and item.widget():
                    widget = item.widget()
                    self.inner_layout.removeWidget(widget)
                    self.inner_layout.addWidget(widget, row, col,alignment=Qt.AlignBottom)
        
        # Place moved widgets in their new position
        for col, widget in enumerate(widgets_to_move):
            self.inner_layout.addWidget(widget, new_index, col,alignment=Qt.AlignBottom)
        
        # Update any necessary data structures to reflect the new order
        # (This part depends on how you're storing channel order information)                
    
    def add_channel(self, channel: Channel):
        # check for the index of the channel
        index = self.sequence.channels.index(channel)
        # add the channel to the layout by shifting the rows down
        n_rows, n_cols = self.get_shape()
        for i in range(n_rows, index-1, -1):
            for j in range(n_cols):
                item = self.inner_layout.itemAtPosition(i, j)
                if item is not None:
                    widget = item.widget()
                    self.inner_layout.removeWidget(widget)
                    self.inner_layout.addWidget(widget, i+1, j,alignment=Qt.AlignBottom)
        # add the channel to the layout by looping over the time instances
        for i, time_instance in enumerate(self.sequence.root_time_instance.get_all_time_instances()):
            button = EventButton(channel, time_instance, self)
            self.inner_layout.addWidget(button, index, i, alignment=Qt.AlignBottom)


    
    def delete_channel(self, channel: Channel):
        # get the row of the channel 
        row = self.get_row(channel=channel)
        if row is not None:
            for i in range(self.inner_layout.columnCount()):
                item = self.inner_layout.itemAtPosition(row, i)
                if item is not None:
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
                        self.inner_layout.removeWidget(widget)
                        
            # Shift remaining widgets up
            for r in range(row + 1, self.inner_layout.rowCount()):
                for c in range(self.inner_layout.columnCount()):
                    item = self.inner_layout.itemAtPosition(r, c)
                    if item is not None:
                        widget = item.widget()
                        self.inner_layout.removeWidget(widget)
                        self.inner_layout.addWidget(widget, r - 1, c,alignment=Qt.AlignBottom)
    def delete_time_instance(self, time_instance):
        # get the column of the time instance 
        col = self.get_col(time_instance=time_instance)
        print(col)
        if col is not None:
            for i in range(self.inner_layout.rowCount()):
                item = self.inner_layout.itemAtPosition(i, col)
                if item is not None:
                    widget = item.widget()
                    if widget is not None:
                        widget.clear_layout()
                        widget.deleteLater()
                        self.inner_layout.removeWidget(widget)
                        
            # Shift remaining widgets left
            for c in range(col + 1, self.inner_layout.columnCount()):
                for r in range(self.inner_layout.rowCount()):
                    item = self.inner_layout.itemAtPosition(r, c)
                    if item is not None:
                        widget = item.widget()
                        self.inner_layout.removeWidget(widget)
                        self.inner_layout.addWidget(widget, r, c - 1,alignment=Qt.AlignBottom)
                        widget.refresh_UI()
        self.order_UI()
    
    
    def add_time_instance(self, time_instance: TimeInstance):
        # figure out the right place to add the time instance
        n_rows,n_cols= self.get_shape()
        current_time_instances = [self.inner_layout.itemAtPosition(0, i).widget().time_instance for i in range(n_cols)]
        # look for the right place to add the time instance
        for i, current_time_instance in enumerate(current_time_instances):
            if time_instance.get_absolute_time() < current_time_instance.get_absolute_time():
                break
        else:
            i = n_cols
        
        #  shift the columns to the right if needed
        if i < n_cols:
            for j in range(n_rows):
                for k in range(n_cols, i-1, -1):
                    item = self.inner_layout.itemAtPosition(j, k)
                    if item is not None:
                        widget = item.widget()
                        self.inner_layout.removeWidget(widget)
                        self.inner_layout.addWidget(widget, j, k + 1,alignment=Qt.AlignBottom)
        # add a dummy button for the new time instance

        # add the time instance for each channel
        for j, channel in enumerate(self.sequence.channels):
            button = EventButton(channel, time_instance, self)
            self.inner_layout.addWidget(button, j , i,alignment=Qt.AlignBottom)
        
        # self.order_UI()
            
    def order_UI(self):
        # get the shape of the grid
        try:
            n_row, n_col = self.get_shape()
            # get the time instances
            current_time_instances = [self.inner_layout.itemAtPosition(0, i).widget().time_instance for i in range(0, n_col+1)]
            target_time_instances = self.sequence.root_time_instance.get_all_time_instances()
            target_time_instances.sort(key=lambda ti: ti.get_absolute_time())
            # make a dictionary of column current index and target index
            # iterate over the layout and reorder the time instances
            for i in range(0, n_col):
                current_time_instance = self.inner_layout.itemAtPosition(0, i).widget().time_instance
                target_time_instance = target_time_instances[i]
                if current_time_instance != target_time_instance:
                    # get the index of the target time instance
                    target_index = current_time_instances.index(target_time_instance)
                    # shift the columns to the right if needed
                    # put the target time instance widget in a list 
                    list_of_widgets = []
                    for j in range(n_row):
                        item = self.inner_layout.itemAtPosition(j, target_index)
                        if item is not None:
                            widget = item.widget()
                            if widget.no_event:
                                widget.refresh_UI()

                            list_of_widgets.append(widget)
                            self.inner_layout.removeWidget(widget)

                    for j in range(n_row):
                        for k in range(i, target_index, 1):
                            item = self.inner_layout.itemAtPosition(j, k)
                            if item is not None:
                                widget = item.widget()
                                if widget.no_event:
                                    widget.refresh_UI()
                                self.inner_layout.removeWidget(widget)
                                self.inner_layout.addWidget(widget, j, k + 1,alignment=Qt.AlignBottom)
                    # reorder the time instance for each channel
                    
                    for j, widget in enumerate(list_of_widgets):
                        self.inner_layout.addWidget(widget, j, i,alignment=Qt.AlignBottom)
        except Exception as e:
            print(e)

class SequenceViewerWdiget(QWidget):
    def __init__(self, sequence:Sequence=None, parent_widget:'SequenceManagerWidget'=None):
        super().__init__()
        self.parent_widget = parent_widget
        if sequence is None:
            sequence = Sequence("unnamed")
        self.sequence = sequence
        self.syncing = False  # Flag to prevent multiple updates
        self.layout_main = QGridLayout()
        
        self.setLayout(self.layout_main)


        self.setup_ui()

    
        
    def setup_ui(self) -> None:
        

        self.combo_box_type_color = QComboBox_NO_mouse()
        self.combo_box_type_color.addItems(list(color_maps.keys()))
        self.combo_box_type_color.currentIndexChanged.connect(self.refresh_events_color)
        self.refresh_events_color()
        # Create and configure widgets
        self.channel_list = ChannelLabelListWidget(self.sequence,parent_widget=self)
        self.event_table = EventsWidget(self.sequence, parent_widget=self)

        self.time_axis = TimeInstanceWidget(self.sequence.root_time_instance, parent_widget=self)

        


        self.scroll_bar1 = self.channel_list.scroll_area.verticalScrollBar()
        self.scroll_bar2 = self.event_table.scroll_area.verticalScrollBar()
        self.scroll_bar1.valueChanged.connect(self.sync_scroll)
        self.scroll_bar2.valueChanged.connect(self.sync_scroll)

        self.scroll_bar3 = self.event_table.scroll_area.horizontalScrollBar()
        self.scroll_bar4 = self.time_axis.scroll_area.horizontalScrollBar()
        self.scroll_bar3.valueChanged.connect(self.sync_scroll_vertical)
        self.scroll_bar4.valueChanged.connect(self.sync_scroll_vertical)


        

        self.layout_main.addWidget(self.combo_box_type_color, 0, 0)
        self.layout_main.addWidget(self.time_axis, 0, 1)  # Top-right slot
        # make the time axis stretch to take up the available space
        self.layout_main.setColumnStretch(1, 1)
        self.layout_main.addWidget(self.channel_list, 1, 0)  # Bottom-left slot
        self.layout_main.addWidget(self.event_table, 1, 1)  # Bottom-right slot
        self.layout_main.setColumnStretch(1, 1)  # Column 0 will take up 1 part of the available space
        self.layout_main.setRowStretch(1, 1)  # Row 1 will take up 1 part of the available space
    
    def refresh_time_axis(self):
        # remove the current time axis
        # remove the current time axis
        self.layout_main.removeWidget(self.time_axis)
        self.time_axis.deleteLater()

        self.time_axis= TimeInstanceWidget(self.sequence.root_time_instance, parent_widget=self)
        self.layout_main.addWidget(self.time_axis, 0, 1)  # Top-right slot
        

    def refresh_channels(self):
        self.channel_list.refresh_UI()
        self.time_axis.refresh_UI()

    def refresh_events_color(self):
        # refresh self color
        color,text_color = voltage_to_color(7,self.combo_box_type_color.currentText())
        
        style = f"""
            background-color: {color};
            font-weight: bold;
            color: {text_color};
            padding: 3px;
            border-radius: 5px;
            QComboBox {{
                border: 1px solid gray;
                padding: 1px 18px 1px 3px;
                min-width: 6em;
            }}
            QComboBox::drop-down {{
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: darkgray;
                border-left-style: solid;
            }}
            QComboBox QAbstractItemView {{
                border: 2px solid darkgray;
                selection-background-color: {color};
                color: {text_color};
            }}
            """ 
        
        self.combo_box_type_color.setStyleSheet(style)
        try:
            # loop over the layout and update the color of the widgets
            for i in range(self.event_table.inner_layout.rowCount()):
                for j in range(self.event_table.inner_layout.columnCount()):
                    item = self.event_table.inner_layout.itemAtPosition(i, j)
                    if item is not None:
                        widget = item.widget()
                        if widget is not None:
                            widget.color_scheme = self.combo_box_type_color.currentText()
                            widget.update_color()
        except Exception as e : 
            pass

    def sync_scroll_vertical(self, value: int) -> None:
        if self.syncing:
            return
        self.syncing = True
        sender = self.sender()
        if sender == self.scroll_bar3:
            self.scroll_bar4.setValue(self.calculate_proportion(value, self.scroll_bar3, self.scroll_bar4))
        else:
            self.scroll_bar3.setValue(self.calculate_proportion(value, self.scroll_bar4, self.scroll_bar3))
        self.syncing = False

    def sync_scroll(self, value: int) -> None:
        if self.syncing:
            return
        self.syncing = True
        sender = self.sender()
        if sender == self.scroll_bar1:
            self.scroll_bar2.setValue(self.calculate_proportion(value, self.scroll_bar1, self.scroll_bar2))
        else:
            self.scroll_bar1.setValue(self.calculate_proportion(value, self.scroll_bar2, self.scroll_bar1))
        self.syncing = False

    def calculate_proportion(self, value: int, scroll_bar_from: QScrollBar, scroll_bar_to: QScrollBar) -> int:
        proportion = value / scroll_bar_from.maximum() if scroll_bar_from.maximum() != 0 else 0
        return int(proportion * scroll_bar_to.maximum())


    def refresh_UI(self):
        layout = self.layout()
        for i in reversed(range(layout.count())):
            widget_to_remove = layout.itemAt(i).widget()
            layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)
        self.setup_ui()
from PyQt5.QtCore import Qt,QMimeData

class DraggableButton(QPushButton):
    save_sequence_signal = pyqtSignal(str)  # Signal to emit a number
    save_sequence_signal_csv = pyqtSignal(str)  # Signal to emit a number
    delete_sequence_signal = pyqtSignal(str)  # Signal to emit a number
    edit_sequence_signal = pyqtSignal(str)  # Signal to emit a number
    plot_sequence_signal = pyqtSignal(str)  # Signal to emit a numbero
    print_sequence_signal = pyqtSignal(str)  # Signal to emit a number


    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.text_to_emit = text 
        self.parent = parent
        self.setAcceptDrops(True)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)

    def mouseMoveEvent(self, event):
        if event.buttons() != Qt.LeftButton:
            return
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.text())
        drag.setMimeData(mime_data)
        drag.exec_(Qt.MoveAction)

    def dragEnterEvent(self, event):
        event.accept()

    def dropEvent(self, event):
        if event.source() != self:
            self.parent.move_button(self, event.source())

    def show_context_menu(self, position):
        context_menu = QMenu(self)
        save_sequence_action = QAction('Save Sequence', self)
        save_sequence_action.triggered.connect(self.save_sequence)
        context_menu.addAction(save_sequence_action)

        save_sequence_action_csv = QAction('Save Sequence as CSV', self)
        save_sequence_action_csv.triggered.connect(self.save_sequence_csv)
        context_menu.addAction(save_sequence_action_csv)


        delete_sequence_action = QAction('delete Sequence', self)
        delete_sequence_action.triggered.connect(self.delete_sequence)
        context_menu.addAction(delete_sequence_action)
        
        edit_sequence_action = QAction('edit Sequence', self)
        edit_sequence_action.triggered.connect(self.edit_sequence)
        context_menu.addAction(edit_sequence_action)
        
        plot_sequence_action = QAction('plot Sequence', self)
        plot_sequence_action.triggered.connect(self.plot_sequence)
        context_menu.addAction(plot_sequence_action)

        print_sequence_action = QAction('print Sequence', self)
        print_sequence_action.triggered.connect(self.print_sequence)
        context_menu.addAction(print_sequence_action)

        
        
        context_menu.exec_(self.mapToGlobal(position))
        

    def save_sequence(self):

        self.save_sequence_signal.emit(self.text_to_emit)

    def save_sequence_csv(self):            
            self.save_sequence_signal_csv.emit(self.text_to_emit)

    def edit_sequence(self):

        self.edit_sequence_signal.emit(self.text_to_emit)
    def delete_sequence(self):

        self.delete_sequence_signal.emit(self.text_to_emit)
    def plot_sequence(self):

        self.plot_sequence_signal.emit(self.text_to_emit)

    def print_sequence(self):

        self.print_sequence_signal.emit(self.text_to_emit)  


class SequenceManagerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.sequence_manager = SequenceManager()
        self.selected_sequence_button = None
        self.setWindowTitle("Sequence Manager")

        self.initUI()

    def initUI(self):
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        

        self.menu_bar = self.create_menu_bar()
        self.layout.addWidget(self.menu_bar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.button_container = QWidget()
        self.seq_buttons_layout = QHBoxLayout(self.button_container)
        self.scroll_area.setWidget(self.button_container)

        self.layout.addWidget(self.scroll_area)

        self.sequence_view_area = QWidget()
        self.sequence_view_layout = QVBoxLayout(self.sequence_view_area)
        self.layout.addWidget(self.sequence_view_area, 2)

        self.update_buttons()

    def create_menu_bar(self):
        menu_bar = QMenuBar(self)
        file_menu = QMenu("File", self)
        tools_menu = QMenu("tools", self)

        load_action = QAction("Load Sequence Manager", self)
        load_action.triggered.connect(self.load_sequence_manager)
        file_menu.addAction(load_action)

        save_action = QAction("Save Sequence Manager", self)
        save_action.triggered.connect(self.save_sequence_manager)
        file_menu.addAction(save_action)

        open_runner_action = QAction("Open Runner", self)
        open_runner_action.triggered.connect(self.open_runner)
        tools_menu.addAction(open_runner_action)


        
        open_camera_action = QAction("Open Camera", self)
        open_camera_action.triggered.connect(self.open_camera)
        tools_menu.addAction(open_camera_action)




        menu_bar.addMenu(file_menu)
        menu_bar.addMenu(tools_menu)
        
        return menu_bar


    def open_runner(self):
        try:
            self.runner_widget = Runner(self.sequence_manager)
            self.runner_widget.show()
        except Exception as e:
            error_message = f"Can not Open Runner, An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
    def open_camera(self):
        try:
            self.camera_widget = ThorCamControlWidget()
            self.camera_widget.show()
        except Exception as e:
            error_message = f"Can not Open Camera, An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
            
            

    def save_sequence_manager(self):
        try:
            file_dialog = QFileDialog(self)
            file_name, _ = file_dialog.getSaveFileName(self, "Save Sequence Manager", "", "JSON Files (*.json)")
            if file_name:
                self.sequence_manager.to_json(file_name=file_name)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)

    def load_sequence_manager(self):
        try:
            file_dialog = QFileDialog(self)
            file_name, _ = file_dialog.getOpenFileName(self, "Open Sequence Manager", "", "JSON Files (*.json)")
            if file_name:
                self.sequence_manager = SequenceManager.from_json(file_name=file_name)
                self.update_buttons()
                self.display_sequence(flag=True)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)

    def save_sequence(self, sequence_name):
        try:
            file_dialog = QFileDialog(self)
            file_name, _ = file_dialog.getSaveFileName(self, "Save Singel Sequence", "", "JSON Files (*.json)")
            if file_name:
                self.sequence_manager.find_sequence_by_name(sequence_name).to_json(file_name)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)
            
        # self.sequence_manager.find_sequence_by_name(sequence_name).to_json(file_name+".json")
    def save_sequence_CSV(self, sequence_name): 
        try:
            file_dialog = QFileDialog(self)
            file_name, _ = file_dialog.getSaveFileName(self, "Save Singel Sequence", "", "CSV Files (*.csv)")
            if file_name:
                self.sequence_manager.find_sequence_by_name(sequence_name).to_csv(file_name+".csv")
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)

    def delete_sequence(self, sequence_name):
        try:
            self.sequence_manager.delete_sequence(sequence_name)
            self.update_buttons()
            self.display_sequence(flag=True)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)

    def edit_sequence(self, sequence_name):
        # ask user for new name
        try:
            new_sequence_name, ok = QInputDialog.getText(self, "Edit Sequence Name", "Enter new sequence name:")
            if ok and new_sequence_name:
                    
                self.sequence_manager.change_sequence_name(old_name=sequence_name, new_name=new_sequence_name)
                self.update_buttons()
                self.display_sequence(flag=True)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)

    def print_sequence(self, sequence_name):
        try:
            seq = self.sequence_manager.find_sequence_by_name(sequence_name)
            text = seq.get_event_tree()

            # make a message to print the text
            msg = QMessageBox()
            msg.setWindowTitle("Sequence Tree")
            msg.setText(text)
            msg.exec_()
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)

        
         
        

    def plot_sequence(self, sequence_name):
        try:
            #makZe a dialog to ask for channel name with a combobox
            channels =["All Channels"]+ [ch.name for ch in self.sequence_manager.find_sequence_by_name(sequence_name).channels] 
            types = ["Normal","With Relations"]
            dialog = CustomDialog(channels, types)
            if dialog.exec_() == QDialog.Accepted:
                channel_name, channel_type,resolution = dialog.get_values()

                if channel_name != "All Channels":
                    if channel_type == "Normal":
                        self.sequence_manager.find_sequence_by_name(sequence_name).plot([channel_name],resolution=resolution)
                    else :
                        self.sequence_manager.find_sequence_by_name(sequence_name).plot_all([channel_name],resolution=resolution)

                else:  
                    if channel_type == "Normal":
                        self.sequence_manager.find_sequence_by_name(sequence_name).plot(resolution=resolution)
                    else :
                        self.sequence_manager.find_sequence_by_name(sequence_name).plot_all(resolution=resolution)
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            QMessageBox.critical(self, "Error", error_message)


    def update_buttons(self):
        for i in reversed(range(self.seq_buttons_layout.count())):
            self.seq_buttons_layout.itemAt(i).widget().setParent(None)

        for seq in self.sequence_manager.main_sequences.values():
            button = DraggableButton(seq.sequence_name, self)
            button.clicked.connect(self.display_sequence)
            button.save_sequence_signal.connect(self.save_sequence)
            button.save_sequence_signal_csv.connect(self.save_sequence_CSV)
            button.delete_sequence_signal.connect(self.delete_sequence)
            button.edit_sequence_signal.connect(self.edit_sequence)
            button.plot_sequence_signal.connect(self.plot_sequence)
            button.print_sequence_signal.connect(self.print_sequence)


            button.setStyleSheet(self.get_button_style(False))
            self.seq_buttons_layout.addWidget(button)

        add_button = DraggableButton("+", self)
        add_button.clicked.connect(self.add_new_sequence)
        add_button.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: 2px solid #4CAF50;
                border-radius: 10px;
                padding: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3e8e41;
            }
        """)
        self.seq_buttons_layout.addWidget(add_button, alignment=Qt.AlignLeft)

    def move_button(self, target, source):
        source_index = self.seq_buttons_layout.indexOf(source)
        target_index = self.seq_buttons_layout.indexOf(target)

        #rearrange the sequences in the sequence manager
        # target_index_manager = self.sequence_manager.main_sequences[target.text()]["index"]


        self.sequence_manager.move_sequence_to_index(source.text(),target_index )


        self.seq_buttons_layout.insertWidget(target_index, source)



    def get_button_style(self, selected):
        if selected:
            return """
                QPushButton {
                    background-color: #FFC300; /* Bright Yellow */
                    color: black; 
                    font-weight: bold;
                    border: 2px solid #C70039;
                    border-radius: 5px;
                    padding: 5x;
                }
                QPushButton:hover {
                    background-color: #FFB200; /* Slightly Darker Yellow */
                }
                QPushButton:pressed {
                    background-color: #FFA200; /* Even Darker Yellow */
                }
            """
        else:
            return """
                QPushButton {
                    background-color: #2196F3;
                    color: black; 
                    font-weight: bold;

                    border: 2px solid #1976D2;
                    border-radius: 5px;
                    padding: 5x;
                }
                QPushButton:hover {
                    background-color: #1E88E5;
                }
                QPushButton:pressed {
                    background-color: #1565C0;
                }
            """

    def add_new_sequence(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Add New Sequence")

        layout = QVBoxLayout(dialog)

        new_sequence_button = QPushButton("Create New Sequence", dialog)
        load_sequence_button = QPushButton("Load Sequence from File", dialog)

        layout.addWidget(new_sequence_button)
        layout.addWidget(load_sequence_button)

        new_sequence_button.clicked.connect(lambda: self.create_new_sequence(dialog))
        load_sequence_button.clicked.connect(lambda: self.load_sequence_from_file(dialog))

        dialog.setLayout(layout)
        dialog.exec_()

    def create_new_sequence(self, dialog):
        sequence_name, ok = QInputDialog.getText(dialog, "Add New Sequence", "Enter sequence name:")
        if ok and sequence_name:
            try:
                self.sequence_manager.add_new_sequence(sequence_name)
                self.update_buttons()
                self.display_sequence(flag=True)
            except Exception as e:
                QMessageBox.critical(dialog, "Error", str(e))
        dialog.accept()

    def load_sequence_from_file(self, dialog):
        file_dialog = QFileDialog(self)
        file_name, _ = file_dialog.getOpenFileName(self, "Load Sequence from JSON", "", "JSON Files (*.json)")
        if file_name:
            try:
                sequence = Sequence.from_json(file_name)
                self.sequence_manager.load_sequence(sequence)
                self.update_buttons()
                self.display_sequence(flag=True)
            except Exception as e:
                QMessageBox.critical(dialog, "Error, File is corrupt due to ", str(e))
        dialog.accept()

    def display_sequence(self, flag=False):
        button = self.sender()
        if not flag:
            sequence_name = button.text()
            sequence = self.sequence_manager.find_sequence_by_name(sequence_name)
        else:
            if len(self.sequence_manager.main_sequences.items() )== 0 :
                return
            sequence = list(self.sequence_manager.main_sequences.values())[-1]
            sequence_name = sequence.sequence_name

        for i in reversed(range(self.sequence_view_layout.count())):
            widget_to_remove = self.sequence_view_layout.itemAt(i).widget()
            self.sequence_view_layout.removeWidget(widget_to_remove)
            widget_to_remove.setParent(None)
        
        synced_table_widget = SequenceViewerWdiget(sequence,self)
        self.sequence_view_layout.addWidget(synced_table_widget)

        if self.selected_sequence_button:
            self.selected_sequence_button.setStyleSheet(self.get_button_style(False))
        if not flag:
            self.selected_sequence_button = button
            self.selected_sequence_button.setStyleSheet(self.get_button_style(True))
        else:
            button = self.seq_buttons_layout.itemAt(self.seq_buttons_layout.count()-2).widget()
            self.selected_sequence_button = button
            self.selected_sequence_button.setStyleSheet(self.get_button_style(True))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SequenceManagerWidget()
    window.show()
    sys.exit(app.exec_())
