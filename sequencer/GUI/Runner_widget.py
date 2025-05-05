from PyQt5.QtWidgets import QWidget, QVBoxLayout,QFileDialog, QHBoxLayout, QPushButton, QComboBox, QListWidget, QListWidgetItem, QAbstractItemView, QMessageBox,QTextEdit,QSpinBox
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QListWidget, QTableWidget,
    QListWidgetItem, QProgressBar, QInputDialog, QMessageBox, QAbstractItemView,QTableWidgetItem
)
from PyQt5.QtCore import Qt
from collections import OrderedDict
from typing import List, Optional
from PyQt5.QtCore import QTimer


from sequencer.Sequence.sequence import Sequence,SequenceManager,creat_seq_manager

from sequencer.Drivers.ADwin_Modules import ADwin_Driver

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QListWidget, QListWidgetItem, QAbstractItemView, QMessageBox
from PyQt5.QtCore import Qt

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QComboBox, QListWidget, QListWidgetItem, QAbstractItemView, QMessageBox, QApplication, QMenu
from PyQt5.QtCore import Qt, QPoint
class ParameterListWidget(QWidget):
    def __init__(self, sequences:List[Sequence]=None, parent=None):
        super(ParameterListWidget, self).__init__(parent)
        if sequences is None:
            sequences = dict()
        # Create the main layout
        self.layout = QVBoxLayout()

        # Create the QTableWidget
        self.table_widget = QTableWidget()

        # Populate the table with parameter names and values
        self.populate_table(sequences)

        # Add the QTableWidget to the layout
        self.layout.addWidget(self.table_widget)

        # Add a button to demonstrate updating sequences

        # Set the layout for the widget
        self.setLayout(self.layout)

    def populate_table(self, sequences:List[Sequence]):
        """Populates the QTableWidget with parameter names and values."""
        try:
            self.table_widget.setColumnCount(len(sequences[0].sweep_dict.keys()))
            list_of_keys=[]
            for key in sequences[0].sweep_dict.keys():
                if isinstance(key, str):
                    list_of_keys.append(key)
                else:
                    list_of_keys.append("_".join([str(i) for i in key]))
                
            self.table_widget.setHorizontalHeaderLabels(list_of_keys)

            

                
            self.table_widget.setRowCount(len(sequences))
            for row, sequence in enumerate(sequences):
                for col, dict_value in enumerate(sequence.sweep_values):
                    # check if the dict_value dictionaary contains the key "value" or "relative time"
                    if "value" in dict_value.keys():
                        self.table_widget.setItem(row, col, QTableWidgetItem(str(dict_value["value"])))
                    else:
                        self.table_widget.setItem(row, col, QTableWidgetItem(str(dict_value["relative_time"])))
        except Exception as e:
            print(e)
            print("No sweeps there")
    def update_sequences(self, new_sequences):
        """Updates the QTableWidget with new sequences."""
        self.populate_table(new_sequences)





class CustomSequenceWidget(QWidget):
    def __init__(self, sequence_manager: SequenceManager,runner_widget):
        super().__init__()
        self.sequence_manager = sequence_manager
        self.runner_widget = runner_widget
        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # ComboBox for sequence names
        self.sequence_combo_box = QComboBox()
        for sequence in self.sequence_manager.main_sequences.values():
            self.sequence_combo_box.addItem(sequence.sequence_name)
        layout.addWidget(self.sequence_combo_box)

        # Button.sequence_name to add sequences to the list
        self.add_sequence_button = QPushButton("Add Sequence")
        self.add_sequence_button.clicked.connect(self.add_sequence_to_list)
        layout.addWidget(self.add_sequence_button)
        # ListWidget to display selected sequences
        self.selected_sequences_list = QListWidget()
        self.selected_sequences_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.selected_sequences_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.selected_sequences_list.customContextMenuRequested.connect(self.show_context_menu)
        layout.addWidget(self.selected_sequences_list)


        self.setLayout(layout)

       

    def add_sequence_to_list(self):
        sequence_name = self.sequence_combo_box.currentText()
        # print(sequence_name)
        if sequence_name:
            self.selected_sequences_list.addItem(sequence_name)
            self.runner_widget.refreash_custom_sweep_queue()



    def get_sequences_names(self):
        return  [self.selected_sequences_list.item(i).text() for i in range(self.selected_sequences_list.count())]



    def show_context_menu(self, position: QPoint):
        context_menu = QMenu(self)
        remove_action = context_menu.addAction("Remove")
        action = context_menu.exec_(self.selected_sequences_list.mapToGlobal(position))

        if action == remove_action:
            item = self.selected_sequences_list.itemAt(position)
            self.selected_sequences_list.takeItem(self.selected_sequences_list.row(item))

from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QProgressBar, QCheckBox,
    QMenuBar, QAction, QMessageBox
)
from PyQt5.QtGui import QIcon
import os 

import json
import random 
import datetime

import time 
class Runner(QWidget):
    def __init__(self, sequence_manager: SequenceManager):
        super().__init__()
        self.sequence_manager = sequence_manager

        self.paramerters_path = os.path.join(os.path.dirname(__file__), 'runner_default_settings.json')

        with open(self.paramerters_path, 'r') as json_file:
            loaded_settings = json.load(json_file)
            self.save_path = loaded_settings["default_save_path"]
        # Create these folders if they don't exist 
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        self.main_sweep_queue = []
        self.custom_sweep_queue = []

        
        
        self.refreash_sweep_queue()
        self.boot_ADwin()

        
        
        # create the folder if it does not exist 
        # if not os.path.exists(self.save_path):
        #     os.makedirs(self.save_path)
        self.initUI()
    def save_as_default_settings(self):
        # Define the default settings
        camera_default_settings = {
            "default_save_path": self.save_path,
        }

        # Write the settings to a JSON file
        with open(self.paramerters_path, 'w') as json_file:
            json.dump(camera_default_settings, json_file, indent=4)
        
        
        
    def boot_ADwin(self):
        # load the default ADwin process
        try:
            self.ADwin = ADwin_Driver()
            # show a message box to inform the user that the ADwin has been booted
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Information)
            msg_box.setWindowTitle("ADwin")
            msg_box.setText("ADwin booted successfully")
            QTimer.singleShot(2000, msg_box.close)  # Close the message box after 2000 milliseconds (2 seconds)
            msg_box.exec_()

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def initUI(self):
        # Create main layout
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # Create menu bar
        menu_bar = QMenuBar(self)
        tools_menu = menu_bar.addMenu('Tools')
        
        load_adwin_action = QAction('Load ADwin', self)
        load_adwin_action.triggered.connect(self.boot_ADwin)
        tools_menu.addAction(load_adwin_action)

        change_save_location_action = QAction('Change Save Location', self)
        change_save_location_action.triggered.connect(self.change_save_location)
        tools_menu.addAction(change_save_location_action)


        # Add menu bar to the layout
        main_layout.setMenuBar(menu_bar)

        # Run Main Button
        self.run_main_button = QPushButton("Run Main")
        self.run_main_button.clicked.connect(self.run_main)
        self.run_Custom_button = QPushButton("Run Custom")
        self.run_Custom_button.clicked.connect(self.run_Custom)
        
        dummy_layout = QHBoxLayout()
        dummy_layout.addWidget(self.run_main_button)
        dummy_layout.addWidget(self.run_Custom_button)
        main_layout.addLayout(dummy_layout)
        
        
        
        # Run Main Button
        self.Save_Data_CheckBox = QCheckBox("Save Data")
        self.Save_Data_CheckBox.clicked.connect(self.saver_handler_checkBox)
        main_layout.addWidget(self.Save_Data_CheckBox)


        # Run Custom Sequence Widget
        self.custom_sequence_widget = CustomSequenceWidget(self.sequence_manager,self)
        main_layout.addWidget(self.custom_sequence_widget)

        # Run Sweep Button
        self.run_sweep_button = QPushButton("Run Sweep")
        self.run_sweep_button.clicked.connect(self.run_sweep)


        self.combo_sweep = QComboBox()
        self.combo_sweep.addItems(["main","custom"])
        self.combo_sweep.currentIndexChanged.connect(self.check_sweep)
        self.randomize_queue_button = QPushButton("Randomize")
        self.randomize_queue_button.clicked.connect(self.randomize_queue)

        self.refreash_queue_button = QPushButton("refresh")
        self.refreash_queue_button.clicked.connect(self.refreash_queue)


        self.repeat_sweep_text = QSpinBox()



        self.sweep_runner_layout = QHBoxLayout()
        self.sweep_runner_layout.addWidget(self.run_sweep_button)
        self.sweep_runner_layout.addWidget(self.combo_sweep)
        self.sweep_runner_layout.addWidget(self.randomize_queue_button)
        self.sweep_runner_layout.addWidget(self.refreash_queue_button)
        self.sweep_runner_layout.addWidget(self.repeat_sweep_text)


        main_layout.addLayout(self.sweep_runner_layout)

        # Progress Bar
        self.progress_bar = QProgressBar()
        main_layout.addWidget(self.progress_bar)

        # print(self.main_sweep_queue)
        self.sweep_viewer =  ParameterListWidget(self.main_sweep_queue)
        
        self.sweep_viewer.populate_table(self.main_sweep_queue)
        main_layout.addWidget(self.sweep_viewer)

        self.setLayout(main_layout)
        self.setWindowTitle('Runner Widget')
        self.setWindowIcon(QIcon('path_to_icon.png'))  # Add path to your icon
        try: 
            self.refreash_queue()
            self.refreash_custom_sweep_queue()
        except:
            pass

        self.show()
    def change_save_location(self):
        self.save_path = QFileDialog.getExistingDirectory(self, "Select Save Folder", self.save_path)
        self.save_as_default_settings()
        
    def saver_handler_checkBox(self):
        pass 

    def refreash_sweep_queue(self):
        # FIX : Display an error message if the channels on different sequences overlap
        try:
            try:
                repeat = int(self.repeat_sweep_text.value())
            except Exception as e:
                repeat = 1
                print(e)
            self.main_sweep_queue =self.sequence_manager.get_sweep_sequences_main(repeat=repeat)
        except Exception as e:
            print("No sweeps there")
            print   (e)
    
    def refreash_custom_sweep_queue(self):
        try:
            self.custom_sweep_queue =self.sequence_manager.get_sweep_sequences_custom(self.custom_sequence_widget.get_sequences_names())
        except Exception as e:
            print("No sweeps there")
            print   (e)

    def refreash_queue(self,Working = False):
        try:
            if not Working:
                self.refreash_sweep_queue()
                self.refreash_custom_sweep_queue()

            if self.combo_sweep.currentText() == "main":
                self.sweep_viewer.populate_table(self.main_sweep_queue)
            else:
                self.sweep_viewer.populate_table(self.custom_sweep_queue)
        except Exception as e:
            self.main_sweep_queue = []
            self.custom_sweep_queue = []
            QMessageBox.critical(self, "Error", str(e))
            
        
    def check_sweep(self):
        try:
            if self.combo_sweep.currentText() == "main":
                self.sweep_viewer.populate_table(self.main_sweep_queue)
            else:
                self.sweep_viewer.populate_table(self.custom_sweep_queue)
        except Exception as e:
            #show error message that there is no custom sweep
            QMessageBox.critical(self, "Error", " No custom sweep found")
            self.combo_sweep.setCurrentIndex(0)

    def randomize_queue(self):
        try:
            
            random.shuffle(self.main_sweep_queue)
            

            try:
                items = list(self.custom_sweep_queue)
                random.shuffle(items)
                # Create a new dictionary from the shuffled list
                self.custom_sweep_queue = dict(items)

            except:
                pass
            
            try:
                if self.combo_sweep.currentText() == "main":
                    self.sweep_viewer.populate_table(self.main_sweep_queue)
                else:
                    self.sweep_viewer.populate_table(self.custom_sweep_queue)
            except:
                self.sweep_viewer.populate_table(self.main_sweep_queue)
        except Exception as e:  
            QMessageBox.critical(self, "Error", str(e))


    def run_main(self):
        try:
            main_sequence = self.sequence_manager.get_main_sequence()
            self.run_sequence(main_sequence)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_Custom(self):
        try:
            custom_sequence = self.sequence_manager.get_custom_sequence(self.custom_sequence_widget.get_sequences_names())
            self.run_sequence(custom_sequence)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_sweep(self):
        try:
            if self.combo_sweep.currentText() == "main":
                self.main_sweep_queue 
                # run the first sequence in the queue and pop it out untill there is not element and refreash UI 
                for sequence in self.main_sweep_queue:
                    self.run_sequence(sequence)
                    self.refreash_queue(Working=True)



            else:
                for key in self.custom_sequence_widget.keys():
                    sequence = self.custom_sweep_queue[key]
                    self.run_sequence(sequence)

                    self.refreash_queue(Working=True)






        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_sequence(self, sequence:Sequence):
        try:
            self.progress_bar.setValue(100)
            if self.Save_Data_CheckBox.isChecked(): 
                now = datetime.datetime.now()
                
                time_format = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
                sequence.to_json(filename=os.path.join(self.save_path, f"{time_format}.json"))
                #remove current data from the folder 
                for file in os.listdir(self.save_path):
                    if file.startswith("current"):
                        os.remove(os.path.join(self.save_path, file))
                time_format = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
                sequence.to_json(filename=os.path.join(self.save_path, f"current_{time_format}.json"))
            
            self.ADwin.add_to_queue(sequence)
            print(f"Running sequence: {sequence.sequence_name}")
            self.ADwin.initiate_all_experiments()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

        

    


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # sequence_manager = create_test_seq_manager()
    sequence_manager= creat_seq_manager()
    # sequence_manager.to_json(file_name="test_camera.json")
    # sequence_manager.add_new_sequence("Seq1")
    # sequence_manager.add_new_sequence("Seq2")
    # sequence_manager.add_new_sequence("Seq3")
    runner = Runner(sequence_manager)
    sys.exit(app.exec_())

