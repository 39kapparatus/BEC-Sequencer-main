import sys
import threading
import queue
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QCheckBox,
                             QHBoxLayout, QPushButton, QComboBox, QSpinBox, QMessageBox,
                             QFormLayout)
from PyQt5.QtCore import QTimer, Qt,QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap,QPainter 
from PIL import Image
from sequencer.imaging.thorlabs_tsi_sdk.thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
from sequencer.imaging.thorlabs_tsi_sdk.thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE, OPERATION_MODE
from sequencer.imaging.thorlabs_tsi_sdk.thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK

# from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame
# from thorlabs_tsi_sdk.tl_camera_enums import SENSOR_TYPE, OPERATION_MODE
# from thorlabs_tsi_sdk.tl_mono_to_color_processor import MonoToColorProcessorSDK
import copy
from PyQt5.QtGui import QImage, QPixmap, QPainter, QFont, QFontMetrics
import numpy as np
from sequencer.Sequence.sequence import Sequence
from fluorescence_count import *
import time

try:
    # if on Windows, use the provided setup script to add the DLLs folder to the PATH
    from sequencer.imaging.THORCAM.windows_setup import configure_path
    configure_path()
except ImportError:
    configure_path = None


import matplotlib.pyplot as plt

# --- Your provided code classes ---
class ImageAcquisitionThread(threading.Thread):
    #while this thread is running, the camera will try to take pictures and put them in the queue
    def __init__(self, camera):
        super(ImageAcquisitionThread, self).__init__()
        self._camera = camera
        self._camera.exposure_time_us=113
        self._camera.gain = 0
        self._previous_timestamp = 0

        if self._camera.camera_sensor_type != SENSOR_TYPE.BAYER:
            self._is_color = False
        else:
            self._mono_to_color_sdk = MonoToColorProcessorSDK()
            self._image_width = self._camera.image_width_pixels
            self._image_height = self._camera.image_height_pixels
            self._mono_to_color_processor = self._mono_to_color_sdk.create_mono_to_color_processor(
                SENSOR_TYPE.BAYER,
                self._camera.color_filter_array_phase,
                self._camera.get_color_correction_matrix(),
                self._camera.get_default_white_balance_matrix(),
                self._camera.bit_depth
            )
            self._is_color = True

        self._bit_depth = camera.bit_depth
        self._camera.image_poll_timeout_ms = 0
        self._image_queue = queue.Queue(maxsize=2)
        self._stop_event = threading.Event()

    def get_output_queue(self):
        return self._image_queue

    def stop(self):
        self._stop_event.set()

    def _get_color_image(self, frame):
        width = frame.image_buffer.shape[1]
        height = frame.image_buffer.shape[0]
        if (width != self._image_width) or (height != self._image_height):
            self._image_width = width
            self._image_height = height
        color_image_data = self._mono_to_color_processor.transform_to_24(frame.image_buffer,
                                                                         self._image_width,
                                                                         self._image_height)
        color_image_data = color_image_data.reshape(self._image_height, self._image_width, 3)
        return Image.fromarray(color_image_data, mode='RGB')

    def _get_image(self, frame):
        scaled_image = frame.image_buffer >> (self._bit_depth - 8)
        return Image.fromarray(scaled_image),frame.image_buffer

    def run(self):
        while not self._stop_event.is_set():
            try:
                frame = self._camera.get_pending_frame_or_null()
                
                if frame is not None:
                    if self._is_color:
                        pil_image = self._get_color_image(frame)
                    else:
                        pil_image,numpy_array = self._get_image(frame)
                    self._image_queue.put_nowait((pil_image,numpy_array))
            except queue.Full:
                #print("queue.Full")
                pass
            except Exception as error:
                print("Here error")
                print(f"Encountered error: {error}, image acquisition will stop.")
                break
        print("Image acquisition has stopped")
        if self._is_color:
            self._mono_to_color_processor.dispose()
            self._mono_to_color_sdk.dispose()



from PyQt5.QtWidgets import QWidget, QLabel, QVBoxLayout, QSizePolicy
from PyQt5.QtCore import QTimer, Qt, QPoint
from PyQt5.QtGui import QImage, QPixmap, QPainter
import queue
import datetime


class DataItem:
    def __init__(self, dictionary_temp=None, images=None):
        self.dictionary_temp = dictionary_temp
        self.images = images if images is not None else []
    
    
    def save(self, file_name):
        # Convert dictionary_temp to a numpy array using np.array with dtype=object
        dictionary_temp_np = np.array([self.dictionary_temp], dtype=object)
        
        # Save images as a numpy array
        images_np = np.array(self.images, dtype=object)
        
        # Save both arrays in an npz file
        np.savez(file_name, dictionary_temp=dictionary_temp_np, images=images_np)
    
    @classmethod
    def load(cls, file_name):
        # Load data from npz file
        data = np.load(file_name, allow_pickle=True)
        
        # Extract dictionary_temp and images
        dictionary_temp = data['dictionary_temp'][0]
        images = data['images'].tolist()
        
        return cls(dictionary_temp=dictionary_temp, images=images)

class LiveViewWidget(QWidget):
    #this is the widget that displays the image, while the thread is running, it will try to access the queue, see if an image is there and if it is, update the live view
    def __init__(self, image_queue,condition,running,main_camera):
        super(LiveViewWidget, self).__init__()
        #the condition is what governs the logic for whether or not we are counting atom number
        self.image_queue = image_queue
        self.main_camera = main_camera
        self.condition=condition
        self.running=running
        self.n=0

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setMinimumSize(1, 1)
        self.count_label = QLabel("Atom Number:")
        self.count_label.setAlignment(Qt.AlignLeft)
        self.count_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.count_label.setMinimumSize(1, 1)
        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.count_label)
        self.setLayout(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)
        self.timer.start(10)



    def save_images(self,numpy_data):
        # check if the save checkbox is checked 
        
            # check if the experiment mode is ongoing
        if self.main_camera.experiment_mode.currentText() == 'Ongoing Experiment':
                # in the source folder the sequence should start with "current"
            print(self.main_camera.default_source_path)
            current_source_file = [file for file in os.listdir(self.main_camera.default_source_path) if file.startswith("current")]
            if current_source_file:
                current_source_file= current_source_file[0]
            else:
                # make a message box to tell the user to select the source folder 
                QMessageBox.warning(self, "Warning", "Please select a valid source file. The current source folder is empty")
                return
            
            
            print(current_source_file)
            print(os.path.join(self.main_camera.default_source_path,current_source_file))
            
            temp_seq = Sequence.from_json(file_name=os.path.join(self.main_camera.default_source_path,current_source_file))
            parameters = temp_seq.get_parameter_dict()
            self.main_camera.parameter_list.update_parameters(parameters)
            
            current_source_file= current_source_file.replace(".json","")
            
            if self.main_camera.save_checkbox.isChecked():
                    # check if current file is also the current file 
                    current_destination_file = [file for file in os.listdir(self.main_camera.default_destination_path) if file.startswith("current")]
                    if current_destination_file:
                        current_destination_file = current_destination_file[0]
                        current_destination_file_temp = current_destination_file.replace("current_","")
                        current_destination_file_temp = current_destination_file_temp.replace(".npz","")
                        print("comparing folders")
                        print(current_destination_file_temp)
                        print(current_source_file)
                        
                        if current_source_file.replace("current_","") == current_destination_file_temp:
                            # save to the destination path
                            # unpack the file and save it again to the destination path
                            
                            old_data =DataItem.load(os.path.join(self.main_camera.default_destination_path,current_destination_file))
                            old_data.images.append(numpy_data)
                            print('saving to the default saving path1')
                            # remove the old npz file 
                            os.remove(os.path.join(self.main_camera.default_destination_path,current_destination_file))
                            old_data.save(os.path.join(self.main_camera.default_destination_path,current_destination_file))
                        else:
                            # save to the default saving path
                            # rename the current file in the destination path folder to be without current
                            os.rename(os.path.join(self.main_camera.default_destination_path,current_destination_file),os.path.join(self.main_camera.default_destination_path,current_destination_file.replace("current_","")))
                            # save to the default saving path
                            print('saving to the default saving path2')
                            with open(os.path.join(self.main_camera.default_source_path, current_source_file+".json")) as json_file:
                                json_str_data = json_file.read()
                                json_data = json.loads(json_str_data)

                                

                            new_data = DataItem(dictionary_temp=json_data, images=[numpy_data])
                            new_data.save(os.path.join(self.main_camera.default_destination_path,current_source_file))
                    else:
                        # save to the default saving path
                        # rename the current file in the destination path folder to be without current
                        # os.rename(os.path.join(self.main_camera.default_destination_path,current_destination_file),os.path.join(self.main_camera.default_destination_path,current_destination_file.replace("current","")))
                        # save to the default saving path
                        print('saving to the default saving path3')
                        with open(os.path.join(self.main_camera.default_source_path, current_source_file+".json")) as json_file:
                            json_str_data = json_file.read()
                            json_data = json.loads(json_str_data)

                        new_data = DataItem(dictionary_temp=json_data, images=[numpy_data])
                        new_data.save(os.path.join(self.main_camera.default_destination_path,current_source_file))
          

        else:
            # save to the default saving path
            if self.main_camera.save_checkbox.isChecked():
                # get the default saving path
                
                now = datetime.datetime.now()
                time_stamp = now.strftime("%Y-%m-%d_%H-%M-%S-%f")
                file_name = f"{time_stamp}"
                file_path = os.path.join(self.main_camera.default_saving_path, file_name)
                print("Saving_images to " , file_path)
                np.save(file_path,numpy_data)


                # plt.imshow(numpy_data, cmap='gray')
                # plt.axis('off')  # Turn off axis numbers and ticks
                # plt.savefig(file_path+".png", bbox_inches='tight', pad_inches=0.0)  # Save as PNG file

                # saved_image = Image.fromarray(numpy_data.astype('uint8'))
                # # Save the saved_image
                # saved_image.save()

    def receive_value(self,value):
        #this function is connected to the emit function of the fluorescence count intialization and the emit of the exposure time value change
        self.exposure_time=value

    def update_image(self):
        try:
            with self.condition:
                image,numpy_data = self.image_queue.get_nowait()
                self.save_images(numpy_data)
                image = image.convert('RGB')
                data = image.tobytes("raw", "RGB")
                q_image = QImage(data, image.width, image.height, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)
                scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.image_label.setPixmap(scaled_pixmap)
                self.n+=1
                if self.running["counting"] == "Once":
                    #sets the ROI, save the calibration image, then set to run continuously
                    self.count,self.ROI=Get_Atom_Number(ExposureTime=self.exposure_time*10**(-6),select_ROI=True,image=q_image)
                    self.count_label.setText("Atom Number: "+f"{self.count:.3e}")
                    try:
                        save_dir = os.path.join('..', 'data', 'saving_folder')
                        save_path = os.path.join(save_dir, 'calibration_picture.jpg')
                        if not os.path.exists(save_dir):
                            raise FileNotFoundError(f"Save directory does not exist: {os.path.abspath(save_dir)}")
                        success = q_image.save(save_path, "JPG")
                        if not success:
                            raise IOError("QImage failed to save the image.")

                    except Exception as e:
                        QMessageBox.critical(self, "Save Error", str(e))
                    self.running["counting"] = "Run"
                    self.condition.notify_all()
                elif self.running["counting"] == "Run" and self.n%100==0:
                    #every 100 frames, update the atom number
                    self.count,self.ROI=Get_Atom_Number(ExposureTime=self.exposure_time*10**(-6),ROI=self.ROI,image=q_image)
                    self.count_label.setText("Atom Number: "+f"{self.count:.3e}")
                    self.n=1
        except queue.Empty:
            pass




class THORCAM_HANDLER():
    def __init__(self):
        self.sdk = TLCameraSDK()
        self.camera = None
        self.acquisition_thread = None
        self.camera_mode = None

    def get_camera_list(self):
        return self.sdk.discover_available_cameras()
    
    def open_camera(self, camera_index=None, serial_number=None):
        try:
            if self.camera:
                self.camera.dispose()
        except:
            print('No camera to dispose')

        if camera_index is not None:
            try:
                self.camera = self.sdk.open_camera(self.get_camera_list()[camera_index])
            except:
                # error message
                error_message = f"An error occurred: {str(e)}"
                QMessageBox.critical(self, "Error", error_message)



            
        elif serial_number is not None:
            self.camera = self.sdk.open_camera(serial_number)
        
        
    
    def change_camera_live_mode(self, live_or_trigger):
        self.camera_mode = live_or_trigger
        try:
            self.camera.disarm()
        except:
            print   ('No camera to disarm')

        
        if live_or_trigger.lower() == 'live':
            self.camera.frames_per_trigger_zero_for_unlimited = 0
            self.camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
        else:
            self.camera.frames_per_trigger_zero_for_unlimited = 1
            self.camera.operation_mode = OPERATION_MODE.HARDWARE_TRIGGERED
        
        self.camera.arm(2)
        if live_or_trigger.lower() == 'live':
            self.camera.issue_software_trigger()
    
    def set_camera_params(self, exposure_time_us, gain):
        print('Setting camera parameters')
        print(f'Exposure time: {exposure_time_us} us')
        print(f'Gain: {gain}')


        if exposure_time_us < 64:
            exposure_time_us = 64
            print('Minimum exposure time is 64 us')
        if gain < 0:
            gain = 0
            print('Minimum gain is 0')

        if gain > 40:
            gain = 40
            print('Maximum gain is 40')

        try:
            self.camera.exposure_time_us = exposure_time_us
        except:
            print('No camera to set exposure')
        try:    
            self.camera.gain = 10*gain
        except:
            print('No camera to set gain')


    
    def dispose_all_camera_resources(self):
        try:        
            self.camera.dispose()
        except:
            print('No camera to dispose')

        try:
            self.sdk.dispose()
        except:
            print('No sdk to dispose')

    def start_acquisition_thread(self):
        self.kill_acquisition_thread()
        self.acquisition_thread = ImageAcquisitionThread(self.camera)
        self.acquisition_thread.start()
    
    def kill_acquisition_thread(self):
        try:
            if self.acquisition_thread:
                self.acquisition_thread.stop()
                self.acquisition_thread.join()
        except:
            print('No thread to kill')

    def __del__(self):
        self.dispose_all_camera_resources()
        self.kill_acquisition_thread()

# --- Custom PyQt Widget for THORCAM_HANDLER control ---

from PyQt5.QtWidgets import QApplication, QMainWindow, QSpinBox, QVBoxLayout, QWidget
from PyQt5.QtCore import pyqtSignal, QTimer


class CustomSpinBox(QSpinBox):
    valueConfirmed = pyqtSignal(int)  # Custom signal to emit when value is confirmed

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value_before_edit = self.value()

        # Initialize the timer
        self.confirmationTimer = QTimer(self)
        self.confirmationTimer.setSingleShot(True)

        # self.confirmationTimer.timeout.connect(self.emitValueConfirmed)
        # self.valueChanged.connect(self.startConfirmationTimer)

    def keyPressEvent(self, event):
        super().keyPressEvent(event)
        if event.key() in (Qt.Key_Enter, Qt.Key_Return):
            self.emitValueConfirmed()

    def emitValueConfirmed(self):
        if self.value() != self._value_before_edit:
            self.valueConfirmed.emit(self.value())
            self._value_before_edit = self.value()
            self.confirmationTimer.stop()

    def startConfirmationTimer(self):
        self.confirmationTimer.start(2000)

import json
import os

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem, QPushButton

class ParameterListWidget(QWidget):
    def __init__(self, parameters=None, parent=None):
        super(ParameterListWidget, self).__init__(parent)
        if parameters is None:
            parameters = []
        # Create the main layout
        self.layout = QVBoxLayout()

        # Create the QTableWidget
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Parameter", "Value"])

        # Populate the table with parameter names and values
        self.populate_table(parameters)

        # Add the QTableWidget to the layout
        self.layout.addWidget(self.table_widget)

        # Add a button to demonstrate updating parameters

        # Set the layout for the widget
        self.setLayout(self.layout)

    def populate_table(self, parameters):
        """Populates the QTableWidget with parameter names and values."""
        self.table_widget.setRowCount(len(parameters))
        print("parameters in table widget")
        print(parameters)
        
        try:
            for row, (name, value) in enumerate(parameters.items()):
                self.table_widget.setItem(row, 0, QTableWidgetItem(name))
                self.table_widget.setItem(row, 1, QTableWidgetItem(str(value)[:8]))
        except:
            print('Error in populating the table')
    def update_parameters(self, new_parameters):
        """Updates the QTableWidget with new parameters."""
        self.populate_table(new_parameters)

import os
import json
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QCheckBox, QFileDialog
from PyQt5.QtCore import Qt
import queue

class ThorCamControlWidget(QWidget):
    exposure = pyqtSignal(float) #signal to emit the exposure time through the fluorescence count initialize function
    def __init__(self, parent=None):
        super(ThorCamControlWidget, self).__init__(parent)

        # Load default parameters
        # Folder paths are in the same directory as the script
        self.parameters_path = os.path.join(os.path.dirname(__file__), 'camera_default_settings.json')

        with open(self.parameters_path, 'r') as json_file:
            loaded_settings = json.load(json_file)
            self.default_source_path = loaded_settings["default_source_path"]
            self.default_destination_path = loaded_settings["default_destination_path"]
            self.default_saving_path = loaded_settings["default_saving_path"]
            self.default_exposure_time = loaded_settings["default_exposure_time"]
            self.default_gain = loaded_settings["default_gain"]

        # Create these folders if they don't exist 
        if not os.path.exists(self.default_source_path):
            os.makedirs(self.default_source_path)
        if not os.path.exists(self.default_destination_path):
            os.makedirs(self.default_destination_path)
        if not os.path.exists(self.default_saving_path):
            os.makedirs(self.default_saving_path)
        
        self.thor_cam = THORCAM_HANDLER()
        self.init_ui()

    def save_as_default_settings(self):
        # Define the default settings
        camera_default_settings = {
            "default_source_path": self.default_source_path,
            "default_destination_path": self.default_destination_path,
            "default_saving_path": self.default_saving_path,
            "default_exposure_time": self.exposure_spin.value(),
            "default_gain": self.gain_spin.value()
        }

        # Write the settings to a JSON file
        with open(self.parameters_path, 'w') as json_file:
            json.dump(camera_default_settings, json_file, indent=4)

    def init_ui(self):
        self.setWindowTitle("ThorCam Control Panel")
        self.resize(800, 600)

        # Layouts
        self.main_layout = QVBoxLayout()
        self.controls_layout = QHBoxLayout()
        self.settings_layout = QHBoxLayout()
        self.save_layout = QHBoxLayout()
        self.count_layout = QHBoxLayout()

        # Live View
        self.condition = threading.Condition() # condition for the counting logic
        self.running= {"counting":"False"} #dict for the condition
        self.image_queue=queue.Queue() #creates the queue
        self.live_view = LiveViewWidget(image_queue=self.image_queue,condition=self.condition,running=self.running,main_camera=self)
        
        # Camera List
        self.refresh_cameras_button = QPushButton("Refresh Cameras")
        self.refresh_cameras_button.clicked.connect(self.refresh_cameras)

        self.camera_list = QComboBox()
        self.refresh_cameras()
        self.camera_list.currentIndexChanged.connect(self.camera_selected)

        # Camera Controls
        self.open_button = QPushButton("Open Camera")
        self.open_button.clicked.connect(self.open_camera)

        self.close_button = QPushButton("Close Camera")
        self.close_button.clicked.connect(self.close_camera)

        # Camera Parameters
        self.exposure_spin = CustomSpinBox()
        self.exposure_spin.setRange(64, 1000000)
        self.exposure_spin.setValue(self.default_exposure_time)

        self.gain_spin = CustomSpinBox()
        self.gain_spin.setRange(0, 100)
        self.gain_spin.setValue(self.default_gain)

        self.exposure_spin.valueConfirmed.connect(self.apply_params)
        self.gain_spin.valueConfirmed.connect(self.apply_params)

        self.gain_spin.confirmationTimer.timeout.connect(self.gain_spin.emitValueConfirmed)
        self.gain_spin.valueChanged.connect(self.gain_spin.startConfirmationTimer)
        self.exposure_spin.confirmationTimer.timeout.connect(self.exposure_spin.emitValueConfirmed)
        self.exposure_spin.valueChanged.connect(self.exposure_spin.startConfirmationTimer)
        self.exposure_spin.valueConfirmed.connect(self.live_view.receive_value) #connects the emit value_confirmed to the live_view. This updates the exposure time

        self.camera_mode_compo = QComboBox()
        self.camera_mode_compo.addItems(['Live', 'Trigger'])
        self.camera_mode_compo.currentIndexChanged.connect(self.change_camera_live_mode)

        # Adding widgets to layouts
        self.settings_layout.addWidget(QLabel("Exposure Time (us):"))
        self.settings_layout.addWidget(self.exposure_spin)
        self.settings_layout.addWidget(QLabel("Gain:"))
        self.settings_layout.addWidget(self.gain_spin)
        
        self.settings_layout.addWidget(QLabel("Camera Mode:"))
        self.settings_layout.addWidget(self.camera_mode_compo, 2)

        self.controls_layout.addWidget(QLabel("Select Camera:"))
        self.controls_layout.addWidget(self.camera_list)
        self.controls_layout.addWidget(self.refresh_cameras_button)
        self.controls_layout.addWidget(self.open_button)
        self.controls_layout.addWidget(self.close_button)

        # Experiment, Save and Count Layouts
        self.experiment_mode = QComboBox()
        self.experiment_mode.addItems(['No Experiment', 'Ongoing Experiment'])
        self.experiment_mode.currentIndexChanged.connect(self.change_experiment_mode)

        self.save_checkbox = QCheckBox("Save Images")
        self.save_checkbox.stateChanged.connect(self.save_images)

        self.save_folder_button = QPushButton("Select Save Folder")
        self.save_folder_button.clicked.connect(self.select_save_folder)
        
        self.destination_folder_button = QPushButton("Select Destination Folder")
        self.destination_folder_button.clicked.connect(self.select_destination_folder)
        
        self.source_folder_button = QPushButton("Select Source Folder")
        self.source_folder_button.clicked.connect(self.select_source_folder)

        self.count_checkbox = QCheckBox("Fluorescence Count")
        self.count_checkbox.stateChanged.connect(self.initialize_count)

        self.update_ROI_button = QPushButton("Update ROI")
        self.update_ROI_button.clicked.connect(self.update_ROI)

        self.save_layout.addWidget(QLabel("Experiment Mode:"))
        self.save_layout.addWidget(self.experiment_mode)
        self.save_layout.addWidget(self.save_checkbox)

        self.count_label=QLabel("Fluorescence Count:")
        self.count_layout.addWidget(self.count_label)
        self.count_layout.addWidget(self.count_checkbox)

        self.main_layout.addLayout(self.controls_layout)
        self.main_layout.addLayout(self.settings_layout)
        self.main_layout.addLayout(self.save_layout)
        self.main_layout.addLayout(self.count_layout)

        self.live_params = QHBoxLayout()
        self.parameter_list = ParameterListWidget()
        
        self.live_params.addWidget(self.parameter_list)
        self.live_params.addWidget(self.live_view, 2)
        self.main_layout.addLayout(self.live_params, 2)

        self.setLayout(self.main_layout)

    def save_images(self, state):
        if state == Qt.Checked:
            self.camera_mode_compo.setCurrentIndex(1)
            self.camera_mode_compo.setEnabled(False)
            self.live_view.save = True

        else:
            self.live_view.save = False
            self.camera_mode_compo.setEnabled(True)

        if self.experiment_mode.currentText() == 'No Experiment':
            if state == Qt.Checked:
                self.save_layout.addWidget(self.save_folder_button)
            else:
                self.save_layout.removeWidget(self.save_folder_button)
                self.save_folder_button.setParent(None)
        elif self.experiment_mode.currentText() == 'Ongoing Experiment':
            if state == Qt.Checked:
                self.save_layout.addWidget(self.destination_folder_button)
            else:
                self.save_layout.removeWidget(self.destination_folder_button)
                self.destination_folder_button.setParent(None)

    def change_experiment_mode(self):
        if self.experiment_mode.currentText() == 'Ongoing Experiment':
            self.camera_mode_compo.setCurrentIndex(1)
            self.camera_mode_compo.setEnabled(False)
            self.save_layout.addWidget(self.source_folder_button)
            
            # Remove the save folder button if it is there
            if self.save_folder_button.parent() is not None:
                self.save_layout.removeWidget(self.save_folder_button)
                self.save_folder_button.setParent(None)
            
            # Add destination folder button if save checkbox is checked
            if self.save_checkbox.isChecked():
                self.save_layout.addWidget(self.destination_folder_button)
        else:
            self.camera_mode_compo.setEnabled(True)
            self.save_layout.removeWidget(self.source_folder_button)
            self.source_folder_button.setParent(None)

            # Remove the destination folder button if it is there
            if self.destination_folder_button.parent() is not None:
                self.save_layout.removeWidget(self.destination_folder_button)
                self.destination_folder_button.setParent(None)

            # Add save folder button if save checkbox is checked
            if self.save_checkbox.isChecked():
                self.save_layout.addWidget(self.save_folder_button)

    def initialize_count(self,state):
        if state == Qt.Checked:
            #updates the exposure value for the live_view
            self.exposure.emit(self.exposure_spin.value())
            with self.condition:
                #starts the calibration
                self.running["counting"] = "Once"
                self.condition.notify_all()
                
                
                #make the update ROI button appear
                self.count_layout.addWidget(self.update_ROI_button)
            
        else:
            try:
                #removes the ROI button and stops the counting
                self.count_layout.removeWidget(self.update_ROI_button)
                self.update_ROI_button.setParent(None)
                with self.condition:
                    self.running["counting"] = "False"
                    self.condition.notify_all()
            except Exception as e:
                print(f"Exception occurred: {e}")
        

    def update_ROI(self): 
        with self.condition:     
            self.running["counting"] = "Once"
            self.condition.notify_all()
            

    def select_save_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Save Folder", self.default_saving_path)
        if folder:
            self.default_saving_path = folder
            self.save_as_default_settings()

    def select_destination_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Destination Folder", self.default_destination_path)
        if folder:
            self.default_destination_path = folder
            self.save_as_default_settings()

    def select_source_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Source Folder", self.default_source_path)
        if folder:
            self.default_source_path = folder
            self.save_as_default_settings()

    def change_camera_live_mode(self):
        mode = self.camera_mode_compo.currentText()
        try:
            if mode.lower().strip() =="live":
                self.save_checkbox.setChecked(False)
            self.thor_cam.change_camera_live_mode(mode)
        except Exception as e:
            print(f'Error changing camera mode: {e}')

    def refresh_cameras(self):
        self.camera_list.clear()
        cameras = self.thor_cam.get_camera_list()
        self.camera_list.addItems(cameras)
        if cameras:
            self.camera_list.setCurrentIndex(0)

    def camera_selected(self, index):
        pass

    def open_camera(self):
        index = self.camera_list.currentIndex()
        if index >= 0:
            self.thor_cam.open_camera(camera_index=index)
            self.thor_cam.change_camera_live_mode(self.camera_mode_compo.currentText())
            self.thor_cam.start_acquisition_thread()
            self.live_view.image_queue = self.thor_cam.acquisition_thread.get_output_queue()
            self.live_view.timer.start(10)
            self.open_button.setEnabled(False)
            self.close_button.setEnabled(True)

    def close_camera(self):
        self.thor_cam.kill_acquisition_thread()
        self.thor_cam.camera.dispose()
        self.live_view.timer.stop()
        self.open_button.setEnabled(True)
        self.close_button.setEnabled(False)

    def apply_params(self):
        exposure_time_us = self.exposure_spin.value()
        gain = self.gain_spin.value()
        self.thor_cam.set_camera_params(exposure_time_us, gain)

    def closeEvent(self, event):
        self.thor_cam.kill_acquisition_thread()
        self.thor_cam.dispose_all_camera_resources()
        event.accept()



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ThorCamControlWidget()
    window.show()
    window.exposure.connect(window.live_view.receive_value) #connects the emit of ThorCamControlWidget to the live_view
    
    sys.exit(app.exec_())