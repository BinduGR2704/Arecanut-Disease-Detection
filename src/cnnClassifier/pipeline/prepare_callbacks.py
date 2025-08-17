# import os
# import urllib.request as request
# from zipfile import ZipFile
# import tensorflow as tf
# import time
# from cnnClassifier.entity.config_entity import PrepareCallbacksConfig # type: ignore


# class PrepareCallback:
#     def __init__(self, config):
#         self.config = config

#     def _create_tb_callbacks(self):
#         return tf.keras.callbacks.TensorBoard(
#             log_dir=str(self.config.tensorboard_root_log_dir)
#         )

#     def _create_ckpt_callbacks(self):
#         print("DEBUG CHECKPOINT TYPE:", type(self.config.checkpoint_model_filepath))  # Debugging print
        
#         # Ensure the directory exists
#         checkpoint_path = str(self.config.checkpoint_model_filepath)
#         checkpoint_dir = os.path.dirname(checkpoint_path)
        
#         if not os.path.exists(checkpoint_dir):
#             os.makedirs(checkpoint_dir)

#         # Force a '.h5' extension if not already
#         if not checkpoint_path.endswith('.h5'):
#             checkpoint_path += '.h5'

#         return tf.keras.callbacks.ModelCheckpoint(
#             filepath=checkpoint_path,  # Ensure it's a string and has the correct extension
#             save_best_only=True
#         )

#     def get_tb_ckpt_callbacks(self):
#         return [
#             self._create_tb_callbacks(),
#             self._create_ckpt_callbacks()
#         ]



# c:\MLProject\venv\python.exe -m pip install ensure

import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig


class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config


    
    #@property
    def _create_tb_callbacks(self):
        timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
        tb_running_log_dir = os.path.join(
            self.config.tensorboard_root_log_dir,
            f"tb_logs_at_{timestamp}",
        )
        return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
    

    #@property
    def _create_ckpt_callbacks(self):
        return tf.keras.callbacks.ModelCheckpoint(
            filepath=str(self.config.checkpoint_model_filepath),
            save_best_only=True
        )


    def get_tb_ckpt_callbacks(self):
        return [
            self._create_tb_callbacks(),
            self._create_ckpt_callbacks()
        ]
    

    