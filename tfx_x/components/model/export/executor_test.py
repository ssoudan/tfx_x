# Lint as: python3
#  Copyright 2021 ssoudan. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import json
import os
import tempfile
from typing import Dict, Text, Any, Optional

import absl
import tensorflow as tf
from tensorflow import keras
from tfx.dsl.io import fileio
from tfx.types import standard_artifacts

from tfx_x.components.model.export import executor
from tfx_x.components.model.export.executor import FUNCTION_NAME_KEY, MODEL_KEY, OUTPUT_KEY
from tfx_x.types.artifacts import ExportedModel


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()

    self.component_id = 'test_component'
    self._model_data_dir = tempfile.mkdtemp()

    num_classes = 10
    input_shape = (28, 28, 1)
    model = keras.Sequential(
      [
        keras.Input(shape=input_shape),
        keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(num_classes, activation="softmax"),
      ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    model.save(os.path.join(self._model_data_dir, 'serving_model_dir'))
    del model

    # Create input dict.
    self._model = standard_artifacts.Model()
    self._model.uri = self._model_data_dir

    self._input_dict = {
      MODEL_KEY: [self._model],
    }

    # Create output dict.
    self._output = ExportedModel()
    self._output_dir = os.path.join(tempfile.mkdtemp())
    self._output.uri = self._output_dir

    self._output_dict_sr = {
      OUTPUT_KEY: [self._output],
    }

    # Create exe properties.
    self._exec_properties = {
      'instance_name': 'something',
      FUNCTION_NAME_KEY: 'tfx_x.components.model.export.executor_test.stuffs',
    }

    # Create context
    self._tmp_dir = os.path.join(self._output_dir, '.temp')
    self._context = executor.Executor.Context(
      tmp_dir=self._tmp_dir, unique_id='2')

  def test(self):
    # Run executor.
    exporter = executor.Executor(self._context)
    exporter.Do(self._input_dict, self._output_dict_sr,
                self._exec_properties)

    # Check outputs.
    self.assertTrue(fileio.exists(self._output_dir))


def stuffs(model: tf.keras.Model, _pipeline_configuration: Dict[Text, Any], _output_dir: Text,
           _model_pushed_dir: Optional[Text]):
  absl.logging.debug('Model signatures: %s', json.dumps(list(model.signatures.keys())))


if __name__ == '__main__':
  tf.test.main()
