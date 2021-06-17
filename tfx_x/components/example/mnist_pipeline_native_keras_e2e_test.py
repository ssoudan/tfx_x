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

# Copyright 2019 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E2E Tests for tfx.examples.mnist.mnist_pipeline_native_keras."""

import os
import unittest
from typing import Text

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.orchestration import metadata
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

from tfx_x.components.example import mnist_pipeline_native_keras


@unittest.skipIf(tf.__version__ < '2',
                 'Uses keras Model only compatible with TF 2.x')
class MNISTPipelineNativeKerasEndToEndTest(tf.test.TestCase):

  def setUp(self):
    super(MNISTPipelineNativeKerasEndToEndTest, self).setUp()
    self._test_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
      self._testMethodName)

    self._pipeline_name = 'keras_test'
    self._data_root = os.path.join(os.path.dirname(__file__), '../testdata')

    to_key_fn = """
def to_key(m):
  return m.features.feature['image_class'].int64_list.value[0]
"""

    predicate_fn = """
def predicate(m):  
  return m.features.feature['image_class'].int64_list.value[0] < 4
    """

    self._custom_config = {'layer_count': 3,
                           'to_key_fn': to_key_fn,
                           'predicate_fn': predicate_fn}

    self._module_file = os.path.join(
      os.path.dirname(__file__), 'mnist_utils_native_keras.py')
    self._serving_model_dir = os.path.join(self._test_dir, 'serving_model')
    self._pipeline_root = os.path.join(self._test_dir, 'tfx', 'pipelines',
                                       self._pipeline_name)
    self._metadata_path = os.path.join(self._test_dir, 'tfx', 'metadata',
                                       self._pipeline_name, 'metadata.db')

  def assertExecutedOnce(self, component: Text) -> None:
    """Check the component is executed exactly once."""
    component_path = os.path.join(self._pipeline_root, component)
    self.assertTrue(fileio.exists(component_path))
    outputs = fileio.listdir(component_path)
    self.assertIn('.system', outputs)
    outputs.remove('.system')
    system_paths = [
      os.path.join('.system', path)
      for path in fileio.listdir(os.path.join(component_path, '.system'))
    ]
    self.assertNotEmpty(system_paths)
    self.assertIn('.system/executor_execution', system_paths)
    outputs.extend(system_paths)
    self.assertNotEmpty(outputs)
    for output in outputs:
      execution = fileio.listdir(os.path.join(component_path, output))
      self.assertLen(execution, 1)

  def assertPipelineExecution(self) -> None:
    self.assertExecutedOnce('ImportExampleGen')
    self.assertExecutedOnce('FromCustomConfig')
    self.assertExecutedOnce('evaluator')
    self.assertExecutedOnce('ExampleValidator')
    self.assertExecutedOnce('pusher')
    self.assertExecutedOnce('SchemaGen')
    self.assertExecutedOnce('Filter')
    self.assertExecutedOnce('StratifiedSampler')
    self.assertExecutedOnce('StatisticsGen')
    self.assertExecutedOnce('trainer')
    self.assertExecutedOnce('Transform')

  def testMNISTPipelineNativeKeras(self):
    if not tf.executing_eagerly():
      self.skipTest('The test requires TF2.')
    BeamDagRunner().run(
      mnist_pipeline_native_keras._create_pipeline(
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        custom_config=self._custom_config,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[]))

    self.assertTrue(fileio.exists(self._serving_model_dir))
    self.assertTrue(fileio.exists(self._metadata_path))
    metadata_config = metadata.sqlite_metadata_connection_config(
      self._metadata_path)
    expected_execution_count = 11
    with metadata.Metadata(metadata_config) as m:
      artifact_count = len(m.store.get_artifacts())
      execution_count = len(m.store.get_executions())
      self.assertGreaterEqual(artifact_count, execution_count)
      self.assertEqual(execution_count, expected_execution_count)

    self.assertPipelineExecution()

    # Runs pipeline the second time.
    BeamDagRunner().run(
      mnist_pipeline_native_keras._create_pipeline(
        pipeline_name=self._pipeline_name,
        data_root=self._data_root,
        custom_config=self._custom_config,
        module_file=self._module_file,
        serving_model_dir=self._serving_model_dir,
        pipeline_root=self._pipeline_root,
        metadata_path=self._metadata_path,
        beam_pipeline_args=[]))

    # Asserts cache execution.
    with metadata.Metadata(metadata_config) as m:
      # Artifact count is unchanged.
      self.assertLen(m.store.get_artifacts(), artifact_count)
      self.assertLen(m.store.get_executions(), expected_execution_count * 2)


if __name__ == '__main__':
  tf.test.main()
