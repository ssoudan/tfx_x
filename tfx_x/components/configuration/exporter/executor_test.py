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
"""Tests for Exporter."""
import os

import tensorflow as tf
from tfx.dsl.io import fileio
from tfx.types import channel_utils, artifact_utils
from tfx.utils import json_utils

from tfx_x.components.configuration import artifacts
from tfx_x.components.configuration.exporter import component, executor
from tfx_x.components.configuration.exporter.executor import CUSTOM_CONFIG_KEY, PIPELINE_CONFIGURATION_KEY


class ExecutorTest(tf.test.TestCase):

  def setUp(self):
    super(ExecutorTest, self).setUp()
    self.name = 'HelloWorld'
    self._output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
      self._testMethodName)

  def testDo(self):
    custom_config = {'pouet': 12,
                     'blah': ['1', '2', '3'],
                     'plaff': 3.1415}

    self._output_configuration_dir = os.path.join(self._output_data_dir,
                                                  'output_examples')
    pipeline_configuration = artifacts.PipelineConfiguration()
    pipeline_configuration.uri = self._output_configuration_dir

    this_component = component.Exporter(custom_config=custom_config,
                                        pipeline_configuration=channel_utils.as_channel([pipeline_configuration]),
                                        instance_name=u'Testing123')
    self.assertEqual(artifacts.PipelineConfiguration.TYPE_NAME,
                     this_component.outputs[PIPELINE_CONFIGURATION_KEY].type_name)
    artifact_collection = this_component.outputs[PIPELINE_CONFIGURATION_KEY].get()
    self.assertIsNotNone(artifact_collection)

    self._input_dict = {
    }

    self._exec_properties = {
      # List needs to be serialized before being passed into Do function.
      CUSTOM_CONFIG_KEY: json_utils.dumps(custom_config)
    }

    self._output_dict = {
      PIPELINE_CONFIGURATION_KEY: [pipeline_configuration],
    }

    # Run executor.
    exporter = executor.Executor()
    exporter.Do(self._input_dict, self._output_dict,
                self._exec_properties)

    output_dir = artifact_utils.get_single_uri(artifact_collection)
    self.assertEqual(output_dir, self._output_configuration_dir)
    self.assertTrue(fileio.exists(os.path.join(output_dir, 'custom_config.json')))


if __name__ == '__main__':
  tf.test.main()
