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
"""Tests for FromCustomConfig."""
import os

import tensorflow as tf
from tfx.types import channel_utils

from tfx_x import PipelineConfiguration
from tfx_x.components.configuration.converter import component
from tfx_x.components.configuration.converter.executor import PIPELINE_CONFIGURATION_KEY


class ExportTest(tf.test.TestCase):

  def setUp(self):
    super(ExportTest, self).setUp()
    self.name = 'HelloWorld'
    self._output_data_dir = os.path.join(
      os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR', self.get_temp_dir()),
      self._testMethodName)

  def testConstruct(self):
    custom_config = {'pouet': 12,
                     'blah': ['1', '2', '3'],
                     'plaff': 3.1415}

    self._output_configuration_dir = os.path.join(self._output_data_dir,
                                                  'output_examples')
    pipeline_configuration = PipelineConfiguration()
    pipeline_configuration.uri = self._output_configuration_dir

    this_component = component.FromCustomConfig(custom_config=custom_config,
                                                pipeline_configuration=channel_utils.as_channel(
                                                  [pipeline_configuration])).with_id(u'Testing123')
    self.assertEqual(PipelineConfiguration.TYPE_NAME,
                     this_component.outputs[PIPELINE_CONFIGURATION_KEY].type_name)
    artifact_collection = this_component.outputs[PIPELINE_CONFIGURATION_KEY].get()
    self.assertIsNotNone(artifact_collection)


if __name__ == '__main__':
  tf.test.main()
