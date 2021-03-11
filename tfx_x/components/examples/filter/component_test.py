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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tfx import types
from tfx.types import channel_utils
from tfx.types import standard_artifacts

from tfx_x.components.examples.filter.component import Filter
from tfx_x.components.examples.filter.executor import FILTERED_EXAMPLES_KEY
from tfx_x.types.artifacts import PipelineConfiguration


class ComponentTest(tf.test.TestCase):

  def testConstructNoPipelineConfiguration(self):
    examples = standard_artifacts.Examples()
    predicate_fn = """
  def predicate(m):
    return m.features.feature[key].float_list.value[0] > 0.5
"""

    filter = Filter(
      pipeline_configuration=None,
      examples=channel_utils.as_channel([examples]),
      filtered_examples=channel_utils.as_channel([standard_artifacts.Examples()]),
      splits_to_transform=['eval'],
      splits_to_copy=['train'],
      predicate_fn=predicate_fn)
    self.assertEqual('Examples', filter.outputs[FILTERED_EXAMPLES_KEY].type_name)

  def testConstructWithPipelineConfiguration(self):
    examples = standard_artifacts.Examples()
    filter = Filter(
      examples=channel_utils.as_channel([examples]),
      filtered_examples=channel_utils.as_channel([standard_artifacts.Examples()]),
      pipeline_configuration=types.Channel(type=PipelineConfiguration),
    )
    # filtered_examples=channel_utils.as_channel([examples]))
    self.assertEqual('Examples', filter.outputs[FILTERED_EXAMPLES_KEY].type_name)

  def testConstructHybridPipelineConfiguration(self):
    predicate_fn = """
  def predicate(m):
    return m.features.feature[key].float_list.value[0] > 0.5
    """

    examples = standard_artifacts.Examples()
    filter = Filter(
      examples=channel_utils.as_channel([examples]),
      pipeline_configuration=types.Channel(type=PipelineConfiguration),
      filtered_examples=channel_utils.as_channel([standard_artifacts.Examples()]),
      predicate_fn=predicate_fn)
    self.assertEqual('Examples', filter.outputs[FILTERED_EXAMPLES_KEY].type_name)

  def testConstructHybridPipelineConfigurationAndDifferentKeyFnKey(self):
    examples = standard_artifacts.Examples()
    filter = Filter(
      examples=channel_utils.as_channel([examples]),
      pipeline_configuration=types.Channel(type=PipelineConfiguration),
      filtered_examples=channel_utils.as_channel([standard_artifacts.Examples()]),
      predicate_fn_key='predicate_fn_key')
    self.assertEqual('Examples', filter.outputs[FILTERED_EXAMPLES_KEY].type_name)


if __name__ == '__main__':
  tf.test.main()
