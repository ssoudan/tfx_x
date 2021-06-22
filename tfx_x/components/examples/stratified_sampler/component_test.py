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

from tfx_x.components.examples.stratified_sampler.component import StratifiedSampler
from tfx_x.components.examples.stratified_sampler.executor import STRATIFIED_EXAMPLES_KEY
from tfx_x import PipelineConfiguration


class ComponentTest(tf.test.TestCase):

  def testConstructNoPipelineConfiguration(self):
    examples = standard_artifacts.Examples()
    to_key_fn = """
  def to_key(m):
    return m.features.feature[key].float_list.value[0] > 0.5
"""

    stratified_sampler = StratifiedSampler(
      pipeline_configuration=None,
      examples=channel_utils.as_channel([examples]),
      stratified_examples=channel_utils.as_channel([standard_artifacts.Examples()]),
      splits_to_transform=['eval'],
      splits_to_copy=['train'],
      to_key_fn=to_key_fn,
      samples_per_key=112)
    self.assertEqual('Examples', stratified_sampler.outputs[STRATIFIED_EXAMPLES_KEY].type_name)

  def testConstructWithPipelineConfiguration(self):
    examples = standard_artifacts.Examples()
    stratified_sampler = StratifiedSampler(
      examples=channel_utils.as_channel([examples]),
      stratified_examples=channel_utils.as_channel([standard_artifacts.Examples()]),
      pipeline_configuration=types.Channel(type=PipelineConfiguration),
    )
    # stratified_examples=channel_utils.as_channel([examples]))
    self.assertEqual('Examples', stratified_sampler.outputs[STRATIFIED_EXAMPLES_KEY].type_name)

  def testConstructHybridPipelineConfiguration(self):
    to_key_fn = """
      def to_key(m):
        return m.features.feature[key].float_list.value[0] > 0.5
    """

    examples = standard_artifacts.Examples()
    stratified_sampler = StratifiedSampler(
      examples=channel_utils.as_channel([examples]),
      pipeline_configuration=types.Channel(type=PipelineConfiguration),
      stratified_examples=channel_utils.as_channel([standard_artifacts.Examples()]),
      to_key_fn=to_key_fn,
      samples_per_key=112)
    self.assertEqual('Examples', stratified_sampler.outputs[STRATIFIED_EXAMPLES_KEY].type_name)

  def testConstructHybridPipelineConfigurationAndDifferentKeyFnKey(self):
    examples = standard_artifacts.Examples()
    stratified_sampler = StratifiedSampler(
      examples=channel_utils.as_channel([examples]),
      pipeline_configuration=types.Channel(type=PipelineConfiguration),
      stratified_examples=channel_utils.as_channel([standard_artifacts.Examples()]),
      to_key_fn_key='to_key_fn_key',
      samples_per_key=112)
    self.assertEqual('Examples', stratified_sampler.outputs[STRATIFIED_EXAMPLES_KEY].type_name)


if __name__ == '__main__':
  tf.test.main()
