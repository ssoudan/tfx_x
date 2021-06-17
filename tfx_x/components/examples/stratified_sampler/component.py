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
"""Stratified sampling on a key (float) based on a threshold"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Text, List

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import ComponentSpec
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter, ExecutionParameter
from tfx.utils import json_utils

from tfx_x.components.examples.stratified_sampler import executor
from tfx_x.components.examples.stratified_sampler.executor import SPLITS_TO_TRANSFORM_KEY, \
  SAMPLES_PER_KEY_KEY, TO_KEY_FN_KEY, SPLITS_TO_COPY_KEY, STRATIFIED_EXAMPLES_KEY, EXAMPLES_KEY, \
  PIPELINE_CONFIGURATION_KEY, TO_KEY_FN_KEY_KEY
from tfx_x.types.artifacts import PipelineConfiguration


class StratifiedSamplerSpec(ComponentSpec):
  """StratifiedSampler component spec."""

  PARAMETERS = {
    SPLITS_TO_TRANSFORM_KEY: ExecutionParameter(type=(str, Text), optional=True),
    SPLITS_TO_COPY_KEY: ExecutionParameter(type=(str, Text), optional=True),
    TO_KEY_FN_KEY: ExecutionParameter(type=Text, optional=True),
    TO_KEY_FN_KEY_KEY: ExecutionParameter(type=Text, optional=True),
    SAMPLES_PER_KEY_KEY: ExecutionParameter(type=int, optional=True),
  }
  INPUTS = {
    EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
    PIPELINE_CONFIGURATION_KEY: ChannelParameter(type=PipelineConfiguration, optional=True),
  }
  OUTPUTS = {
    STRATIFIED_EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }


class StratifiedSampler(base_component.BaseComponent):
  """A TFX component to do stratified sampling.
  StratifiedSampler consumes examples data, and produces examples data
  
  ## Example
  ```
    # Uses StratifiedSampler to inference on examples.
    stratified_sampler = StratifiedSampler(
        key='trip_miles',
        examples=example_gen.outputs['examples'])
  ```
  """

  SPEC_CLASS = StratifiedSamplerSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               examples: types.Channel,
               to_key_fn: Optional[Text] = None,
               to_key_fn_key: Optional[Text] = 'to_key_fn',
               pipeline_configuration: Optional[types.Channel] = None,
               stratified_examples: Optional[types.Channel] = None,
               splits_to_transform: Optional[List[Text]] = None,
               splits_to_copy: Optional[List[Text]] = None,
               samples_per_key: Optional[int] = None):
    """Construct an StratifiedSampler component.
    Args:
      examples: A Channel of 'Examples' type, usually produced by ExampleGen
        component. _required_
      pipeline_configuration: A Channel of 'PipelineConfiguration' type, usually produced by FromCustomConfig
        component.
      stratified_examples: Channel of `Examples` to store the inference
        results.
      splits_to_transform: Optional list of split names to transform.
      splits_to_copy: Optional list of split names to copy.
      samples_per_key: Number of samples per key.
      to_key_fn_key: the name of the key that contains the to_key_fn - default is 'to_key_fn'.
      to_key_fn: To key function, the function that will extract the key - must be 'to_key: Example -> key
                 For example something like:
                 >>> def to_key(m):
                 >>>   return m.features.feature['trip_miles'].float_list.value[0] > 42.
    """
    stratified_examples = stratified_examples or types.Channel(
      type=standard_artifacts.Examples)

    if stratified_examples is None:
      stratified_examples = types.Channel(type=standard_artifacts.Examples, matching_channel_name='examples')

    spec = StratifiedSamplerSpec(
      examples=examples,
      pipeline_configuration=pipeline_configuration,
      stratified_examples=stratified_examples,
      splits_to_transform=json_utils.dumps(splits_to_transform),
      splits_to_copy=json_utils.dumps(splits_to_copy),
      to_key_fn=to_key_fn,
      to_key_fn_key=to_key_fn_key,
      samples_per_key=samples_per_key)
    super(StratifiedSampler, self).__init__(spec=spec)
