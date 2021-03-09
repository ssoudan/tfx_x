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

from tfx_x.components.examples.stratified_sampler import executor
from tfx_x.components.examples.stratified_sampler.executor import SPLITS_TO_TRANSFORM_KEY, THRESHOLD_KEY, \
  COUNT_PER_KEY_KEY, KEY_KEY, SPLITS_TO_COPY_KEY, SAMPLING_RESULT_KEY, EXAMPLES_KEY


class StratifiedSamplerSpec(ComponentSpec):
  """StratifiedSampler component spec."""

  PARAMETERS = {
    SPLITS_TO_TRANSFORM_KEY: ExecutionParameter(type=List[Text], optional=True),
    SPLITS_TO_COPY_KEY: ExecutionParameter(type=List[Text], optional=True),
    KEY_KEY: ExecutionParameter(type=Text),
    COUNT_PER_KEY_KEY: ExecutionParameter(type=int),
    THRESHOLD_KEY: ExecutionParameter(type=float, optional=True),
  }
  INPUTS = {
    EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }
  OUTPUTS = {
    SAMPLING_RESULT_KEY: ChannelParameter(type=standard_artifacts.Examples),
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
               key: Text,
               examples: types.Channel,
               sampling_result: Optional[types.Channel] = None,
               splits_to_transform: Optional[List[Text]] = None,
               splits_to_copy: Optional[List[Text]] = None,
               count_per_key: Optional[int] = None,
               threshold: Optional[float] = None,
               instance_name: Optional[Text] = None):
    """Construct an StratifiedSampler component.
    Args:
      examples: A Channel of 'ExamplesPath' type, usually produced by ExampleGen
        component. _required_
      sampling_result: Channel of `ExamplesPath` to store the inference
        results.
      splits_to_transform: Optional list of split names to transform.
      splits_to_copy: Optional list of split names to copy.
      instance_name: Optional name assigned to this specific instance of
        StratifiedSampler. Required only if multiple StratifiedSampler components are
        declared in the same pipeline.
    """
    sampling_result = sampling_result or types.Channel(
      type=standard_artifacts.Examples)
    spec = StratifiedSamplerSpec(
      examples=examples,
      sampling_result=sampling_result,
      splits_to_transform=splits_to_transform,
      splits_to_copy=splits_to_copy,
      key=key,
      count_per_key=count_per_key,
      threshold=threshold)
    super(StratifiedSampler, self).__init__(spec=spec, instance_name=instance_name)
