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

from tfx_x.components.examples.filter import executor
from tfx_x.components.examples.filter.executor import SPLITS_TO_TRANSFORM_KEY, \
  PREDICATE_FN_KEY, SPLITS_TO_COPY_KEY, FILTERED_EXAMPLES_KEY, EXAMPLES_KEY, \
  PIPELINE_CONFIGURATION_KEY, PREDICATE_FN_KEY_KEY
from tfx_x import PipelineConfiguration


class FilterSpec(ComponentSpec):
  """Filter component spec."""

  PARAMETERS = {
    SPLITS_TO_TRANSFORM_KEY: ExecutionParameter(type=(str, Text), optional=True),
    SPLITS_TO_COPY_KEY: ExecutionParameter(type=(str, Text), optional=True),
    PREDICATE_FN_KEY: ExecutionParameter(type=Text, optional=True),
    PREDICATE_FN_KEY_KEY: ExecutionParameter(type=Text, optional=True),
  }
  INPUTS = {
    EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
    PIPELINE_CONFIGURATION_KEY: ChannelParameter(type=PipelineConfiguration, optional=True),
  }
  OUTPUTS = {
    FILTERED_EXAMPLES_KEY: ChannelParameter(type=standard_artifacts.Examples),
  }


class Filter(base_component.BaseComponent):
  """A TFX component to do filtering.
  Filter consumes examples data, and produces examples data
  
  ## Example
    # Uses Filter to inference on examples.
    >>> filter = Filter(
    >>>    predicate_fn="def predicate(m):...",
    >>>    examples=example_gen.outputs['examples'])

  """

  SPEC_CLASS = FilterSpec
  EXECUTOR_SPEC = executor_spec.BeamExecutorSpec(executor.Executor)

  def __init__(self,
               examples: types.Channel,
               predicate_fn: Optional[Text] = None,
               predicate_fn_key: Optional[Text] = 'predicate_fn',
               pipeline_configuration: Optional[types.Channel] = None,
               filtered_examples: Optional[types.Channel] = None,
               splits_to_transform: Optional[List[Text]] = None,
               splits_to_copy: Optional[List[Text]] = None):
    """Construct an Filter component.
    Args:
      examples: A Channel of 'Examples' type, usually produced by ExampleGen
        component. _required_
      pipeline_configuration: A Channel of 'PipelineConfiguration' type, usually produced by FromCustomConfig
        component.
      filtered_examples: Channel of `Examples` to store the inference
        results.
      splits_to_transform: Optional list of split names to transform.
      splits_to_copy: Optional list of split names to copy.
      predicate_fn_key: the name of the key that contains the predicate_fn - default is 'predicate_fn'.
      predicate_fn: To key function, the function that will tell if a example must be kept.
                 Must be 'predicate: Example -> bool. For example something like:
                 >>> def predicate(m):
                       return m.features.feature['trip_miles'].float_list.value[0] > 42.
    """
    filtered_examples = filtered_examples or types.Channel(
      type=standard_artifacts.Examples)

    if filtered_examples is None:
      filtered_examples = types.Channel(type=standard_artifacts.Examples, matching_channel_name='examples')

    spec = FilterSpec(
      examples=examples,
      pipeline_configuration=pipeline_configuration,
      filtered_examples=filtered_examples,
      splits_to_transform=json_utils.dumps(splits_to_transform),
      splits_to_copy=json_utils.dumps(splits_to_copy),
      predicate_fn=predicate_fn,
      predicate_fn_key=predicate_fn_key)
    super(Filter, self).__init__(spec=spec)
