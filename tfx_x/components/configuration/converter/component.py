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
"""
PipelineConfiguration converter component
"""

from typing import Optional, Text, Dict, Any

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.utils import json_utils

from tfx_x import PipelineConfiguration
from tfx_x.components.configuration.converter import executor
from tfx_x.components.configuration.converter.executor import CUSTOM_CONFIG_KEY, PIPELINE_CONFIGURATION_KEY



class FromCustomConfigSpec(types.ComponentSpec):
  """ComponentSpec configuration converter component."""

  PARAMETERS = {
    # These are parameters that will be passed in the call to
    # create an instance of this component.
    CUSTOM_CONFIG_KEY: ExecutionParameter(type=(str, Text), optional=True),
  }
  INPUTS = {
  }
  OUTPUTS = {
    # This will be a dictionary which this component will populate
    PIPELINE_CONFIGURATION_KEY: ChannelParameter(type=PipelineConfiguration),
  }


class FromCustomConfig(base_component.BaseComponent):
  """FromCustomConfig configuration.

  This custom component class consists of only a constructor.
  """

  SPEC_CLASS = FromCustomConfigSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               custom_config: Optional[Dict[Text, Any]] = None,
               pipeline_configuration: types.Channel = None):
    """Construct a pipeline configuration converter component.

    Args:
      pipeline_configuration: A Channel of type `artifacts.PipelineConfiguration`.
      custom_config: The configuration.
    """
    if not pipeline_configuration:
      pipeline_configuration = channel_utils.as_channel([PipelineConfiguration()])

    if not custom_config:
      custom_config = {}

    spec = FromCustomConfigSpec(custom_config=json_utils.dumps(custom_config),
                                pipeline_configuration=pipeline_configuration)
    super(FromCustomConfig, self).__init__(spec=spec)
