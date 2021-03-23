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

from typing import Optional, Text

from tfx import types
from tfx.dsl.components.base import base_component
from tfx.dsl.components.base import executor_spec
from tfx.types import channel_utils
from tfx.types import standard_artifacts
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter

from tfx_x import PipelineConfiguration
from tfx_x.components.model.transform import executor
from tfx_x.components.model.transform.executor import OUTPUT_MODEL_KEY, INPUT_MODEL_KEY, FUNCTION_NAME_KEY, \
  PIPELINE_CONFIGURATION_KEY


class TransformSpec(types.ComponentSpec):
  """ComponentSpec for model Transform Component."""

  PARAMETERS = {
    FUNCTION_NAME_KEY: ExecutionParameter(type=Text),
  }
  INPUTS = {
    INPUT_MODEL_KEY: ChannelParameter(type=standard_artifacts.Model),
    PIPELINE_CONFIGURATION_KEY: ChannelParameter(type=PipelineConfiguration, optional=True),
  }
  OUTPUTS = {
    OUTPUT_MODEL_KEY: ChannelParameter(type=standard_artifacts.Model),
  }


class Transform(base_component.BaseComponent):
  """Model Transformation TFX Component.

  """

  SPEC_CLASS = TransformSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               function_name: Text = None,
               input_model: types.Channel = None,
               output_model: types.Channel = None,
               pipeline_configuration: Optional[types.Channel] = None,
               instance_name: Optional[Text] = None):
    """Construct a model transformation component.

    Args:
      function_name: The instance_name of the function to apply on the model.
      input_model: A Channel of type `standard_artifacts.Model`.
      pipeline_configuration: A Channel of 'PipelineConfiguration' type, usually produced by FromCustomConfig component.
      output_model: A Channel of type `standard_artifacts.Model`.
      instance_name: The instance_name of the instance - Optional.
    """

    if not output_model:
      output_model = channel_utils.as_channel([standard_artifacts.Model()])

    spec = TransformSpec(function_name=function_name,
                         pipeline_configuration=pipeline_configuration,
                         input_model=input_model,
                         output_model=output_model,
                         instance_name=instance_name)
    super(Transform, self).__init__(spec=spec, instance_name=instance_name)
