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
from tfx.types import standard_artifacts, channel_utils
from tfx.types.component_spec import ChannelParameter
from tfx.types.component_spec import ExecutionParameter
from tfx.types.standard_component_specs import MODEL_BLESSING_KEY, INFRA_BLESSING_KEY, PUSHED_MODEL_KEY

from tfx_x import PipelineConfiguration
from tfx_x.components.model.export import executor
from tfx_x.components.model.export.executor import OUTPUT_KEY, MODEL_KEY, FUNCTION_NAME_KEY, \
  PIPELINE_CONFIGURATION_KEY
from tfx_x import ExportedModel


class ExportSpec(types.ComponentSpec):
  """ComponentSpec for model Export Component."""

  PARAMETERS = {
    FUNCTION_NAME_KEY: ExecutionParameter(type=Text),
  }
  INPUTS = {
    MODEL_KEY: ChannelParameter(type=standard_artifacts.Model),
    PIPELINE_CONFIGURATION_KEY: ChannelParameter(type=PipelineConfiguration, optional=True),
    MODEL_BLESSING_KEY: ChannelParameter(type=standard_artifacts.ModelBlessing, optional=True),
    INFRA_BLESSING_KEY: ChannelParameter(type=standard_artifacts.InfraBlessing, optional=True),
    PUSHED_MODEL_KEY: ChannelParameter(type=standard_artifacts.PushedModel, optional=True),
  }
  OUTPUTS = {
    OUTPUT_KEY: ChannelParameter(type=ExportedModel),
  }


class Export(base_component.BaseComponent):
  """Model Transformation TFX Component.

  """

  SPEC_CLASS = ExportSpec
  EXECUTOR_SPEC = executor_spec.ExecutorClassSpec(executor.Executor)

  def __init__(self,
               function_name: Text = None,
               model: types.Channel = None,
               model_blessing: Optional[types.Channel] = None,
               infra_blessing: Optional[types.Channel] = None,
               pushed_model: Optional[types.Channel] = None,
               output: types.Channel = None,
               pipeline_configuration: Optional[types.Channel] = None):
    """Construct a model export component.

    Args:
      function_name: The instance_name of the function to apply on the model.
      model: A Channel of type `standard_artifacts.Model`.
      model_blessing: A Channel of type `standard_artifacts.ModelBlessing`.
      infra_blessing: A Channel of type `standard_artifacts.InfraBlessing`.
      pushed_model: A Channel of type `standard_artifacts.PushedModel`.
      output: A Channel of type `ExportedModel`.
      pipeline_configuration: A Channel of 'PipelineConfiguration' type, usually produced by FromCustomConfig component.
    """

    if not output:
      output = channel_utils.as_channel([ExportedModel()])

    spec = ExportSpec(function_name=function_name,
                      pipeline_configuration=pipeline_configuration,
                      model=model,
                      model_blessing=model_blessing,
                      infra_blessing=infra_blessing,
                      pushed_model=pushed_model,
                      output=output)
    super(Export, self).__init__(spec=spec)
