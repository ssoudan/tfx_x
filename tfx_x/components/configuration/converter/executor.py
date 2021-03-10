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
"""Executor for pipeline configuration converter"""

import os
from typing import Any, Dict, List, Text

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils
from tfx.utils import io_utils

CUSTOM_CONFIG_KEY = 'custom_config'
PIPELINE_CONFIGURATION_KEY = 'pipeline_configuration'


class Executor(base_executor.BaseExecutor):
  """Executor for FromCustomConfig."""

  def Do(self,
         input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Stores `custom_config` as an artifact of type `artifacts.PipelineConfiguration`.

    Args:
      input_dict: Empty
      output_dict: Output dict from key to a list of artifacts, including:
        - pipeline_configuration: A list of type `artifacts.PipelineConfiguration`
      exec_properties: A dict of execution properties, including:
        - custom_config: the configuration to save.
        - instance_name: Optional unique name. Necessary iff multiple FromCustomConfig components
          are declared in the same pipeline.

    Returns:
      None

    Raises:
      OSError and its subclasses
      ValueError
    """
    self._log_startup(input_dict, output_dict, exec_properties)

    pipeline_configuration = artifact_utils.get_single_instance(output_dict[PIPELINE_CONFIGURATION_KEY])
    custom_config = exec_properties.get(CUSTOM_CONFIG_KEY, "{}")

    output_dir = artifact_utils.get_single_uri([pipeline_configuration])
    output_file = os.path.join(output_dir, 'custom_config.json')

    io_utils.write_string_file(output_file, custom_config)
