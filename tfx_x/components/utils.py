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

import os

from absl import logging
import apache_beam as beam
import tensorflow as tf
from typing import Any, Dict, Mapping, List, Text

from tfx import types
from tfx.dsl.components.base import base_executor
from tfx.types import artifact_utils, Artifact
from tfx.utils import io_utils


def copy_over(input_artifact, output_artifact, splits_to_copy):
  """
  Copy data from specified splits
  Args:
    input_artifact: location where the input splits are
    output_artifact: location where to copy them
    splits_to_copy: list of split names to copy
  Returns:
    None
  """
  split_to_instance = {}

  for split in splits_to_copy:
    uri = artifact_utils.get_split_uri(input_artifact, split)
    split_to_instance[split] = uri

  for split, instance in split_to_instance.items():
    input_dir = instance
    output_dir = artifact_utils.get_split_uri([output_artifact], split)
    for filename in tf.io.gfile.listdir(input_dir):
      input_uri = os.path.join(input_dir, filename)
      output_uri = os.path.join(output_dir, filename)
      io_utils.copy_file(src=input_uri, dst=output_uri, overwrite=True)
