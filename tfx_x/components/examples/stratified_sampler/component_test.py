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
from tfx.types import channel_utils
from tfx.types import standard_artifacts

from tfx_x.components.examples.stratified_sampler.component import StratifiedSampler
from tfx_x.components.examples.stratified_sampler.executor import SAMPLING_RESULT_KEY


class ComponentTest(tf.test.TestCase):

  def testConstruct(self):
    examples = standard_artifacts.Examples()
    stratified_sampler = StratifiedSampler(
      examples=channel_utils.as_channel([examples]),
      splits_to_transform=['eval'],
      splits_to_copy=['train'],
      key='trip_miles',
      count_per_key=112)
    self.assertEqual('Examples', stratified_sampler.outputs[SAMPLING_RESULT_KEY].type_name)


if __name__ == '__main__':
  tf.test.main()
