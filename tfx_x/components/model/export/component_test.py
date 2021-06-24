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

import tensorflow as tf
from tfx.types import channel_utils
from tfx.types import standard_artifacts

from tfx_x.components.model.export import component
from tfx_x.components.model.export.executor import OUTPUT_KEY
from tfx_x import ExportedModel


def pouet(model, _pipeline_configuration):
  return model, None, None


class TransformTest(tf.test.TestCase):

  def setUp(self):
    super(TransformTest, self).setUp()
    self.name = 'HelloWorld'

  def testConstruct(self):
    model = standard_artifacts.Model()
    model_blessing = standard_artifacts.ModelBlessing()
    output = ExportedModel()
    this_component = component.Export(function_name='component_test.pouet', model=channel_utils.as_channel([model]),
                                      model_blessing=channel_utils.as_channel([model_blessing]),
                                      output=channel_utils.as_channel([output])).with_id(u'Testing123')
    self.assertEqual(ExportedModel.TYPE_NAME,
                     this_component.outputs[OUTPUT_KEY].type_name)
    artifact_collection = this_component.outputs[OUTPUT_KEY].get()
    self.assertIsNotNone(artifact_collection)


if __name__ == '__main__':
  tf.test.main()
