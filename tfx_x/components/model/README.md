# Model

## Description

These components transform models.

## Artifact

- `ExportedModel` containing a `model.json` which describe the signature of the model being deployed and other things.

## Usage

# Create the stratified sampler

```python
def transform_fn(model, pipeline_configuration):
  # transform the model 

  signatures = {
    'serving_default': model.serve.get_concrete_function(),
  }

  options = tf.saved_model.SaveOptions(function_aliases={
    'my_func': func,
  })

  return model, signatures, options


...

transformer = Transform(input_model=...,
                        function_name='....transform_fn')
```

the function 'function_name' refers to, must be of type `(Kodel) -> (Model, Dict[Text, Any], SaveOptions)`.

# Export metadata on the model

```python

def export_fn(model, pipeline_configuration, output_dir):
  json.dumps({'something': 'else'}, os.path.join(output_dir, 'model.json'))


export = Export(model=...,
                model_blessing=...,
                infra_blessing=...,
                pope_blessing=...,
                function_name='....export_fn')

```