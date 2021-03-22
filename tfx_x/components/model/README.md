# Model

## Description

These components transform models.

## Usage

# Create the stratified sampler

```python
def transform_fn(model):
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
