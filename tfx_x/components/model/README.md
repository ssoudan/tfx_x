# Model

## Description

These components transform models.

## Usage

# Create the stratified sampler

```python
  ...
transformer = Transform(input_model=...,
                        function_name=...)
```

the function 'function_name' refers to, must be of type `(model) -> model`.