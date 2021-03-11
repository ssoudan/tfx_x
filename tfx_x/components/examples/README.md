# Examples

## Description

These components deal with transformation of Examples.

- `StratifiedSampler` does 'stratified sampling' on the input examples
- `Filter` filters the examples based on the provided predicate. 
- `Sample` - to come 

## Usage

```python
from tfx_x.components import StratifiedSampler
from tfx.orchestration import pipeline
from tfx.orchestration import metadata


def create_pipeline(...):
  # define the function that will create the key from an Example:
  to_key_fn = """
    def to_key(m):
      return 12
    """
  example_gen = ...
  ...
  # Create the stratified sampler
  stratified_sampler = StratifiedSampler(examples=example_gen.outputs['examples'],
                                         samples_per_key=123,
                                         splits_to_transform=['eval'],
                                         splits_to_copy=['train'],
                                         to_key_fn=to_key_fn)
  ...
  # Use its output
  my_custom_component = MyComponent(examples=stratified_sampler.outputs['filtered_examples'])
  ...
  return pipeline.Pipeline(
    pipeline_name=pipeline_name,
    pipeline_root=pipeline_root,
    components=[
      ...
      stratified_sampler,
      my_custom_component,
      ...
    ],
    enable_cache=True,
    metadata_connection_config=metadata.sqlite_metadata_connection_config(metadata_path),
    beam_pipeline_args=beam_pipeline_args)
```