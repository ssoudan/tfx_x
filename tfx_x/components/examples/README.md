# Examples

## Description
These components deal with transformation of Examples.

- `StratifiedSampler` does 'stratified sampling' on the input examples

## Usage

    ...
    stratified_sampler = StratifiedSampler(key='class_label', examples=example_gen.outputs['examples']) 
    ...
    my_custom_component = MySuperCustomComponent(examples=stratified_sampler.outputs['sampling_result'], ...)
    ...
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
          ...
          stratified_sampler,
          my_custom_compenent,
          ...
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
        beam_pipeline_args=beam_pipeline_args)