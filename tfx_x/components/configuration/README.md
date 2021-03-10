# Configuration

## Description
These components convert, - and later will - export, transform and import configuration. We want to treat the pipeline 
configuration as an artifact and be able to pass it to our own components while having it stored immutably somewhere.

That way we can store the custom_config standard components use, but we can also create our own components that get 
their configuration from an artifact. For now we will start with a single artifact `PipelineConfiguration` but might 
want to create more in the future.

## Artifact

- `PipelineConfiguration` containing a `json.dumps()` of `custom_config` in `<uri>/custom_config.json`.

## Usage

See [README](../example/README.md) for a complete example.

    ...
    pipeline_configuration = FromCustomConfig(custom_config=custom_config) 
    ...
    my_custom_component = MySuperCustomComponent(config_exporter.outputs['pipeline_configuration'], ...)
    ...
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
          pipeline_configuration,
          my_custom_compenent,
          ...
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
        beam_pipeline_args=beam_pipeline_args)