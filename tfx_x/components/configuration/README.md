# Configuration

## Description
These components export - and later will - transform and import configuration. We want to treat the pipeline 
configuration as an artifact.

That way we can store the custom_config standard components use, but we can also create our own components that get 
their configuration from an artifact.

## Artifact

- `PipelineConfiguration` containing a `json.dumps()` of `custom_config` in `<uri>/custom_config.json`.

## Usage

    ...
    config_exporter = Exporter(custom_config=custom_config) 
    ...
    my_custom_component = MySuperCustomComponent(config_exporter.outputs['pipeline_configuration'], ...)
    ...
    return pipeline.Pipeline(
        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=[
          config_exporter,
          my_custom_compenent,
          ...
        ],
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
          metadata_path),
        beam_pipeline_args=beam_pipeline_args)