import logging

import click
from upath import UPath


@click.group()
@click.option(
    "--log-level",
    type=click.Choice(
        ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], case_sensitive=False
    ),
    default="INFO",
)
def cli(log_level: str) -> None:
    logging.basicConfig(level=getattr(logging, log_level.upper()))


@cli.command()
@click.option(
    "-c",
    "--config-file",
    required=True,
    # NOT ``click.Path(exists=True)``: that stats the value on the local filesystem, so a
    # config written beside a remote basedir ("s3://bucket/logs/<task>-meta/config.json",
    # which is exactly what ``Worker.get_command`` passes) is rejected as nonexistent
    # before the worker starts. A missing file still fails loudly, at the read below.
    type=str,
)
def blockwise_worker(config_file: str) -> None:
    import json

    from volara.blockwise import BlockwiseTask, get_blockwise_tasks_type

    config_path = UPath(config_file)
    config_json = json.loads(config_path.read_text())

    BlockwiseTasks = get_blockwise_tasks_type()
    config = BlockwiseTasks.validate_python(config_json)
    assert isinstance(config, BlockwiseTask)
    config.process_blocks()
