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
    "-c", "--config-file", required=True, type=click.Path(exists=True, dir_okay=False)
)
def blockwise_worker(config_file: UPath) -> None:
    import json

    from volara.blockwise import BlockwiseTask, get_blockwise_tasks_type

    config_file = UPath(config_file)
    config_json = json.loads(config_file.open("r").read())

    BlockwiseTasks = get_blockwise_tasks_type()
    config = BlockwiseTasks.validate_python(config_json)
    assert isinstance(config, BlockwiseTask)
    config.process_blocks()
