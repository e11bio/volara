import logging
from pathlib import Path

import click


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
def blockwise_worker(config_file: Path) -> None:
    import json
    from pathlib import Path

    from volara.blockwise import BlockwiseTask, BlockwiseTasks

    config_file = Path(config_file)
    config_json = json.loads(config_file.open("r").read())

    config = BlockwiseTasks.validate_python(config_json)
    assert isinstance(config, BlockwiseTask)
    config.process_blocks()
