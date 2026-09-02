"""Inference-only replacement for MLSimKit's distributed-training launcher."""

from typing import Sequence

import click


def parse_args(args: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split MLSimKit arguments from arguments following ``--launch-args``."""
    command_args: list[str] = []
    launch_args: list[str] = []
    destination = command_args
    for arg in args:
        if arg == "--launch-args":
            destination = launch_args
        else:
            destination.append(arg)
    return launch_args, command_args


@click.command(context_settings={"ignore_unknown_options": True})
@click.option("--dry-run", is_flag=True, default=False)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def accelerate(dry_run: bool, args: Sequence[str]) -> None:
    """Describe or reject distributed training in an inference-only package."""
    launch_args, command_args = parse_args(args)
    planned_command = [
        "accelerate",
        "launch",
        "--no-python",
        *launch_args,
        "mlsimkit-learn",
        "--accelerate-mode",
        *command_args,
    ]
    if dry_run:
        click.echo(f"Would run command: {' '.join(planned_command)}")
        return
    raise click.ClickException(
        "Distributed training is unavailable in the inference-only agent package"
    )
