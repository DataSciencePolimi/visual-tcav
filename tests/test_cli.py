"""
test_cli.py
-----------
Smoke tests for the CLI (visual_tcav/__main__.py).

Uses click.testing.CliRunner which runs the CLI in-process.

IMPORTANT: we import 'cli' (the click Group object), not 'main'
(which is just a plain Python wrapper function). CliRunner needs
a click Command or Group object, not a plain function.
"""

import pytest
from click.testing import CliRunner
from visual_tcav.__main__ import cli


@pytest.fixture
def runner():
    """click.testing.CliRunner — runs CLI commands in-process."""
    return CliRunner()


class TestCLIMain:
    """Tests for the top-level CLI entry point."""

    def test_help_exits_zero(self, runner):
        """visual-tcav --help exits with code 0."""
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize("subcommand", ["info", "local", "global"])
    def test_all_subcommands_listed_in_help(self, runner, subcommand):
        """All three subcommands appear in the main help output."""
        result = runner.invoke(cli, ["--help"])
        assert subcommand in result.output

    def test_no_args_exits_cleanly(self, runner):
        """visual-tcav with no arguments exits with code 0 or 2."""
        result = runner.invoke(cli, [])
        assert result.exit_code in (0, 2)


class TestCLIInfo:
    """Tests for the 'info' subcommand."""

    def test_help_shows_model_option(self, runner):
        """visual-tcav info --help shows --model option."""
        result = runner.invoke(cli, ["info", "--help"])
        assert result.exit_code == 0
        assert "--model" in result.output

    @pytest.mark.parametrize("model_name,expected_layer", [
        ("resnet50", "layer4"),
        ("resnet18", "layer4"),
        ("vgg16", "features"),
    ])
    def test_info_prints_layers(self, runner, model_name, expected_layer):
        """visual-tcav info --model <name> prints expected layer names."""
        result = runner.invoke(cli, ["info", "--model", model_name])
        assert result.exit_code == 0
        assert expected_layer in result.output

    def test_info_unknown_model_fails(self, runner):
        """visual-tcav info --model unknown exits with non-zero code."""
        result = runner.invoke(cli, ["info", "--model", "unknown_xyz"])
        assert result.exit_code != 0


class TestCLILocal:
    """Tests for the 'local' subcommand."""

    def test_help_shows_required_options(self, runner):
        """visual-tcav local --help shows all required options."""
        result = runner.invoke(cli, ["local", "--help"])
        assert result.exit_code == 0
        for option in ["--model", "--image", "--concepts", "--layers"]:
            assert option in result.output

    def test_missing_args_exits_with_error(self, runner):
        """visual-tcav local without required args exits with non-zero code."""
        result = runner.invoke(cli, ["local"])
        assert result.exit_code != 0


class TestCLIGlobal:
    """Tests for the 'global' subcommand."""

    def test_help_shows_required_options(self, runner):
        """visual-tcav global --help shows all required options."""
        result = runner.invoke(cli, ["global", "--help"])
        assert result.exit_code == 0
        for option in ["--model", "--images-dir", "--concepts", "--layers"]:
            assert option in result.output

    def test_missing_args_exits_with_error(self, runner):
        """visual-tcav global without required args exits with non-zero code."""
        result = runner.invoke(cli, ["global"])
        assert result.exit_code != 0