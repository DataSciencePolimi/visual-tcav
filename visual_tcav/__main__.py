"""
__main__.py
-----------
Command-line interface for visual-tcav.

Allows running Visual-TCAV explanations directly from the terminal
without writing any Python code. Useful for batch processing,
scripting, and running on remote servers or supercomputers.

Usage
-----
    # Local explanation (single image)
    visual-tcav local \\
        --model resnet50 \\
        --image ./zebra.jpg \\
        --concepts striped dotted \\
        --concept-dir ./concept_images \\
        --layers layer4 \\
        --output ./results

    # Global explanation (folder of images)
    visual-tcav global \\
        --model resnet50 \\
        --images-dir ./test_images/zebra \\
        --concepts striped dotted \\
        --concept-dir ./concept_images \\
        --layers layer4 \\
        --output ./results

    # Show available layers for a model
    visual-tcav info --model resnet50

Install
-------
    pip install visual-tcav
    # Then run: visual-tcav --help
"""

import sys
import os

import click

sys.dont_write_bytecode = True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

@click.group()
@click.version_option()
def cli():
    """
    visual-tcav: Concept-based Attribution and Saliency Maps for XAI in PyTorch.

    Run Visual-TCAV explanations from the command line.
    Use --help on any subcommand for details.
    """
    pass


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--model", "-m",
    required=True,
    type=str,
    help=(
        "Model name (e.g. resnet50) or path to a saved .pt model file. "
        "Supported names: resnet18, resnet50, resnet101, vgg16, vgg19."
    ),
)
def info(model):
    """
    Show available CNN layers for a model.

    Use the layer names shown here with the --layers option
    in the local and global subcommands.

    Example:

        visual-tcav info --model resnet50
    """
    from visual_tcav import TorchModelWrapper

    click.echo(f"Loading model: {model}")

    if os.path.isfile(model):
        wrapper = TorchModelWrapper(
            model_name=os.path.splitext(os.path.basename(model))[0],
            model_path=model,
        )
    else:
        # String-based loading via _build_wrapper
        from visual_tcav.visual_tcav import _build_wrapper
        wrapper = _build_wrapper(model, model_name=model)

    wrapper.info()


# ---------------------------------------------------------------------------
# Subcommand: local
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--model", "-m",
    required=True,
    type=str,
    help=(
        "Model name (e.g. resnet50) or path to a saved .pt model file."
    ),
)
@click.option(
    "--image", "-i",
    required=True,
    type=click.Path(exists=True),
    help="Path to the test image to explain.",
)
@click.option(
    "--concepts", "-c",
    required=True,
    multiple=True,
    help=(
        "Concept names to analyze. Repeat for multiple concepts: "
        "--concepts striped --concepts dotted"
    ),
)
@click.option(
    "--concept-dir", "-cd",
    required=True,
    type=click.Path(exists=True),
    help=(
        "Path to the concept images folder. Must contain one subfolder "
        "per concept and a 'random' subfolder for random images."
    ),
)
@click.option(
    "--random-dir", "-rd",
    default=None,
    type=click.Path(),
    help=(
        "Path to the random images folder. "
        "Defaults to <concept-dir>/random/ if not provided."
    ),
)
@click.option(
    "--layers", "-l",
    required=True,
    multiple=True,
    help=(
        "CNN layer names to analyze. Repeat for multiple layers: "
        "--layers layer3 --layers layer4. "
        "Run 'visual-tcav info --model <name>' to see available layers."
    ),
)
@click.option(
    "--n-classes",
    default=3,
    show_default=True,
    type=int,
    help="Number of top predicted classes to explain.",
)
@click.option(
    "--m-steps",
    default=50,
    show_default=True,
    type=int,
    help="Interpolation steps for Integrated Gradients. Higher = more accurate but slower.",
)
@click.option(
    "--output", "-o",
    default="./results",
    show_default=True,
    type=click.Path(),
    help="Output directory for saving the explanation figure.",
)
@click.option(
    "--cache-dir",
    default=".cache",
    show_default=True,
    type=click.Path(),
    help="Directory for caching CAVs and random activations.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching. Forces recomputation of all CAVs and activations.",
)
def local(model, image, concepts, concept_dir, random_dir, layers,
          n_classes, m_steps, output, cache_dir, no_cache):
    """
    Explain a single image using LocalVisualTCAV.

    Produces a concept map (WHERE the CNN detected each concept)
    and attribution scores (HOW MUCH each concept influenced the prediction).

    Example:

        visual-tcav local \\
            --model resnet50 \\
            --image ./zebra.jpg \\
            --concepts striped --concepts dotted \\
            --concept-dir ./concept_images \\
            --layers layer4 \\
            --output ./results
    """
    from visual_tcav import LocalVisualTCAV

    os.makedirs(output, exist_ok=True)

    image_name = os.path.splitext(os.path.basename(image))[0]
    save_path = os.path.join(output, f"{image_name}_local_explanation.png")

    click.echo(f"\nRunning LocalVisualTCAV...")
    click.echo(f"  Model:    {model}")
    click.echo(f"  Image:    {image}")
    click.echo(f"  Concepts: {list(concepts)}")
    click.echo(f"  Layers:   {list(layers)}")
    click.echo(f"  Output:   {save_path}\n")

    use_cache = not no_cache

    tcav = LocalVisualTCAV(
        model=model,
        test_image_path=image,
        concept_names=list(concepts),
        concept_base_dir=concept_dir,
        random_dir=random_dir,
        layer_names=list(layers),
        n_classes=n_classes,
        m_steps=m_steps,
        cache_dir=cache_dir,
    )

    tcav.predict().info()
    tcav.explain(cache_cav=use_cache, cache_random=use_cache)
    tcav.plot(save_path=save_path)

    click.echo(f"\nDone. Figure saved to: {save_path}")


# ---------------------------------------------------------------------------
# Subcommand: global
# ---------------------------------------------------------------------------

@cli.command(name="global")
@click.option(
    "--model", "-m",
    required=True,
    type=str,
    help="Model name (e.g. resnet50) or path to a saved .pt model file.",
)
@click.option(
    "--images-dir", "-id",
    required=True,
    type=click.Path(exists=True),
    help="Path to a folder containing test images of ONE class.",
)
@click.option(
    "--concepts", "-c",
    required=True,
    multiple=True,
    help="Concept names to analyze.",
)
@click.option(
    "--concept-dir", "-cd",
    required=True,
    type=click.Path(exists=True),
    help=(
        "Path to the concept images folder. Must contain one subfolder "
        "per concept and a 'random' subfolder for random images."
    ),
)
@click.option(
    "--random-dir", "-rd",
    default=None,
    type=click.Path(),
    help=(
        "Path to the random images folder. "
        "Defaults to <concept-dir>/random/ if not provided."
    ),
)
@click.option(
    "--layers", "-l",
    required=True,
    multiple=True,
    help="CNN layer names to analyze.",
)
@click.option(
    "--n-classes",
    default=3,
    show_default=True,
    type=int,
    help="Number of top predicted classes to explain.",
)
@click.option(
    "--m-steps",
    default=50,
    show_default=True,
    type=int,
    help="Interpolation steps for Integrated Gradients.",
)
@click.option(
    "--max-images",
    default=50,
    show_default=True,
    type=int,
    help="Maximum number of test images to process.",
)
@click.option(
    "--output", "-o",
    default="./results",
    show_default=True,
    type=click.Path(),
    help="Output directory for saving figures and the stats table.",
)
@click.option(
    "--cache-dir",
    default=".cache",
    show_default=True,
    type=click.Path(),
    help="Directory for caching CAVs and random activations.",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Disable caching. Forces recomputation of all CAVs and activations.",
)
def global_cmd(model, images_dir, concepts, concept_dir, random_dir, layers,
               n_classes, m_steps, max_images, output, cache_dir, no_cache):
    """
    Explain a class of images using GlobalVisualTCAV.

    Runs the pipeline on a folder of images and computes attribution statistics
    (mean, standard deviation, 95% confidence interval) across all images.

    Example:

        visual-tcav global \\
            --model resnet50 \\
            --images-dir ./test_images/zebra \\
            --concepts striped --concepts dotted \\
            --concept-dir ./concept_images \\
            --layers layer4 \\
            --output ./results
    """
    from visual_tcav import GlobalVisualTCAV

    os.makedirs(output, exist_ok=True)

    folder_name = os.path.basename(images_dir.rstrip("/\\"))
    plot_path = os.path.join(output, f"{folder_name}_global_explanation.png")
    stats_path = os.path.join(output, f"{folder_name}_stats.txt")

    click.echo(f"\nRunning GlobalVisualTCAV...")
    click.echo(f"  Model:      {model}")
    click.echo(f"  Images dir: {images_dir}")
    click.echo(f"  Concepts:   {list(concepts)}")
    click.echo(f"  Layers:     {list(layers)}")
    click.echo(f"  Max images: {max_images}")
    click.echo(f"  Output:     {output}\n")

    use_cache = not no_cache

    tcav = GlobalVisualTCAV(
        model=model,
        test_images_dir=images_dir,
        concept_names=list(concepts),
        concept_base_dir=concept_dir,
        random_dir=random_dir,
        layer_names=list(layers),
        n_classes=n_classes,
        m_steps=m_steps,
        max_test_images=max_images,
        cache_dir=cache_dir,
    )

    tcav.explain(cache_cav=use_cache, cache_random=use_cache)
    tcav.statsInfo()
    tcav.plot(save_path=plot_path)

    click.echo(f"\nDone. Figure saved to: {plot_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    """Entry point registered in pyproject.toml as 'visual-tcav'."""
    cli()


if __name__ == "__main__":
    main()