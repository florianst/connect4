import click
import numpy as np
from matplotlib import pyplot as plt


@click.command()
@click.argument('path', type=click.Path(exists=True))
def run(path):
    data = np.loadtxt(path)

    fig = plt.figure(figsize=(3.25, 3))
    ax = fig.gca()

    ax.plot(
        data[:, 0],
        data[:, 2],
    )
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Winning rate (%)')
    ax.set_ylim(0, 100)

    fig.tight_layout()
    fig.savefig(path.replace('.txt', '.pgf'), dpi=320)
    fig.savefig(path.replace('.txt', '.png'), dpi=320)


if __name__ == '__main__':
    run()
