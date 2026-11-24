"""Единый стиль графиков и сохранение в results/figures."""
import matplotlib.pyplot as plt

# Категориальная палитра в фиксированном порядке: цвет закреплён за сущностью
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLORS = {
    "Dummy": MUTED,
    "LogReg": BLUE,
    "sklearn MLP": ORANGE,
    "Numpy MLP": AQUA,
    "PyTorch MLP": YELLOW,
}


def setup_style():
    """Тонкие подписи, бледная сетка, без лишних рамок."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "lines.linewidth": 2,
            "legend.frameon": False,
        }
    )


def save(fig, figures_dir, name):
    """Сохраняет график и возвращает путь, чтобы было видно, где он лежит."""
    path = figures_dir / f"{name}.png"
    fig.savefig(path)
    return path
