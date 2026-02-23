import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from hopfield import HopfieldNetwork
from patterns import ALL_PATTERNS, corrupt_pattern, GRID_SIZE


def state_to_grid(state):
    """Reshape a flat 1D state vector into a 2D grid for display."""
    return state.reshape(GRID_SIZE, GRID_SIZE)


def run(noise_level=0.3, steps=20, pattern_index=0):
    # --- Setup ---
    n_neurons = GRID_SIZE * GRID_SIZE
    net = HopfieldNetwork(n_neurons)
    net.train(ALL_PATTERNS)

    # Pick a pattern to recall, corrupt it, then run recall
    original = ALL_PATTERNS[pattern_index]
    noisy = corrupt_pattern(original, noise_level=noise_level)
    history = net.recall(noisy, steps=steps)

    # Compute energy at each step for the energy plot
    energies = [net.energy(s) for s in history]

    # --- Figure layout ---
    # Left column: original + noisy (static)
    # Middle: animated recall frames
    # Right: energy over time
    fig = plt.figure(figsize=(12, 4))
    fig.suptitle("Hopfield Network Recall", fontsize=13)

    ax_orig   = fig.add_subplot(1, 4, 1)
    ax_noisy  = fig.add_subplot(1, 4, 2)
    ax_recall = fig.add_subplot(1, 4, 3)
    ax_energy = fig.add_subplot(1, 4, 4)

    # Static: original pattern
    ax_orig.imshow(state_to_grid(original), cmap="gray", vmin=-1, vmax=1)
    ax_orig.set_title("Original")
    ax_orig.axis("off")

    # Static: corrupted input
    ax_noisy.imshow(state_to_grid(noisy), cmap="gray", vmin=-1, vmax=1)
    ax_noisy.set_title(f"Noisy ({int(noise_level*100)}% flipped)")
    ax_noisy.axis("off")

    # Animated: recall state at each step
    recall_img = ax_recall.imshow(state_to_grid(history[0]), cmap="gray", vmin=-1, vmax=1)
    recall_title = ax_recall.set_title("Recall — step 0")
    ax_recall.axis("off")

    # Energy plot: draw the full curve, then animate a moving dot
    ax_energy.plot(energies, color="steelblue", linewidth=2)
    energy_dot, = ax_energy.plot(0, energies[0], "ro", markersize=8)
    ax_energy.set_title("Energy")
    ax_energy.set_xlabel("Step")
    ax_energy.set_ylabel("E")

    def animate(frame):
        recall_img.set_data(state_to_grid(history[frame]))
        recall_title.set_text(f"Recall — step {frame}")
        energy_dot.set_data([frame], [energies[frame]])
        return recall_img, recall_title, energy_dot

    ani = animation.FuncAnimation(
        fig,
        animate,
        frames=len(history),
        interval=400,       # ms between frames
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Change pattern_index to 1 to test recall of pattern_O instead
    run(noise_level=0.3, steps=20, pattern_index=0)
