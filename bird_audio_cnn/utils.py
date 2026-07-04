
import matplotlib.pyplot as plt


def show_spectrogram(spec):

    plt.figure(figsize=(8,4))

    plt.imshow(
        spec.squeeze().numpy(),
        origin="lower",
        aspect="auto"
    )

    plt.colorbar()

    plt.title(
        "Mel Spectrogram"
    )

    plt.show()
