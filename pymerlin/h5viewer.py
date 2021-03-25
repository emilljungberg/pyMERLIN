from .dataIO import *
from .plot import plot_3plane
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="h5viewer", usage='h5viewer file.h5')
    parser.add_argument("h5file", metavar="H5", help="File input", type=str)

    args = parser.parse_args()
    f = args.h5file
    I, spacing = read_image_h5(f)

    print("Displaying %s" % f)
    plot_3plane(abs(I), title=f, cmap='gray', vmin=None, vmax=None)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()