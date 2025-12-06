import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def plot_2d_ellipsoid_boundary(mean, covariance, ax=None, color='blue', linewidth=0.5):
    """
    Plot the boundary of a 2D ellipsoid given the mean and covariance matrix.

    Parameters:
    - mean (array-like): The mean of the distribution (2D).
    - covariance (array-like): The covariance matrix (2x2).
    - ax (matplotlib.axes._axes.Axes, optional): Axes on which to plot the ellipsoid.
    - color (str, optional): Color of the ellipsoid boundary.

    Returns:
    - None
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Calculate eigenvalues and eigenvectors of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # Sort eigenvalues and corresponding eigenvectors in descending order
    order = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    
    nstd = 3
    # print("eigenvalues: ", eigenvalues)
    width, height = 2 * nstd * np.sqrt(eigenvalues)

    # Calculate the angle of rotation
    angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))

    # Plot the ellipsoid boundary
    ellipse_boundary = Ellipse(xy=mean, width=width, height=height,
                               angle=angle, fill=False, edgecolor=color, linewidth=linewidth)

    # Add the Ellipse boundary to the plot
    ax.add_patch(ellipse_boundary)

    return ellipse_boundary, ax

if __name__ == '__main__':
    # Example usage:
    mean = np.array([2, 3])  # Mean of the distribution
    covariance = np.array([[3, 1], [1, 2]])  # Covariance matrix

    # Create a plot
    fig, ax = plt.subplots()

    # Plot the boundary of the 2D ellipsoid
    ellipse_boundary, ax = plot_2d_ellipsoid_boundary(mean, covariance, ax)

    # Set axis limits
    ax.set_xlim(mean[0] - 5, mean[0] + 5)
    ax.set_ylim(mean[1] - 5, mean[1] + 5)

    # Show the plot
    plt.show()