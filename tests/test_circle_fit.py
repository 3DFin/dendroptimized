import numpy as np

from dendroptimized import circle_fit


def generate_circle_points(center=(0, 0), radius=1.0, n_points=50, noise=0.0):
    """
    Generate 2D points randomly distributed around a circle.

    Parameters
    ----------
    center : tuple of float
        (x, y) coordinates of the circle center.
    radius : float
        Radius of the circle.
    num_points : int
        Number of points to generate.
    noise : float
        Standard deviation of gaussian noise added to the points.

    Returns
    -------
    points : ndarray of shape (N, 2)
        The generated point cloud.
    """
    rng = np.random.default_rng(1337)

    angles = 2 * np.pi * rng.uniform(size=n_points)
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)

    if noise > 0:
        x += rng.normal(0, noise, size=n_points)
        y += rng.normal(0, noise, size=n_points)

    return np.column_stack((x, y))


def test_circles():
   circle_cloud = generate_circle_points(center=(1, 2), radius=0.3)
   circle_params = circle_fit(circle_cloud)
   np.testing.assert_equal(circle_params, [1, 2, 0.3])
