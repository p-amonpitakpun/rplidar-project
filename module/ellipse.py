import matplotlib.pyplot as plt
import numpy as np


def rotate_matrix(x):
    cos_x = np.cos(x)
    sin_x = np.sin(x)
    return np.array([[cos_x, -sin_x], [sin_x, cos_x]])


def test_ellipse():

    # known attributes
    center = np.array([0, 0])
    phi = np.pi / 6 * 0
    a = 1
    b = 0.5
    c = np.sqrt(a**2 - b**2)

    # vectors
    matrix_R = rotate_matrix(phi)
    vector_A = np.matmul(matrix_R, np.array([a, 0])) + center
    print('A =', vector_A)
    vector_B = np.matmul(matrix_R, np.array([0, b])) + center
    print('B =', vector_B)
    vector_F1 = center + np.matmul(matrix_R, np.array([c, 0]))
    vector_F2 = center - np.matmul(matrix_R, np.array([c, 0]))
    print('C =', vector_F1, vector_F2)

    # plot
    x_0, y_0 = center
    plt.plot([x_0, vector_A[0]], [y_0, vector_A[1]], c='r', linewidth=1)
    plt.plot([x_0, vector_B[0]], [y_0, vector_B[1]], c='g', linewidth=1)
    # plt.plot([x_0, vector_F1[0]], [y_0, vector_F1[1]], c='b', linewidth=1)
    # plt.plot([x_0, vector_F2[0]], [y_0, vector_F2[1]], c='b', linewidth=1)
    plt.scatter([x_0], [y_0], c='k', s=1)

    # trigon func
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # constant variables
    A = (cos_phi**2) / (a**2) + (sin_phi**2) / (b**2)
    B = 2 * cos_phi * sin_phi * (1 / (a**2) - 1 / (b**2))
    C = (sin_phi**2) / (a**2) + (cos_phi**2) / (b**2)

    print('A =', A)
    print('B =', B)
    print('C =', C)

    for theta in np.linspace(0, 2 * np.pi, 1000):
        print('theta =', theta)

        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        H_0 = A * (x_0**2) + (B * x_0 * y_0) + C * (y_0**2)
        H_x = -(2 * A * x_0) - (B * y_0)
        H_y = -(2 * C * y_0) - (B * x_0)
        print('H =', H_0, H_x, H_y)

        A_r = A * sin_theta**2 + B * sin_theta * cos_theta + C * cos_theta**2
        B_r = H_x * sin_theta + H_y * cos_theta
        C_r = H_0
        print('const =', A_r, B_r, C_r)
        print('B^2 - 4AC =', B_r**2 - 4 * A_r * C_r)

        if (B_r**2 - 4 * A_r * C_r) < 0:
            R = [-B_r / (2 * A_r)]
        elif B_r == 0:
            R = [np.sqrt((1 - C_r) / A_r), -np.sqrt((1 - C_r) / A_r)]
        elif A_r != 0:
            R = [(-B_r + np.sqrt(B_r**2 - 4 * A_r * C_r)) / (2 * A_r),
                 (-B_r - np.sqrt(B_r**2 - 4 * A_r * C_r)) / (2 * A_r)]
        else:
            R = [(1 - C_r) / B_r]
        print('answer =', R)
        print()

        X = np.array([[r * sin_theta, r * cos_theta] for r in R])
        plt.scatter(X[:, 0], X[:, 1], s=1, c='cyan')

    plt.grid()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.show()


def ellipse(a, b, x_0, y_0, alpha):
    cos_alpha = np.cos(alpha)
    sin_alpha = np.sin(alpha)

    def calc(x, y):
        return ((((x - x_0) * cos_alpha) + ((y - y_0) * sin_alpha))**2 /
                (a**2)) + ((((x - x_0) * sin_alpha) -
                            ((y - y_0) * cos_alpha))**2 / (b**2))

    return calc


def gaussian(mu, sig):
    def func(x):
        return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

    return func


def test_distant():

    # known attributes
    x_0, y_0 = 100, 100
    phi = np.pi / 6 * 0
    a = 200
    b = 100
    c = np.sqrt(a**2 - b**2)

    fig, ax = plt.subplots(figsize=(10, 10))

    e = ellipse(a, b, x_0, y_0, phi)
    g = gaussian(1, 0.2)

    X = np.random.uniform(low=[-400, -400], high=[400, 400], size=(2000, 2))
    i = 1
    for x, y in X:
        r = e(x, y)
        # print(i)
        i += 1
        # if y < 0.5 and x < 1:
        # plt.scatter([x], [r], c=('b' if r <= 1 else 'r'), s=1, alpha=g(r))
        ax.scatter([x], [y], s=1, c='b', alpha=g(r))
    ax.set_xlim(-5000, 5000)
    ax.set_ylim(-5000, 5000)
    plt.show()

    X = np.linspace(0, 1000, 1000)
    R = [g(e(x, 0)) for x in X]
    plt.plot(X, R, c='b')
    plt.show()


if __name__ == "__main__":
    test_distant()
