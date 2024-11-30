import numpy as np

def clamp(value, minimum, maximum):
    if value < minimum:
        return minimum

    if value > maximum:
        return maximum

    return value

def get_angle(a, b, c):
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(np.degrees(radians))
    return angle


def get_distance(point1=None, point2=None):
    if not point1 or not point2:
        return

    x1, y1, x2, y2 = point1.x, point1.y, point2.x, point2.y
    L = np.hypot(x2 - x1, y2 - y1)
    
    return np.interp(L, [0, 1], [0, 1000])