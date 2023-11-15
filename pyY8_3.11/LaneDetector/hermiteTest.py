import cv2
import numpy as np

def interpolate(points, tension, img, height):
    c = tension

    for i in range(0, len(points) - 1):
        t = 0

        if i == 0:
            tangentx1 = 0
            tangenty1 = 0
            tangentx2 = (points[i + 1][0] - points[i][0]) * c
            tangenty2 = (points[i + 1][1] - points[i][1]) * c
        elif i == len(points) - 2:
            tangentx1 = (points[i + 1][0] - points[i - 1][0]) * c
            tangenty1 = (points[i + 1][1] - points[i - 1][1]) * c
            tangentx2 = 0
            tangenty2 = 0
        else:
            tangentx1 = (points[i + 1][0] - points[i - 1][0]) * c
            tangenty1 = (points[i + 1][1] - points[i - 1][1]) * c
            tangentx2 = (points[i + 2][0] - points[i][0]) * c
            tangenty2 = (points[i + 2][1] - points[i][1]) * c

        while t <= 1:
            tt = t ** 2
            ttt = t ** 3
            h1 = (2*(ttt)) - (3*(tt)) + 1
            h2 = (-2 * (ttt)) + (3*(tt))
            h3 = (ttt) - (2*(tt)) + t
            h4 = (ttt) - (tt)
            x = int(h1 * points[i][0] + h2 * points[i + 1][0] + h3 * tangentx1 + h4 * tangentx2)
            y = int(height - (h1 * points[i][1] + h2 * points[i + 1][1] + h3 * tangenty1 + h4 * tangenty2))
            putPixel(img, x, height - y, (42, 73, 120), 2)
            t += 0.005

    #for point in points:
        #putPixel(img, point[0], point[1], (224, 224, 224), 5)


def putPixel(image, x, y, color, size):
    cv2.rectangle(image, (x, y), (x + size, y + size), color, -1)


def main():
    img_height = 600
    img_width = 600

    # Create a black image
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    points = [[100, 100], [200, 300], [400, 500], [500, 300], [300, 100], [334, 230]]

    interpolate(points, 0.5, img, img_height)

    cv2.imshow('Interpolation with OpenCV', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
