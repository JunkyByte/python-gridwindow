import cv2
import numpy as np
import rpack


class GridWindow_2:
    def __init__(self, max_width=1280, max_height=960):
        self.name = 'autogrid'
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, max_width, max_height)
        self._waitkey = 1
        self.max_width = max_width
        self.max_height = max_height
        self.last_sizes = None
        self.positions = None

    def update(self, images):
        if not isinstance(images, list):
            images = [images]

        sizes = [img.shape[:2][::-1] for img in images]
        if self.last_sizes == None or not sizes == self.last_sizes:
            self.positions = rpack.pack(sizes, max_width=self.max_width, max_height=self.max_height)
            self.last_sizes = sizes
        max_x, max_y = rpack.bbox_size(sizes, self.positions)

        window = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        for img, (x, y) in zip(images, self.positions):
            window[y: y + img.shape[0], x: x + img.shape[1]] = img

        cv2.imshow(self.name, window)
        return cv2.waitKey(self._waitkey)

class GridWindow:
    def __init__(self, n, names=None, max_width=1280, max_height=960):
        self.name = 'grid window'
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, max_width, max_height)

        self.grid_size = n ** (1 / 2)
        if self.grid_size != int(self.grid_size):
            self.grid_size = self.grid_size + 1
        self.grid_size = int(self.grid_size)

        self.width = max_width // self.grid_size
        self.height = max_height // self.grid_size

        self.names = names
        self._waitkey = 1
        self.indices = np.mgrid[0:self.grid_size, 0:self.grid_size].T.reshape((-1, 2))

        if n == 2:
            self.window = np.zeros((max_height // 2, max_width, 3), dtype=np.uint8)
        else:
            self.window = np.zeros((max_height, max_width, 3), dtype=np.uint8)

    def update(self, images):
        if not isinstance(images, list):
            images = [images]
        assert self.names is None or len(images) == len(self.names)

        for ith, (j, i) in enumerate(self.indices):
            if ith >= len(images):
                break

            ys, ye = i * self.height, (i + 1) * self.height
            xs, xe = j * self.width, (j + 1) * self.width
            frame = cv2.resize(images[ith], (self.width, self.height), cv2.INTER_LINEAR)

            self.window[ys:ye, xs:xe] = frame

            if self.names is not None:
                cv2.putText(self.window, str(self.names[ith]).upper(), (xs + 5, ys + 30), 1, 2, (255, 0, 0), 0, cv2.LINE_AA)

        cv2.imshow(self.name, self.window)
        return cv2.waitKey(self._waitkey)


if __name__ == '__main__':
    import glob
    n = 4
    names = ['N' + str(i) for i in range(n)]
    # w = GridWindow(n, names=names, max_width=640, max_height=640)
    w = GridWindow_2(max_width=1024, max_height=640)

    videos = glob.glob('../samples/*.mp4')
    frames = []
    for v in videos:
        cap = cv2.VideoCapture(v)
        fs = []
        ret, frame = cap.read()
        while ret:
            fs.append(frame)
            ret, frame = cap.read()
        frames.append(fs)
        print(f'Video sample with {len(fs)} frames')

    w._waitkey = 50
    ss = np.random.randint(120, 320, size=n * 2)
    sizes = [(ss[i], ss[i + 1]) for i in range(0, len(ss), 2)]
    for frame in frames:
        frames = [cv2.resize(frame, s, cv2.INTER_LINEAR) for s in sizes]
        if w.update(frames) & 0xFF == ord('q'):
            break
