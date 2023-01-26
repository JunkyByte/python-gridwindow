import cv2
import numpy as np

import rpack


class GridWindow_2:
    def __init__(self, max_width: int, max_height: int, autoscale: bool = True):
        self.name = 'autogrid'
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, max_width, max_height)
        self.max_width = max_width
        self.max_height = max_height
        self.autoscale = autoscale

        self._waitkey = 1
        self.s = 1
        self.last_sizes = None
        self.positions = None

    def find_max_scale(self, sizes: list[tuple[int, int]]):
        max_sx = self.max_width / max(s[0] for s in sizes)
        max_sy = self.max_height / max(s[1] for s in sizes)
        print(sizes, self.max_width, self.max_height)
        print(max_sx, max_sy)
        return min(max_sx, max_sy)

    def update(self, images: list):
        sizes = [img.shape[:2][::-1] for img in images]
        source_change = self.last_sizes != sizes
        self.last_sizes = sizes
        print('Change', source_change)
        if source_change:
            self.s = self.find_max_scale(sizes)

        while True:  # Find a way to represent the images
            try:
                if self.s != 1:
                    print('Rescaling with ratio', self.s)
                    images = [cv2.resize(img, (int(s[0] * self.s), int(s[1] * self.s)), cv2.INTER_LINEAR)
                              for s, img in zip(sizes, images)]
                curr_sizes = [img.shape[:2][::-1] for img in images]
                if source_change:
                    print('packing')
                    self.positions = rpack.pack(curr_sizes, max_width=self.max_width, max_height=self.max_height)
                    print('finished packing!')
                max_x, max_y = rpack.bbox_size(curr_sizes, self.positions)
                cv2.resizeWindow(self.name, max_x, max_y)
                break
            except rpack.PackingImpossibleError as e:
                print(e)
                if not self.autoscale:
                    raise e
                self.s *= 4 / 5
                # TODO: This can ofc be more efficient
                # (especially at start -> scale so that highest is < max_height)

        window = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        for img, (x, y) in zip(images, self.positions):
            window[y: y + img.shape[0], x: x + img.shape[1]] = img

        cv2.imshow(self.name, window)
        return cv2.waitKey(self._waitkey)


if __name__ == '__main__':
    import glob
    import itertools
    w = GridWindow_2(max_width=1024, max_height=768)

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

    w._waitkey = 30
    for ith, frame in enumerate(itertools.zip_longest(*frames, fillvalue=None)):
        if any(f is None for f in frame):
            break
        if ith % 10 == 0:
            ss = np.random.randint(240, 640, size=len(videos) * 2)
            sizes = [(ss[i], ss[i + 1]) for i in range(0, len(ss), 2)]
        frame = [cv2.resize(f, s, cv2.INTER_LINEAR) for (f, s) in zip(frame, sizes)]
        if w.update(frame) & 0xFF == ord('q'):
            break
