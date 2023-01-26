import cv2
import numpy as np

import rpack


class MagicGrid:
    """
        MagicGrid allows to create a grid based window by leveraging opencv.

        MagicGrid allows to control a single opencv window to display multiple images without
        having to take care of resolution, position or scale.
        The 'best' way to display the images by respecting a max resolution will be found automatically.
    """
    def __init__(self, max_width: int, max_height: int, waitKey: int = 15, autoscale: bool = True):
        """
        Construct a MagicGrid opencv window.

        Parameters
        ----------
        max_width: int
            max width the window can take in pixels
        max_height: int
            max height the window can take in pixels
        waitKey: int
            time passed to cv2.waitKey at each update of the visualization
        autoscale: bool
            whether or not to autoscale the images to fit. For maximum
            flexibility this should be True, if False and the images do not fit
            in a (max_width, max_height) grid update will throw an error.
        """
        self.name: str = 'autogrid'
        cv2.namedWindow(self.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.name, max_width, max_height)
        self.max_width: int = max_width
        self.max_height: int = max_height
        self.autoscale: bool = autoscale

        self._waitkey: int = waitKey
        self.s: float = 1
        self.last_sizes: list[tuple[int, int]] = None
        self.positions: list[tuple[int, int]] = None

    def _find_max_scale(self, sizes: list[tuple[int, int]]) -> float:
        max_sx = self.max_width / (np.mean([s[0] for s in sizes]) * np.sqrt(len(sizes)))
        max_sy = self.max_height / (np.mean([s[1] for s in sizes]) * np.sqrt(len(sizes)))
        return max(max_sx, max_sy)  # TODO: min is more efficient, max is better

    def update(self, images: list[np.ndarray]) -> int:
        """
        Update the view with new images, a list of images in np.ndarray, bgr format should be passed.

        Parameters
        ----------
        images
            the list of images

        Returns
        -------
        int
            The result of cv2.waitKey function

        Raises
        ------
        rpack.PackingImpossibleError:
            If the images cannot fit in the grid requested (should only happen when autoscale is False).
        """
        sizes = [img.shape[:2][::-1] for img in images]
        source_change = self.last_sizes != sizes
        self.last_sizes = sizes
        if source_change:
            self.s = self._find_max_scale(sizes)

        while True:  # Find a way to represent the images
            try:
                if self.s != 1:
                    images = [cv2.resize(img, (int(s[0] * self.s), int(s[1] * self.s)), cv2.INTER_LINEAR)
                              for s, img in zip(sizes, images)]
                curr_sizes = [img.shape[:2][::-1] for img in images]
                if source_change:
                    self.positions = rpack.pack(curr_sizes, max_width=self.max_width, max_height=self.max_height)
                max_x, max_y = rpack.bbox_size(curr_sizes, self.positions)
                cv2.resizeWindow(self.name, max_x, max_y)
                break
            except rpack.PackingImpossibleError as e:
                if not self.autoscale:
                    raise e
                self.s *= 9 / 10  # TODO: This is fine ?

        window = np.zeros((max_y, max_x, 3), dtype=np.uint8)
        for img, (x, y) in zip(images, self.positions):
            window[y: y + img.shape[0], x: x + img.shape[1]] = img

        cv2.imshow(self.name, window)
        return cv2.waitKey(self._waitkey)


if __name__ == '__main__':
    import glob
    import itertools
    import random
    w = MagicGrid(max_width=800, max_height=600)

    videos = glob.glob('../samples/*.mp4')
    videos = random.choices(videos, k=random.randint(2, 10))
    frames = []
    for v in videos:
        cap = cv2.VideoCapture(v)
        fs = []
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.5), int(frame.shape[0] * 0.5)), cv2.INTER_LINEAR)
            fs.append(frame)
        frames.append(fs)
        print(f'Video sample with {len(fs)} frames')

    w._waitkey = 30
    for ith, frame in enumerate(itertools.zip_longest(*frames, fillvalue=None)):
        if any(f is None for f in frame):
            break
        # if ith % 10 == 0:
        #     ss = np.random.randint(240, 480, size=len(videos) * 2)
        #     sizes = [(ss[i], ss[i + 1]) for i in range(0, len(ss), 2)]
        # frame = [cv2.resize(f, s, cv2.INTER_LINEAR) for (f, s) in zip(frame, sizes)]
        if w.update(frame) & 0xFF == ord('q'):
            break
