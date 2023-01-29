""" Demo for MagicGrid usage. """
import cv2
import numpy as np
from gridwindow import MagicGrid


if __name__ == '__main__':
    import glob
    import itertools
    import random
    w = MagicGrid(max_width=800, max_height=450)

    videos = glob.glob('samples/*.mp4')
    videos = random.choices(videos, k=random.randint(2, 10))
    frames = []
    for v in videos:
        cap = cv2.VideoCapture(v)
        fs = []
        ret = True
        while ret:
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (int(frame.shape[1] * 0.75), int(frame.shape[0] * 0.75)), cv2.INTER_LINEAR)
            fs.append(frame)
        frames.append(fs)
        print(f'Video sample with {len(fs)} frames')

    w.waitKey = 60
    # print('Applying random resolution to each video!')
    # ss = np.random.randint(240, 480, size=len(videos) * 2)
    # sizes = [(ss[i], ss[i + 1]) for i in range(0, len(ss), 2)]
    for ith, frame in enumerate(itertools.zip_longest(*frames, fillvalue=None)):
        if any(f is None for f in frame):
            break

        # frame = [cv2.resize(f, s, cv2.INTER_LINEAR) for (f, s) in zip(frame, sizes)]
        if w.update(frame) & 0xFF == ord('q'):
            break

    w.close()
