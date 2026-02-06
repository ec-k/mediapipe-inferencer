from abc import ABC, abstractclassmethod
from queue import Queue
import numpy as np
import cv2

class ImageProvider(ABC):
    def __init__(self, cache_queue_length):
        self.__cache_queue_length = cache_queue_length
        self.__frames = Queue()
        self.__latest_frame = None

    @abstractclassmethod
    def _fetch_image(self):
        pass

    @property
    def latest_frame(self):
        return self.__latest_frame

    def update(self):
        image = self._fetch_image()
        self.__frames.put(image)
        self.__latest_frame = image
        if self.__frames.qsize() > self.__cache_queue_length:
            self.__pop()

    def __pop(self):
        return self.__frames.get()

class WebcamImageProvider(ImageProvider):
    def __init__(self, cache_queue_length, device_index):
        super().__init__(cache_queue_length)
        self.__cap = cv2.VideoCapture(device_index)
    def _fetch_image(self):
        success, image = self.__cap.read()
        return image
    @property
    def is_opened(self) -> bool:
        return self.__cap.isOpened()
    def release_capture(self) -> None:
        self.__cap.release()

class MmapImageProvider(ImageProvider):
    def __init__(self, cache_queue_length, data_file_path, shape):
        super().__init__(cache_queue_length)
        self.__mmap = np.memmap(
            data_file_path,
            dtype='uint8',
            mode='r',
            shape=shape
        )
        self.update()
    def _fetch_image(self):
        return self.__mmap


class MmapImageWriter:
    """Write images to shared memory via memory-mapped file in a background thread."""

    def __init__(self, file_path: str, shape: tuple, dtype: str = 'uint8'):
        import threading

        self._shape = shape
        self._mmap = np.memmap(
            file_path,
            dtype=dtype,
            mode='w+',
            shape=shape
        )
        self._queue = Queue(maxsize=1)
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self) -> None:
        """Background thread that writes images to shared memory."""
        while self._running:
            try:
                image = self._queue.get(timeout=0.1)
            except:
                continue
            if image is None:
                break
            self._write_internal(image)

    def _write_internal(self, image: np.ndarray) -> None:
        """Actual write operation."""
        if image.shape[:2] != self._shape[:2]:
            image = cv2.resize(image, (self._shape[1], self._shape[0]))
        if len(self._shape) == 3 and self._shape[2] == 4 and len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
        self._mmap[:] = image
        self._mmap.flush()

    def write(self, image: np.ndarray) -> None:
        """Queue image for writing. Drops frame if queue is full."""
        try:
            self._queue.put_nowait(image.copy())
        except:
            pass  # Drop frame if queue is full

    def close(self) -> None:
        """Stop worker thread and close the memory-mapped file."""
        self._running = False
        try:
            self._queue.put_nowait(None)
        except:
            pass
        self._thread.join(timeout=1.0)
        if self._mmap is not None:
            self._mmap.flush()
            del self._mmap
            self._mmap = None