from abc import ABC, abstractclassmethod
from queue import Queue
import numpy as np
import cv2
from pygrabber.dshow_graph import FilterGraph


def get_camera_devices() -> list[str]:
    """Get list of available camera device names using DirectShow."""
    graph = FilterGraph()
    return graph.get_input_devices()


def find_camera_index_by_name(device_name: str) -> int | None:
    """Find camera index by device name. Returns None if not found."""
    devices = get_camera_devices()
    for index, name in enumerate(devices):
        if name == device_name:
            return index
    return None

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
        self.__cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
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
