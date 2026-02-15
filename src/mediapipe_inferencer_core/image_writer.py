from queue import Queue
import struct
import time
import numpy as np
import cv2


class MmapImageWriter:
    """Write images to shared memory via memory-mapped file in a background thread.

    File format (compatible with Unity SharedMemoryFrameReader):
        Header (28 bytes):
            - frameSequence: uint32 (offset 0)
            - timestamp: int64 (offset 4)
            - width: uint32 (offset 12)
            - height: uint32 (offset 16)
            - pixelFormat: uint32 (offset 20) - 0 = RGB24
            - dataSize: uint32 (offset 24)
        Data:
            - RGB24 pixel data (offset 28)
    """

    HEADER_SIZE = 28
    PIXEL_FORMAT_RGB24 = 0

    def __init__(self, file_path: str, shape: tuple, dtype: str = 'uint8'):
        import threading

        height, width = shape[:2]
        self._width = width
        self._height = height
        self._data_size = width * height * 3  # RGB24
        self._total_size = self.HEADER_SIZE + self._data_size
        self._frame_sequence = 0

        self._mmap = np.memmap(
            file_path,
            dtype=dtype,
            mode='w+',
            shape=(self._total_size,)
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
        # Resize if needed
        if image.shape[0] != self._height or image.shape[1] != self._width:
            image = cv2.resize(image, (self._width, self._height))

        # Convert BGR to RGB (OpenCV uses BGR, Unity expects RGB)
        if len(image.shape) == 3 and image.shape[2] >= 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Increment frame sequence
        self._frame_sequence += 1
        timestamp = int(time.time() * 1000)  # milliseconds

        # Write header: frameSequence(I), timestamp(q), width(I), height(I), pixelFormat(I), dataSize(I)
        header = struct.pack(
            '<IqIIII',
            self._frame_sequence,
            timestamp,
            self._width,
            self._height,
            self.PIXEL_FORMAT_RGB24,
            self._data_size
        )
        self._mmap[:self.HEADER_SIZE] = np.frombuffer(header, dtype=np.uint8)

        # Write pixel data
        self._mmap[self.HEADER_SIZE:] = image.flatten()
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
