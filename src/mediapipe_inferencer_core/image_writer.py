from queue import Queue
import numpy as np
import cv2


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
        # Convert BGR to BGRA if mmap expects 4 channels but input has 3 channels
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
