"""Worker thread infrastructure.

Every long-running operation - hashing, analysis, inference, report generation,
batch processing, weight downloads - runs on a worker thread. The GUI thread
only ever receives signals, which is what keeps the interface responsive during
model loading and inference (specification S20, S46).

Two mechanisms are provided:

* :class:`BaseWorker`, a ``QThread`` subclass for a single long operation.
* :class:`WorkerPool`, a ``QThreadPool`` wrapper for many small independent
  jobs (used by batch processing).

Both expose the same signal vocabulary: ``started``, ``progress``, ``status``,
``finished``, ``error``, ``cancelled``.
"""

from __future__ import annotations

import logging
import traceback
from typing import Any, Callable, Optional

from PyQt5.QtCore import (
    QMutex,
    QMutexLocker,
    QObject,
    QRunnable,
    QThread,
    QThreadPool,
    pyqtSignal,
    pyqtSlot,
)

from core.exceptions import OperationCancelled

logger = logging.getLogger(__name__)

__all__ = ["WorkerSignals", "BaseWorker", "FunctionWorker", "WorkerPool", "PoolTask"]


class WorkerSignals(QObject):
    """The signal set every worker emits.

    Signals:
        started: Work has begun on the worker thread.
        progress: ``(percent, message)`` updates, 0-100.
        status: A free-text status line for the status bar.
        finished: ``(result)`` - emitted once on success.
        error: ``(message, traceback_text)`` - emitted once on failure.
        cancelled: Emitted once when the user aborted the operation.
    """

    started = pyqtSignal()
    progress = pyqtSignal(int, str)
    status = pyqtSignal(str)
    finished = pyqtSignal(object)
    error = pyqtSignal(str, str)
    cancelled = pyqtSignal()


class BaseWorker(QThread):
    """A cancellable ``QThread`` that reports progress via signals.

    Subclasses implement :meth:`execute` and must poll :meth:`is_cancelled`
    (or pass :meth:`is_cancelled` into engine calls that accept a cancel check).

    Example:
        >>> class Hash(BaseWorker):
        ...     def execute(self):
        ...         return hash_file(path, cancelled=self.is_cancelled)
        >>> worker = Hash()
        >>> worker.finished.connect(on_done)
        >>> worker.start()
    """

    started_work = pyqtSignal()
    progress = pyqtSignal(int, str)
    status = pyqtSignal(str)
    finished_work = pyqtSignal(object)
    error = pyqtSignal(str, str)
    cancelled_work = pyqtSignal()

    #: Short description used in the status bar and log.
    description: str = "Working"

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._mutex = QMutex()
        self._cancelled = False
        self._result: Any = None

    # ------------------------------------------------------------ cancellation
    def cancel(self) -> None:
        """Request cancellation; the worker stops at its next check point."""
        with QMutexLocker(self._mutex):
            self._cancelled = True
        logger.info("Cancellation requested for %s", type(self).__name__)

    def is_cancelled(self) -> bool:
        """Whether cancellation has been requested."""
        with QMutexLocker(self._mutex):
            return self._cancelled

    @property
    def result(self) -> Any:
        """The value returned by :meth:`execute`, once finished."""
        return self._result

    # -------------------------------------------------------------- reporting
    def report(self, percent: int, message: str = "") -> None:
        """Emit a progress update from inside :meth:`execute`."""
        self.progress.emit(max(0, min(100, int(percent))), message)

    def report_status(self, message: str) -> None:
        """Emit a status line from inside :meth:`execute`."""
        self.status.emit(message)

    # ------------------------------------------------------------- QThread API
    def run(self) -> None:
        """Run :meth:`execute`, translating outcomes into signals."""
        self.started_work.emit()
        logger.info("%s started", type(self).__name__)
        try:
            result = self.execute()
        except OperationCancelled:
            logger.info("%s cancelled", type(self).__name__)
            self.cancelled_work.emit()
            return
        except Exception as exc:
            detail = traceback.format_exc()
            logger.error("%s failed: %s", type(self).__name__, exc)
            logger.debug(detail)
            self.error.emit(str(exc), detail)
            return

        if self.is_cancelled():
            self.cancelled_work.emit()
            return

        self._result = result
        logger.info("%s finished", type(self).__name__)
        self.finished_work.emit(result)

    def execute(self) -> Any:
        """Perform the work. Subclasses must override.

        Returns:
            Any value; it is delivered through :attr:`finished_work`.
        """
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    def stop_and_wait(self, timeout_ms: int = 15000) -> bool:
        """Cancel and block until the thread exits.

        Args:
            timeout_ms: Maximum wait before giving up.

        Returns:
            ``True`` when the thread finished cleanly.
        """
        if not self.isRunning():
            return True
        self.cancel()
        if not self.wait(timeout_ms):
            logger.warning(
                "%s did not stop within %d ms", type(self).__name__, timeout_ms
            )
            return False
        return True


class FunctionWorker(BaseWorker):
    """Runs an arbitrary callable on a worker thread.

    The callable receives ``progress`` and ``cancelled`` keyword arguments when
    it declares them, so engine functions can be reused unchanged.
    """

    def __init__(
        self,
        function: Callable[..., Any],
        *args: Any,
        description: str = "Working",
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self.description = description

    def execute(self) -> Any:
        """Invoke the wrapped callable, injecting progress/cancel hooks."""
        import inspect

        kwargs = dict(self._kwargs)
        try:
            signature = inspect.signature(self._function)
            parameters = signature.parameters
        except (TypeError, ValueError):  # pragma: no cover - builtins
            parameters = {}

        if "progress" in parameters and "progress" not in kwargs:
            kwargs["progress"] = lambda p, m="": self.report(p, m)
        if "cancelled" in parameters and "cancelled" not in kwargs:
            kwargs["cancelled"] = self.is_cancelled

        return self._function(*self._args, **kwargs)


class PoolTask(QRunnable):
    """A single unit of work for :class:`WorkerPool`."""

    def __init__(
        self,
        function: Callable[..., Any],
        *args: Any,
        task_id: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.signals = WorkerSignals()
        self.task_id = task_id
        self._function = function
        self._args = args
        self._kwargs = kwargs
        self._mutex = QMutex()
        self._cancelled = False
        self.setAutoDelete(True)

    def cancel(self) -> None:
        """Request cancellation of this task."""
        with QMutexLocker(self._mutex):
            self._cancelled = True

    def is_cancelled(self) -> bool:
        """Whether this task has been cancelled."""
        with QMutexLocker(self._mutex):
            return self._cancelled

    @pyqtSlot()
    def run(self) -> None:
        """Execute the task, emitting signals for the outcome."""
        if self.is_cancelled():
            self.signals.cancelled.emit()
            return
        self.signals.started.emit()
        try:
            import inspect

            kwargs = dict(self._kwargs)
            try:
                parameters = inspect.signature(self._function).parameters
            except (TypeError, ValueError):  # pragma: no cover
                parameters = {}
            if "progress" in parameters and "progress" not in kwargs:
                kwargs["progress"] = lambda p, m="": self.signals.progress.emit(p, m)
            if "cancelled" in parameters and "cancelled" not in kwargs:
                kwargs["cancelled"] = self.is_cancelled

            result = self._function(*self._args, **kwargs)
        except OperationCancelled:
            self.signals.cancelled.emit()
            return
        except Exception as exc:
            self.signals.error.emit(str(exc), traceback.format_exc())
            return
        self.signals.finished.emit(result)


class WorkerPool(QObject):
    """A bounded thread pool for many small independent jobs."""

    def __init__(self, max_threads: int = 4, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(max(1, int(max_threads)))
        self._tasks: list = []

    @property
    def pool(self) -> QThreadPool:
        """The underlying ``QThreadPool``."""
        return self._pool

    @property
    def active_count(self) -> int:
        """Number of currently running tasks."""
        return self._pool.activeThreadCount()

    def submit(self, task: PoolTask) -> PoolTask:
        """Queue ``task`` for execution."""
        self._tasks.append(task)
        self._pool.start(task)
        return task

    def cancel_all(self) -> None:
        """Cancel queued and running tasks."""
        self._pool.clear()
        for task in self._tasks:
            task.cancel()

    def wait_for_done(self, timeout_ms: int = 30000) -> bool:
        """Block until every task completes or ``timeout_ms`` elapses."""
        return self._pool.waitForDone(timeout_ms)
