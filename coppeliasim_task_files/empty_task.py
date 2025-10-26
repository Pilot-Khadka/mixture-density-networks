from typing import List
from rlbench.backend.task import Task


class EmptyTask(Task):
    def init_task(self) -> None:
        # TODO: This is called once when a task is initialised.
        pass

    def init_episode(self, index: int) -> List[str]:
        # TODO: This is called at the start of each episode.
        return [""]

    def variation_count(self) -> int:
        return 1
