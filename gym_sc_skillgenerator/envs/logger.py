from typing import Dict, Any, Union, Tuple, List

from stable_baselines3.common.logger import KVWriter, FormatUnsupportedError, Video, Figure, Image, SeqWriter
from azureml.core import Run


class AzureRunLogger(KVWriter, SeqWriter):
    def write_sequence(self, sequence: List) -> None:
        self._run.log_list("seq", sequence)

    def close(self) -> None:
        pass

    def __init__(self, run=None):
        if run is None:
            run = Run.get_context()
        self._run = run

    def write(self, key_values: Dict[str, Any], key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
              step: int = 0) -> None:
        for key, value in key_values.items():

            if isinstance(value, Video):
                raise FormatUnsupportedError(["azure"], "video")

            elif isinstance(value, Figure):
                raise FormatUnsupportedError(["azure"], "figure")

            elif isinstance(value, Image):
                raise FormatUnsupportedError(["azure"], "image")

            elif value is not None:
                self._run.log(key, value)
