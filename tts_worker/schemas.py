import base64
import json
from typing import List, Optional, Union

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder


class Request(BaseModel):
    text: str
    speaker: str
    speed: float


@dataclass
class ResponseContent:
    audio: Union[bytes, str]
    text: str
    normalized_text: str
    duration_frames: Optional[List[int]] = None
    sampling_rate: Optional[int] = None
    win_length: Optional[int] = None
    hop_length: Optional[int] = None


@dataclass
class Response:
    status_code: int = 200
    status: Optional[str] = None
    content: Optional[ResponseContent] = None

    def __post_init__(self):
        if self.status_code == 200 and self.content is None:
            self.status_code = 500

    def encode(self) -> bytes:
        if self.content and type(self.content.audio) == bytes:
            self.content.audio = base64.b64encode(self.content.audio).decode('utf-8')
        return json.dumps(self, default=pydantic_encoder).encode()
