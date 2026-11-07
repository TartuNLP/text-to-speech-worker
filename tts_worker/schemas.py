import base64
import json

from pydantic import BaseModel
from pydantic.dataclasses import dataclass
from pydantic.json import pydantic_encoder


class Request(BaseModel):
    text: str
    speaker: str
    speed: float


@dataclass
class ResponseContent:
    audio: bytes | str
    text: str
    normalized_text: str
    duration_frames: list[int] | None = None
    sampling_rate: int | None = None
    win_length: int | None = None
    hop_length: int | None = None


@dataclass
class Response:
    status_code: int = 200
    status: str | None = None
    content: ResponseContent | None = None

    def __post_init__(self):
        if self.status_code == 200 and self.content is None:
            self.status_code = 500

    def encode(self) -> bytes:
        if self.content and type(self.content.audio) == bytes:
            self.content.audio = base64.b64encode(self.content.audio).decode('utf-8')
        return json.dumps(self, default=pydantic_encoder).encode()
