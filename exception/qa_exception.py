
class AgentException(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code

class UnsupportedFileTypeException(AgentException):
    def __init__(self, message: str = "Unsupported file type", code: int = 101):
        super().__init__(message, code)