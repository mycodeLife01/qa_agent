class AgentException(Exception):
    def __init__(self, message: str, code: int):
        super().__init__(message)
        self.code = code


class AgentUnsupportedFileTypeException(AgentException):
    def __init__(self, message: str = "Unsupported file type", code: int = 101):
        super().__init__(message, code)


class AgentMissingParamsException(AgentException):
    def __init__(self, message: str = "Missing params", code: int = 102):
        super().__init__(message, code)


class AgentInvalidParamsException(AgentException):
    def __init__(self, message: str = "Invalid params", code: int = 103):
        super().__init__(message, code)
