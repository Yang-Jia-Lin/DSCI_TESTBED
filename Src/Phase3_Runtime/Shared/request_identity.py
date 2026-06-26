"""Request identity fields shared by Device, Edge, and Cloud runtimes."""

IDENTITY_FIELDS = (
    "round_id",
    "user_id",
    "request_id",
    "decision_id",
    "decision_version",
)


def request_identity(payload: dict, *, require_all: bool = True) -> dict:
    identity = {key: payload.get(key) for key in IDENTITY_FIELDS}
    missing = [key for key, value in identity.items() if value is None or value == ""]
    if require_all and missing:
        raise ValueError(f"Request identity missing fields: {missing}")
    return {key: value for key, value in identity.items() if value is not None}
