"""OpenAI function-tool schemas. Stdlib only so the baseline can import it."""

TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "check_availability",
            "description": "Check whether a table is available at the requested date, time and party size.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {"type": "string", "description": "Date as YYYY-MM-DD"},
                    "time": {"type": "string", "description": "Time as HH:MM, 24-hour"},
                    "party_size": {"type": "integer", "description": "Number of guests"},
                },
                "required": ["date", "time", "party_size"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "book_reservation",
            "description": "Book a reservation. Only call after availability was checked and the caller explicitly confirmed.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name for the reservation"},
                    "party_size": {"type": "integer", "description": "Number of guests"},
                    "date": {"type": "string", "description": "Date as YYYY-MM-DD"},
                    "time": {"type": "string", "description": "Time as HH:MM, 24-hour"},
                },
                "required": ["name", "party_size", "date", "time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_reservation",
            "description": "Cancel the caller's confirmed reservations on a given date.",
            "strict": True,
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name on the reservation"},
                    "date": {"type": "string", "description": "Date as YYYY-MM-DD"},
                },
                "required": ["name", "date"],
                "additionalProperties": False,
            },
        },
    },
]
