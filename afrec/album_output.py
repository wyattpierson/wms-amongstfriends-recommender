"""Bounded output schemas for album-evidence v1.1."""

def obj(properties):
    return {"type": "object", "properties": properties,
            "required": list(properties), "additionalProperties": False}


TEXT = {"type": "string"}
CONFIDENCE = {"type": "number", "minimum": 0, "maximum": 1}
DISCOVERY = {"type": "array", "maxItems": 8, "items": obj({
    "artist": TEXT, "album": TEXT, "reason": TEXT, "confidence": CONFIDENCE,
    "genre": TEXT, "era": TEXT, "role": TEXT, "lane": TEXT,
})}
SELECTION = obj({"verdicts": {"type": "array", "maxItems": 8, "items": obj({
    "id": {"type": "integer", "minimum": 1, "maximum": 8},
    "verdict": {"type": "string", "enum": ["keep", "reject"]},
    "confidence": CONFIDENCE,
    "review_ids": {"type": "array", "items": {"type": "integer", "minimum": 1}},
    "reason": TEXT, "contrast": TEXT, "artist_stereotype_risk": {"type": "boolean"},
})}})
