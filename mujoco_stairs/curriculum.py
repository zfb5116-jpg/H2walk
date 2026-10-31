"""Advance flat -> low stairs -> 20 cm only after demonstrated success."""

from collections import deque


class StairCurriculum:
    HEIGHTS = (0.0, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20)

    def __init__(self, start_stage: str = "flat", window: int = 20, success_rate: float = 0.8):
        if start_stage not in ("flat", "low", "20cm"):
            raise ValueError("start_stage must be flat, low, or 20cm")
        if window < 1 or not 0.0 < success_rate <= 1.0:
            raise ValueError("curriculum window must be positive and success_rate in (0, 1]")
        self.level = {"flat": 0, "low": 1, "20cm": len(self.HEIGHTS) - 1}[start_stage]
        self.window = int(window)
        self.success_rate = float(success_rate)
        self.outcomes = deque(maxlen=self.window)

    @property
    def height(self) -> float:
        return self.HEIGHTS[self.level]

    def record(self, success: bool) -> None:
        self.outcomes.append(bool(success))

    def advance(self) -> bool:
        if (len(self.outcomes) == self.window
                and sum(self.outcomes) / self.window >= self.success_rate
                and self.level < len(self.HEIGHTS) - 1):
            self.level += 1
            self.outcomes.clear()
            return True
        return False

    def state(self) -> dict:
        return {"level": self.level, "outcomes": list(self.outcomes),
                "window": self.window, "success_rate": self.success_rate}

    def restore(self, state: dict) -> None:
        level = int(state["level"])
        if not 0 <= level < len(self.HEIGHTS):
            raise ValueError("invalid saved curriculum level")
        if int(state["window"]) != self.window or float(state["success_rate"]) != self.success_rate:
            raise ValueError("saved curriculum settings differ from this run")
        self.level = level
        self.outcomes.clear()
        self.outcomes.extend(bool(x) for x in state["outcomes"])
