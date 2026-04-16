from __future__ import annotations

from dataclasses import dataclass
from math import exp, log1p
import json
from typing import Any, Dict, Optional

from requests import post


def _memory_extract_json(response):
    try:
        if response is None:
            return {}
        if hasattr(response, "json"):
            data = response.json()
        else:
            raw = getattr(response, "__dict__", {}).get("_content", b"")
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            data = json.loads(raw or "{}")
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class AffectiveTraits:
    sensitivity: float = 1.0
    reward_sensitivity: float = 1.0
    resilience: float = 1.0
    visibility_need: float = 1.0


class StressRewardSystem:
    DEFAULT_CONFIG: Dict[str, Any] = {
        "traits": {
            "sensitivity": 1.0,
            "reward_sensitivity": 1.0,
            "resilience": 1.0,
            "visibility_need": 1.0,
        },
        "coupling": {
            "reward_buffers_stress_alpha": 0.30,
            "stress_reduces_reward_beta": 0.20,
        },
        "churn": {
            "enabled": False,
            "stress_weight": 1.5,
            "reward_weight": 1.0,
            "bias": -2.2,
            "temperature": 0.35,
            "min_probability": 0.0,
            "max_probability": 0.95,
        },
        "events": {
            "reaction": {
                "like": {"stress": -0.005, "reward": 0.03},
                "dislike": {"stress": 0.03, "reward": -0.02},
            },
            "comment": {
                "positive": {"stress": -0.02, "reward": 0.07},
                "neutral": {"stress": 0.0, "reward": 0.01},
                "critical": {"stress": 0.03, "reward": -0.01},
                "hostile": {"stress": 0.10, "reward": -0.05},
                "supportive": {"stress": -0.05, "reward": 0.08},
            },
            "share": {
                "positive": {"stress": -0.01, "reward": 0.08},
                "hostile": {"stress": 0.12, "reward": -0.06},
            },
            "moderation": {
                "protected": {"stress": -0.08, "reward": 0.03},
                "sanctioned": {"stress": 0.05, "reward": -0.06},
            },
        },
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = deep_update(self.DEFAULT_CONFIG, config or {})

    @staticmethod
    def _clamp01(value: Any) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except Exception:
            return 0.0

    def churn_enabled(self) -> bool:
        return bool((self.config.get("churn") or {}).get("enabled", False))

    def build_traits(self, overrides: Optional[Dict[str, Any]] = None) -> AffectiveTraits:
        cfg = deep_update(self.config["traits"], overrides or {})
        return AffectiveTraits(
            sensitivity=cfg["sensitivity"],
            reward_sensitivity=cfg["reward_sensitivity"],
            resilience=cfg["resilience"],
            visibility_need=cfg["visibility_need"],
        )

    def compute_reaction_delta(self, *, reaction: str, traits: Optional[AffectiveTraits] = None, current_stress: Optional[float] = None, current_reward: Optional[float] = None, source_status: float = 1.0, relation_weight: float = 1.0, importance: float = 1.0, volume: int = 1) -> Dict[str, float]:
        if reaction not in self.config["events"]["reaction"]:
            raise ValueError(f"Unsupported reaction: {reaction}")
        return self._compute_delta(
            family="reaction",
            subtype=reaction,
            traits=traits,
            current_stress=current_stress,
            current_reward=current_reward,
            source_status=source_status,
            relation_weight=relation_weight,
            importance=importance,
            volume=volume,
        )

    def compute_comment_delta(self, *, tone: str, traits: Optional[AffectiveTraits] = None, current_stress: Optional[float] = None, current_reward: Optional[float] = None, directness: float = 1.0, public_exposure: float = 1.0, source_status: float = 1.0, relation_weight: float = 1.0, importance: float = 1.0, support_strength: float = 1.0) -> Dict[str, float]:
        if tone not in self.config["events"]["comment"]:
            raise ValueError(f"Unsupported comment tone: {tone}")
        return self._compute_delta(
            family="comment",
            subtype=tone,
            traits=traits,
            current_stress=current_stress,
            current_reward=current_reward,
            directness=directness,
            public_exposure=public_exposure,
            source_status=source_status,
            relation_weight=relation_weight,
            importance=importance,
            support_strength=support_strength,
        )

    def compute_share_delta(self, *, tone: str, traits: Optional[AffectiveTraits] = None, current_stress: Optional[float] = None, current_reward: Optional[float] = None, public_exposure: float = 1.0, source_status: float = 1.0, relation_weight: float = 1.0, importance: float = 1.0) -> Dict[str, float]:
        if tone not in self.config["events"]["share"]:
            raise ValueError(f"Unsupported share tone: {tone}")
        return self._compute_delta(
            family="share",
            subtype=tone,
            traits=traits,
            current_stress=current_stress,
            current_reward=current_reward,
            public_exposure=public_exposure,
            source_status=source_status,
            relation_weight=relation_weight,
            importance=importance,
        )

    def _compute_delta(self, *, family: str, subtype: str, traits: Optional[AffectiveTraits] = None, current_stress: Optional[float] = None, current_reward: Optional[float] = None, source_status: float = 1.0, relation_weight: float = 1.0, directness: float = 1.0, public_exposure: float = 1.0, importance: float = 1.0, volume: int = 1, support_strength: float = 1.0) -> Dict[str, float]:
        if volume < 1:
            raise ValueError("volume must be >= 1")
        traits = traits or self.build_traits()
        stress_ctx = 0.2 if current_stress is None else current_stress
        reward_ctx = 0.4 if current_reward is None else current_reward

        base = self.config["events"][family][subtype]
        ds = base["stress"] * importance
        dr = base["reward"] * importance

        if family == "reaction":
            ds *= log1p(volume)
            dr *= source_status * relation_weight * log1p(volume)
        elif family == "comment" and subtype == "hostile":
            ds *= directness * public_exposure * source_status * relation_weight
            dr *= directness
        elif family == "comment" and subtype == "supportive":
            ds *= support_strength
            dr *= support_strength
        elif family == "share" and subtype == "positive":
            dr *= public_exposure
        elif family == "share" and subtype == "hostile":
            ds *= public_exposure * source_status
            dr *= public_exposure

        ds *= traits.sensitivity * (1.0 + 0.35 * stress_ctx)
        dr *= traits.reward_sensitivity * (1.0 + 0.25 * reward_ctx)

        alpha = float(self.config["coupling"]["reward_buffers_stress_alpha"])
        beta = float(self.config["coupling"]["stress_reduces_reward_beta"])

        if ds > 0:
            ds *= max(0.0, 1.0 - alpha * reward_ctx * traits.resilience)
        elif ds < 0:
            ds *= 1.0 + 0.5 * traits.resilience

        if dr > 0:
            dr *= max(0.0, 1.0 - beta * stress_ctx)
        elif dr < 0:
            dr *= 1.0 + 0.25 * stress_ctx

        return {
            "delta_stress": float(ds),
            "delta_reward": float(dr),
            "projected_stress": self._clamp01(stress_ctx + ds),
            "projected_reward": self._clamp01(reward_ctx + dr),
        }

    def compute_current_stress_reward(self, *, base_url: str, agent_id: str, current_tid: str, backward_rounds: int = 24) -> Dict[str, float]:
        response = post(
            f"{base_url.rstrip('/')}/get_stress_reward",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=json.dumps(
                {
                    "agent_id": agent_id,
                    "tid": current_tid,
                    "backward_rounds": backward_rounds,
                }
            ),
        )
        payload = _memory_extract_json(response)
        return {
            "stress": self._clamp01(payload.get("stress", 0.0)),
            "reward": self._clamp01(payload.get("reward", 0.0)),
        }

    def compute_churn_probability(self, *, current_stress: float, current_reward: float) -> float:
        churn_cfg = self.config.get("churn") or {}
        stress = self._clamp01(current_stress)
        reward = self._clamp01(current_reward)
        temperature = max(1e-6, float(churn_cfg.get("temperature", 0.35)))
        logits = (
            float(churn_cfg.get("stress_weight", 1.5)) * stress
            - float(churn_cfg.get("reward_weight", 1.0)) * reward
            + float(churn_cfg.get("bias", -2.2))
        )
        probability = 1.0 / (1.0 + exp(-(logits / temperature)))
        probability = max(float(churn_cfg.get("min_probability", 0.0)), probability)
        probability = min(float(churn_cfg.get("max_probability", 0.95)), probability)
        return self._clamp01(probability)
