# models.py - Data structures and models
"""
Data models for multimodal LLM load testing.
Defines the structure for test data, metrics, and responses.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class TestCase:
    """Represents a single test case with expected outcomes."""
    voice_file_path: str = ""
    agent_first_name: str = ""
    agent_last_name: str = ""
    expected_greeting: bool = False
    expected_intro_self: bool = False
    expected_inform_license: bool = False
    expected_inform_objective: bool = False
    expected_inform_benefit: bool = False
    expected_inform_interval: bool = False

    @property
    def expected_outcomes(self) -> List[bool]:
        """Get all expected boolean outcomes as a list."""
        return [
            self.expected_greeting,
            self.expected_intro_self,
            self.expected_inform_license,
            self.expected_inform_objective,
            self.expected_inform_benefit,
            self.expected_inform_interval,
        ]

    @property
    def outcome_field_names(self) -> List[str]:
        """Get field names for boolean outcomes."""
        return [
            'is_greeting',
            'is_introself',
            'is_informlicense',
            'is_informobjective',
            'is_informbenefit',
            'is_informinterval',
        ]

@dataclass
class RequestResult:
    """Represents the result of a single API request."""
    test_case: TestCase
    response_data: Dict[str, Any]
    success: bool
    timeout: bool
    error_message: Optional[str]
    timestamp: datetime
    response_time_ms: float

@dataclass
class PerformanceMetrics:
    """Tracks overall performance and accuracy metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    
    # Field-specific accuracy scores
    greeting_accuracy: float = 0.0
    intro_self_accuracy: float = 0.0
    inform_license_accuracy: float = 0.0
    inform_objective_accuracy: float = 0.0
    inform_benefit_accuracy: float = 0.0
    inform_interval_accuracy: float = 0.0
    
    overall_accuracy: float = 0.0
    average_f2_score: float = 0.0

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.failed_requests / self.total_requests) * 100

    @property
    def timeout_rate(self) -> float:
        """Calculate timeout rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.timeout_requests / self.total_requests) * 100