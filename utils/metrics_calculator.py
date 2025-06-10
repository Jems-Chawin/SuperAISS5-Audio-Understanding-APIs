# metrics_calculator.py - Metrics calculation and tracking
"""
Calculates and tracks accuracy metrics for load testing results.
"""

import numpy as np
from sklearn.metrics import fbeta_score, confusion_matrix
from typing import List, Dict, Any, Tuple
from .models import PerformanceMetrics, RequestResult
from dataclasses import asdict
import logging

logger = logging.getLogger(__name__)

class MetricsCalculator:
    """Calculates and tracks accuracy metrics for load test results."""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.request_results: List[RequestResult] = []
        self.f2_scores: List[float] = []
        
        # Track all predictions for proper F2 calculation
        self.all_predictions = {
            'greeting': {'expected': [], 'actual': []},
            'intro_self': {'expected': [], 'actual': []},
            'inform_license': {'expected': [], 'actual': []},
            'inform_objective': {'expected': [], 'actual': []},
            'inform_benefit': {'expected': [], 'actual': []},
            'inform_interval': {'expected': [], 'actual': []}
        }

    def record_request_result(self, result: RequestResult) -> None:
        """Record a request result and update metrics."""
        self.request_results.append(result)
        self.metrics.total_requests += 1
        
        if result.timeout:
            self.metrics.timeout_requests += 1
            logger.debug(f"Request timeout recorded for {result.test_case.voice_file_path}")
            return
        
        if not result.success:
            self.metrics.failed_requests += 1
            logger.debug(f"Request failure recorded: {result.error_message}")
            return
        
        self.metrics.successful_requests += 1
        self._update_accuracy_metrics(result)

    def _update_accuracy_metrics(self, result: RequestResult) -> None:
        """Update accuracy metrics based on successful request result."""
        expected_outcomes = result.test_case.expected_outcomes
        actual_outcomes = self._extract_actual_outcomes(result.response_data)
        
        # Calculate overall F2 score for this request
        f2_score = fbeta_score(expected_outcomes, actual_outcomes, beta=2, zero_division=1)
        self.f2_scores.append(f2_score)
        
        # Store predictions for field-level F2 calculation
        self._store_field_predictions(expected_outcomes, actual_outcomes)
        
        # Calculate field-level accuracy
        field_accuracies = self._calculate_field_accuracies(expected_outcomes, actual_outcomes)
        
        # Update running averages
        self._update_running_averages(field_accuracies)

    def _extract_actual_outcomes(self, response_data: Dict[str, Any]) -> List[bool]:
        """Extract actual boolean outcomes from API response."""
        outcome_fields = [
            'is_greeting', 'is_introself', 'is_informlicense',
            'is_informobjective', 'is_informbenefit', 'is_informinterval'
        ]
        
        return [response_data.get(field, False) for field in outcome_fields]

    def _store_field_predictions(self, expected: List[bool], actual: List[bool]) -> None:
        """Store predictions for calculating F2 scores across all samples."""
        field_names = [
            'greeting', 'intro_self', 'inform_license',
            'inform_objective', 'inform_benefit', 'inform_interval'
        ]
        
        for i, field_name in enumerate(field_names):
            self.all_predictions[field_name]['expected'].append(expected[i])
            self.all_predictions[field_name]['actual'].append(actual[i])

    def _calculate_field_accuracies(self, expected: List[bool], actual: List[bool]) -> Dict[str, bool]:
        """Calculate accuracy for each field."""
        field_names = [
            'greeting', 'intro_self', 'inform_license',
            'inform_objective', 'inform_benefit', 'inform_interval'
        ]
        
        return {
            field_name: (exp == act)
            for field_name, exp, act in zip(field_names, expected, actual)
        }

    def _update_running_averages(self, field_accuracies: Dict[str, bool]) -> None:
        """Update running average accuracies."""
        successful_count = self.metrics.successful_requests
        
        # Update each field's running average
        accuracy_fields = [
            'greeting_accuracy', 'intro_self_accuracy', 'inform_license_accuracy',
            'inform_objective_accuracy', 'inform_benefit_accuracy', 'inform_interval_accuracy'
        ]
        
        field_keys = list(field_accuracies.keys())
        
        for accuracy_field, field_key in zip(accuracy_fields, field_keys):
            current_avg = getattr(self.metrics, accuracy_field)
            new_value = 1.0 if field_accuracies[field_key] else 0.0
            
            new_avg = ((current_avg * (successful_count - 1)) + new_value) / successful_count
            setattr(self.metrics, accuracy_field, new_avg)
        
        # Update overall accuracy
        self.metrics.overall_accuracy = sum(
            getattr(self.metrics, field) for field in accuracy_fields
        ) / len(accuracy_fields)
        
        # Update average F2 score
        if self.f2_scores:
            self.metrics.average_f2_score = sum(self.f2_scores) / len(self.f2_scores)

    def _calculate_field_f2_scores(self) -> Dict[str, float]:
        """Calculate F2 score for each field across all predictions."""
        f2_scores = {}
        
        for field_name, predictions in self.all_predictions.items():
            expected = predictions['expected']
            actual = predictions['actual']
            
            if expected and actual:
                # Calculate F2 score across all samples for this field
                f2 = fbeta_score(expected, actual, beta=2, zero_division=1)
                f2_scores[field_name] = f2
            else:
                f2_scores[field_name] = 0.0
                
        return f2_scores

    def _calculate_field_metrics(self) -> Dict[str, Dict[str, float]]:
        """Calculate detailed metrics for each field including TP, FP, TN, FN."""
        field_metrics = {}
        
        for field_name, predictions in self.all_predictions.items():
            expected = predictions['expected']
            actual = predictions['actual']
            
            if expected and actual:
                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(expected, actual, labels=[False, True]).ravel()
                
                # Calculate metrics
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f2 = fbeta_score(expected, actual, beta=2, zero_division=1)
                
                field_metrics[field_name] = {
                    'true_positives': int(tp),
                    'false_positives': int(fp),
                    'true_negatives': int(tn),
                    'false_negatives': int(fn),
                    'precision': precision,
                    'recall': recall,
                    'f2_score': f2
                }
            else:
                field_metrics[field_name] = {
                    'true_positives': 0,
                    'false_positives': 0,
                    'true_negatives': 0,
                    'false_negatives': 0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f2_score': 0.0
                }
                
        return field_metrics

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report."""
        field_f2_scores = self._calculate_field_f2_scores()
        field_metrics = self._calculate_field_metrics()
        
        return {
            'performance_metrics': asdict(self.metrics),
            'rates': {
                'success_rate': self.metrics.success_rate,
                'failure_rate': self.metrics.failure_rate,
                'timeout_rate': self.metrics.timeout_rate,
            },
            'accuracy_breakdown': {
                'greeting_accuracy': f"{self.metrics.greeting_accuracy:.2%}",
                'intro_self_accuracy': f"{self.metrics.intro_self_accuracy:.2%}",
                'inform_license_accuracy': f"{self.metrics.inform_license_accuracy:.2%}",
                'inform_objective_accuracy': f"{self.metrics.inform_objective_accuracy:.2%}",
                'inform_benefit_accuracy': f"{self.metrics.inform_benefit_accuracy:.2%}",
                'inform_interval_accuracy': f"{self.metrics.inform_interval_accuracy:.2%}",
                'overall_accuracy': f"{self.metrics.overall_accuracy:.2%}",
                'average_f2_score': f"{self.metrics.average_f2_score:.2%}",
            },
            'f2_score_breakdown': {
                'greeting_f2': f"{field_f2_scores['greeting']:.2%}",
                'intro_self_f2': f"{field_f2_scores['intro_self']:.2%}",
                'inform_license_f2': f"{field_f2_scores['inform_license']:.2%}",
                'inform_objective_f2': f"{field_f2_scores['inform_objective']:.2%}",
                'inform_benefit_f2': f"{field_f2_scores['inform_benefit']:.2%}",
                'inform_interval_f2': f"{field_f2_scores['inform_interval']:.2%}",
                'overall_f2': f"{self.metrics.average_f2_score:.2%}",
            },
            'detailed_field_metrics': field_metrics
        }