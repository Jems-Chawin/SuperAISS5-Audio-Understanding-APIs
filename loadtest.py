# loadtest.py - Main load test implementation
# locust -f loadtest.py --users 10 --spawn-rate 1 --run-time 10m --host http://172.16.30.124:4000 --web-port 9000
"""
Main load test implementation using Locust framework.
Tests multimodal LLM API with audio files and agent data.
Modified to stop after 100 requests.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import random
import time
from datetime import datetime
from pathlib import Path

from locust import HttpUser, task, constant, events
from locust.env import Environment

import gevent
import threading

from utils.models import TestCase, RequestResult
from utils.dataset_manager import TestDatasetManager
from utils.metrics_calculator import MetricsCalculator
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

# ==================================== CONFIG ====================================
# DATASET_CSV_PATH = "processed_train.csv"
# AUDIO_DIRECTORY = f"/home/siamai/data/datasets/Audio Understanding SCBx/speechs/train"

DATASET_CSV_PATH = "processed_handmade.csv"
AUDIO_DIRECTORY = f"./data_handmade/trainData/speech"

API_ENDPOINT = "/eval"
REQUEST_TIMEOUT_SECONDS = 180 # change this to 180 to match the evaluation criteria
MAX_REQUESTS = 300  # Stop after 100 requests


# ================== GLOBAL INSTANCES - initialized once for all users ==================
dataset_manager = None
metrics_calculator = None
request_counter = 0
test_complete = False
request_lock = threading.Lock()

def initialize_global_resources():
    """Initialize global resources that are shared across all users."""
    global dataset_manager, metrics_calculator, request_counter, test_complete
    
    try:
        dataset_manager = TestDatasetManager(DATASET_CSV_PATH, AUDIO_DIRECTORY)
        metrics_calculator = MetricsCalculator()
        request_counter = 0
        test_complete = False
        
        logger.info("Global resources initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize global resources: {e}")
        raise


# ==================================== LOCUST ====================================
class MultimodalAPILoadTestUser(HttpUser):
    """Locust user for testing multimodal LLM API."""
    
    wait_time = constant(0)  # No wait between requests
    
    def on_start(self):
        """Initialize user when starting."""
        self.user_id = f"user_{random.randint(1000, 9999)}"
        logger.info(f"Load test user started: {self.user_id}")

    @task
    def test_multimodal_api_endpoint(self):
        """Main task: test the multimodal API endpoint."""
        global request_counter, test_complete
        
        # Check if we've reached the request limit
        with request_lock:
            if test_complete or request_counter >= MAX_REQUESTS:
                return
            
            # Increment counter atomically
            current_count = request_counter
            request_counter += 1
        
        logger.info(f"Processing request {current_count + 1}/{MAX_REQUESTS}")
        
        try:
            # Get test case and prepare request
            test_case = dataset_manager.get_next_test_case()
            request_payload = self._prepare_api_request(test_case)
            
            # Make API request and record result
            start_time = time.time()
            result = self._make_api_request(test_case, request_payload, start_time)
            
            # Record result for metrics
            metrics_calculator.record_request_result(result)
            
            # Check if we've completed all requests
            with request_lock:
                completed_requests = metrics_calculator.metrics.total_requests
                if completed_requests >= MAX_REQUESTS and not test_complete:
                    test_complete = True
                    logger.info(f"Completed {MAX_REQUESTS} requests. Waiting for any in-flight requests...")
                    # Give a small delay to ensure all metrics are recorded
                    gevent.sleep(2)
                    self.environment.runner.quit()
            
        except Exception as e:
            logger.error(f"Error in test task: {e}")
            # Record failure
            error_result = RequestResult(
                test_case=None,
                response_data={},
                success=False,
                timeout=False,
                error_message=str(e),
                timestamp=datetime.now(),
                response_time_ms=0
            )
            metrics_calculator.record_request_result(error_result)
            
            # Check again after recording the error
            with request_lock:
                completed_requests = metrics_calculator.metrics.total_requests
                if completed_requests >= MAX_REQUESTS and not test_complete:
                    test_complete = True
                    logger.info(f"Completed {MAX_REQUESTS} requests (including errors). Stopping...")
                    gevent.sleep(2)
                    self.environment.runner.quit()

    def _prepare_api_request(self, test_case: TestCase) -> tuple:
        """Prepare multipart form data for API request."""
        # Agent data as JSON
        agent_data = {
            "agent_fname": test_case.agent_first_name,
            "agent_lname": test_case.agent_last_name
        }
        
        # Read audio file
        audio_file_path = dataset_manager.get_audio_file_path(test_case)
        
        with open(audio_file_path, "rb") as audio_file:
            audio_content = audio_file.read()
        
        # Prepare multipart form data
        files = {
            "voice_file": (audio_file_path.name, audio_content, "application/octet-stream")
        }
        data = {
            "agent_data": json.dumps(agent_data)
        }
        
        return files, data

    def _make_api_request(self, test_case: TestCase, request_payload: tuple, start_time: float) -> RequestResult:
        """Make API request and return structured result."""
        files, data = request_payload
        
        with self.client.post(
            API_ENDPOINT,
            files=files,
            data=data,
            timeout=REQUEST_TIMEOUT_SECONDS,
            catch_response=True
        ) as response:
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Check for timeout
            if response.elapsed.total_seconds() >= REQUEST_TIMEOUT_SECONDS:
                response.failure("Request timeout")
                return RequestResult(
                    test_case=test_case,
                    response_data={},
                    success=False,
                    timeout=True,
                    error_message="Request timeout",
                    timestamp=datetime.now(),
                    response_time_ms=response_time_ms
                )
            
            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}: {response.text[:100]}"
                response.failure(error_msg)
                return RequestResult(
                    test_case=test_case,
                    response_data={},
                    success=False,
                    timeout=False,
                    error_message=error_msg,
                    timestamp=datetime.now(),
                    response_time_ms=response_time_ms
                )
            
            # Parse JSON response
            try:
                response_json = response.json()
                response.success()
                
                return RequestResult(
                    test_case=test_case,
                    response_data=response_json,
                    success=True,
                    timeout=False,
                    error_message=None,
                    timestamp=datetime.now(),
                    response_time_ms=response_time_ms
                )
                
            except Exception as e:
                error_msg = f"JSON parse error: {e}"
                response.failure(error_msg)
                return RequestResult(
                    test_case=test_case,
                    response_data={},
                    success=False,
                    timeout=False,
                    error_message=error_msg,
                    timestamp=datetime.now(),
                    response_time_ms=response_time_ms
                )


@events.test_start.add_listener
def on_test_start(environment: Environment, **kwargs):
    """Initialize resources when test starts."""
    print("🚀 Initializing multimodal LLM load test...")
    print(f"📊 Test will stop after {MAX_REQUESTS} requests")
    initialize_global_resources()
    print("✅ Load test initialization complete")


@events.test_stop.add_listener
def on_test_stop(environment: Environment, **kwargs):
    """Generate and save results when test stops."""
    print("\n" + "="*80)
    print("🎯 MULTIMODAL LLM LOAD TEST RESULTS")
    print("="*80)
    
    # Generate summary report
    summary = metrics_calculator.get_summary_report()
    
    # Print performance metrics
    print_performance_summary(summary)
    
    # Save detailed results
    save_detailed_results(summary)
    
    print("="*80)


# ==================================== REPORTS ====================================
def print_performance_summary(summary: Dict[str, Any]) -> None:
    """Print formatted performance summary."""
    metrics = summary['performance_metrics']
    rates = summary['rates']
    accuracy = summary['accuracy_breakdown']
    f2_breakdown = summary.get('f2_score_breakdown', {})
    detailed_metrics = summary.get('detailed_field_metrics', {})
    
    print(f"📊 Request Performance:")
    print(f"   Total Requests: {metrics['total_requests']}")
    print(f"   Successful: {metrics['successful_requests']}")
    print(f"   Failed: {metrics['failed_requests']}")
    print(f"   Timeouts: {metrics['timeout_requests']}")
    print(f"   Success Rate: {rates['success_rate']:.2f}%")
    print(f"   Failure Rate: {rates['failure_rate']:.2f}%")
    print(f"   Timeout Rate: {rates['timeout_rate']:.2f}%")
    
    print(f"\n🎯 Accuracy Metrics:")
    print(f"   Greeting Detection: {accuracy['greeting_accuracy']}")
    print(f"   Self Introduction: {accuracy['intro_self_accuracy']}")
    print(f"   License Information: {accuracy['inform_license_accuracy']}")
    print(f"   Objective Information: {accuracy['inform_objective_accuracy']}")
    print(f"   Benefit Information: {accuracy['inform_benefit_accuracy']}")
    print(f"   Interval Information: {accuracy['inform_interval_accuracy']}")
    
    print(f"\n📈 F₂ Score Breakdown:")
    print(f"   Greeting F₂: {f2_breakdown.get('greeting_f2', 'N/A')}")
    print(f"   Self Introduction F₂: {f2_breakdown.get('intro_self_f2', 'N/A')}")
    print(f"   License Information F₂: {f2_breakdown.get('inform_license_f2', 'N/A')}")
    print(f"   Objective Information F₂: {f2_breakdown.get('inform_objective_f2', 'N/A')}")
    print(f"   Benefit Information F₂: {f2_breakdown.get('inform_benefit_f2', 'N/A')}")
    print(f"   Interval Information F₂: {f2_breakdown.get('inform_interval_f2', 'N/A')}")
    
    # Print detailed metrics for fields where F2 differs significantly from accuracy
    print(f"\n📊 Detailed Metrics (where F₂ ≠ Accuracy):")
    for field_name, field_data in detailed_metrics.items():
        field_accuracy = getattr(metrics, f"{field_name}_accuracy", 0.0)
        field_f2 = field_data['f2_score']
        
        # Only show details if F2 differs from accuracy by more than 1%
        if abs(field_f2 - field_accuracy) > 0.01:
            print(f"\n   {field_name.replace('_', ' ').title()}:")
            print(f"      Precision: {field_data['precision']:.2%}")
            print(f"      Recall: {field_data['recall']:.2%}")
            print(f"      True Positives: {field_data['true_positives']}")
            print(f"      False Positives: {field_data['false_positives']}")
            print(f"      False Negatives: {field_data['false_negatives']}")
    
    print(f"\n🏆 Overall Performance:")
    print(f"   Overall Accuracy: {accuracy['overall_accuracy']}")
    print(f"   Average F₂ Score: {accuracy['average_f2_score']}")

def save_detailed_results(summary: Dict[str, Any]) -> None:
    """Save detailed results to JSON file."""
    results_dir = Path("./loadtest_results")
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f"multimodal_loadtest_results_{timestamp}.json"
    
    detailed_results = {
        'summary': summary,
        'test_configuration': {
            'dataset_path': DATASET_CSV_PATH,
            'audio_directory': AUDIO_DIRECTORY,
            'api_endpoint': API_ENDPOINT,
            'request_timeout': REQUEST_TIMEOUT_SECONDS,
            'max_requests': MAX_REQUESTS,
        },
        'individual_results': [
            {
                'timestamp': result.timestamp.isoformat(),
                'voice_file': result.test_case.voice_file_path if result.test_case else None,
                'success': result.success,
                'timeout': result.timeout,
                'response_time_ms': result.response_time_ms,
                'error_message': result.error_message,
                'expected_outcomes': result.test_case.expected_outcomes if result.test_case else None,
                'actual_response': result.response_data,
            }
            for result in metrics_calculator.request_results
        ]
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, default=str)
    
    print(f"\n💾 Detailed results saved to: {results_file}")

# ==================================== MAIN ====================================
if __name__ == "__main__":
    print("🚀 Multimodal LLM Load Test")
    print("📋 Configuration:")
    print(f"   Dataset: {DATASET_CSV_PATH}")
    print(f"   Audio Directory: {AUDIO_DIRECTORY}")
    print(f"   API Endpoint: {API_ENDPOINT}")
    print(f"   Request Timeout: {REQUEST_TIMEOUT_SECONDS}s")
    print(f"   Maximum Requests: {MAX_REQUESTS}")
    print("\n💡 Usage:")
    print("   locust -f loadtest_main.py --users 10 --spawn-rate 1 --host http://172.16.30.124:4000")
    print("\n📁 Output:")
    print("   Results will be saved to ./load_test_results/")
    print("\n⚠️  Note: Test will automatically stop after 100 requests")

# locust -f loadtest.py --users 10 --spawn-rate 1 --run-time 60m --host http://172.16.30.124:4000 --web-port 9000
# locust -f loadtest.py --headless --users 10 --spawn-rate 1 --run-time 60m --host http://172.16.30.124:4000 --web-port 9000