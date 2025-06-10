# dataset_manager.py - Dataset management
"""
Manages test dataset loading and provides test cases for load testing.
"""

import csv
import random
import os
from typing import List
from pathlib import Path
from .models import TestCase
import logging

logger = logging.getLogger(__name__)

class DatasetLoadError(Exception):
    """Raised when dataset cannot be loaded."""
    pass

class TestDatasetManager:
    """Manages loading and providing test cases from CSV dataset."""
    
    def __init__(self, csv_path: str, audio_directory: str):
        self.csv_path = Path(csv_path)
        self.audio_directory = Path(audio_directory)
        self.test_cases: List[TestCase] = []
        self.current_index = 0
        
        self._validate_paths()
        self._load_test_cases()
        
        logger.info(f"Loaded {len(self.test_cases)} test cases from {self.csv_path}")

    def _validate_paths(self) -> None:
        """Validate that required paths exist."""
        if not self.csv_path.exists():
            raise DatasetLoadError(f"CSV file not found: {self.csv_path}")
        
        if not self.audio_directory.exists():
            raise DatasetLoadError(f"Audio directory not found: {self.audio_directory}")

    def _load_test_cases(self) -> None:
        """Load test cases from CSV file."""
        boolean_fields = [
            'is_greeting', 'is_introself', 'is_informlicense', 
            'is_informobjective', 'is_informbenefit', 'is_informinterval'
        ]
        
        try:
            with open(self.csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row_num, row in enumerate(reader, start=2):  # Start at 2 for Excel-like numbering
                    try:
                        # Convert string boolean values to actual booleans
                        for field in boolean_fields:
                            if field in row:
                                row[field] = self._parse_boolean(row[field])
                        
                        # Create TestCase with clearer field mapping
                        test_case = TestCase(
                            voice_file_path=row['voice_file_path'],
                            agent_first_name=row['agent_fname'],
                            agent_last_name=row['agent_lname'],
                            expected_greeting=row['is_greeting'],
                            expected_intro_self=row['is_introself'],
                            expected_inform_license=row['is_informlicense'],
                            expected_inform_objective=row['is_informobjective'],
                            expected_inform_benefit=row['is_informbenefit'],
                            expected_inform_interval=row['is_informinterval'],
                        )
                        
                        self.test_cases.append(test_case)
                        
                    except KeyError as e:
                        logger.error(f"Missing required field in row {row_num}: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error parsing row {row_num}: {e}")
                        continue
        
        except Exception as e:
            raise DatasetLoadError(f"Failed to load CSV: {e}")
        
        if not self.test_cases:
            raise DatasetLoadError("No valid test cases found in CSV")

    def _parse_boolean(self, value: str) -> bool:
        """Parse string value to boolean."""
        if isinstance(value, bool):
            return value
        return str(value).lower().strip() in ('true', '1', 'yes', 'on')

    def get_next_test_case(self) -> TestCase:
        """Get next test case in round-robin fashion."""
        if not self.test_cases:
            raise ValueError("No test cases available")
        
        test_case = self.test_cases[self.current_index % len(self.test_cases)]
        self.current_index += 1
        return test_case

    def get_random_test_case(self) -> TestCase:
        """Get random test case."""
        if not self.test_cases:
            raise ValueError("No test cases available")
        
        return random.choice(self.test_cases)

    def get_audio_file_path(self, test_case: TestCase) -> Path:
        """Get full path to audio file for test case."""
        filename = test_case.voice_file_path
        if not filename.lower().endswith('.wav'):
            filename = f"{filename}.wav"
        
        audio_path = self.audio_directory / filename
        
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        return audio_path