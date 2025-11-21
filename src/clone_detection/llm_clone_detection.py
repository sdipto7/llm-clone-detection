import csv
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm
from prompts import (
    get_system_prompt_for_direct_clone_detection,
    get_prompt_for_direct_clone_detection,
    get_system_prompt_for_algorithm_based_clone_detection,
    get_prompt_to_generate_algorithm_from_code,
    get_prompt_for_algorithm_based_clone_detection
)

load_dotenv(override=True)

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/llm_clone_detection.log",
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DETECTION_MODE_DIRECT = 'direct'
DETECTION_MODE_ALGORITHM = 'algorithm-based'
CLONE_LABEL = 'clone'
NON_CLONE_LABEL = 'non_clone'
LANGUAGE_JAVA = 'Java'
LANGUAGE_PYTHON = 'Python'
FREE_MODEL_MAX_REQUESTS_PER_MINUTE = 16
MAX_RETRY_ATTEMPTS = 5
RATE_LIMIT_WAIT_SECONDS = 65

BASE_URL = "https://openrouter.ai/api/v1"
SUPPORTED_LLM_API_MAP = {
    "gpt-oss": "openai/gpt-oss-20b:free",
    "deepseek-r1": "deepseek/deepseek-r1:free",
    "deepseek-v3": "deepseek/deepseek-chat-v3-0324:free",
    "llama-3.3": "meta-llama/llama-3.3-70b-instruct:free",
    "qwen-2.5-coder": "qwen/qwen-2.5-coder-32b-instruct:free",
    "qwen-2.5": "qwen/qwen-2.5-72b-instruct:free",
    "grok-4.1": "x-ai/grok-4.1-fast:free"
}

class LLMCloneDetector:
    """Detects code clones using Large Language Models."""

    def __init__(self, dataset_root: Path, output_root: Path, model: str, detection_mode: str):
        """
        Initialize LLM Clone Detector.

        Args:
            dataset_root: Path to preprocessed dataset
            output_root: Path to save results
            detection_mode: 'direct' or 'algorithm_based'
        """
        self.dataset_root = dataset_root
        self.detection_mode = detection_mode

        self._initialize_llm_config(model)
        self._initialize_rate_limiting()

        self.output_root = output_root.joinpath(self.resolve_model_name_for_path(self.model))
        self._initialize_problem_cache()

    def _initialize_llm_config(self, model: str):
        """Initialize LLM configuration."""
        self.base_url = BASE_URL
        self.api_key = os.getenv("API_KEY")
        self.model = SUPPORTED_LLM_API_MAP.get(model)

    def _initialize_rate_limiting(self):
        """Initialize rate limiting for free models."""
        self.is_free_model = ":free" in self.model
        self.request_timestamps = deque()

    def _initialize_problem_cache(self):
        """Initialize problem-level cache directory."""
        self.problem_cache_dir = self.output_root / "problem_cache" / self.detection_mode
        self.problem_cache_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Problem cache directory: {self.problem_cache_dir}")

    def resolve_model_name_for_path(self, model_name: str) -> str:
        return model_name.rsplit("/")[-1].split(":")[0].replace("-", "_")

    def __enter__(self):
        self._validate_configuration()
        self._setup_directories()
        logging.info(f"Initialized LLMCloneDetector with mode: {self.detection_mode}")
        return self

    def __exit__(self, exception, _, __):
        if exception is None:
            print(f"Clone detection ({self.detection_mode} mode) completed successfully")
            logging.info(f"Clone detection ({self.detection_mode} mode) completed successfully")
        else:
            print(f"Clone detection failed due to {exception}")
            logging.error(f"Clone detection failed due to {exception}")

    def detect_clones(self) -> List[Dict]:
        """Process all problems for clone detection."""
        metadata_files = self._get_metadata_files()
        logging.info(f"Found {len(metadata_files)} problems to process")

        results, stats = self._process_all_problems(metadata_files)
        self._save_all_results(results)
        self._log_completion_stats(stats)

        return results

    def _validate_configuration(self):
        """Validate required configuration and directories."""
        self.data_dir = self.dataset_root.joinpath("data")
        self.metadata_dir = self.dataset_root.joinpath("metadata")

        required_dirs = {
            'dataset_root': self.dataset_root,
            'data_dir': self.data_dir,
            'metadata_dir': self.metadata_dir
        }

        for name, directory in required_dirs.items():
            if not directory.exists():
                error_msg = f"Directory {directory} does not exist"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)

        if not self.api_key:
            error_msg = "API_KEY not found in environment variables"
            logging.error(error_msg)
            raise ValueError(f"{error_msg}. Please set it in .env file")

    def _setup_directories(self):
        """Create output directories."""
        self.output_root.mkdir(parents=True, exist_ok=True)

    def _get_metadata_files(self) -> List[Path]:
        """Get list of metadata files to process."""
        return sorted(self.metadata_dir.glob("p*.csv"))

    def _get_problem_cache_file(self, problem_id: str) -> Path:
        """Get cache file path for a problem."""
        return self.problem_cache_dir / f"{problem_id}.json"

    def _is_problem_cached(self, problem_id: str) -> bool:
        """Check if problem has been fully processed."""
        cache_file = self._get_problem_cache_file(problem_id)
        
        if not cache_file.exists():
            return False
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            if cache_data.get('status') == 'completed' and cache_data.get('num_comparisons', 0) > 0:
                logging.info(f"✓ {problem_id} already processed ({cache_data['num_comparisons']} comparisons)")
                return True
            
        except Exception as e:
            logging.warning(f"Invalid cache for {problem_id}: {e}")
            cache_file.unlink(missing_ok=True)
        
        return False

    def _load_cached_problem_results(self, problem_id: str) -> List[Dict]:
        """Load cached results for a problem."""
        cache_file = self._get_problem_cache_file(problem_id)
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            results = cache_data.get('results', [])
            logging.info(f"Loaded {len(results)} cached results for {problem_id}")
            return results
            
        except Exception as e:
            logging.error(f"Error loading cached results for {problem_id}: {e}")
            return []

    def _save_problem_to_cache(self, problem_id: str, results: List[Dict]):
        """Save problem results to cache."""
        cache_file = self._get_problem_cache_file(problem_id)
        
        try:
            cache_data = {
                'problem_id': problem_id,
                'detection_mode': self.detection_mode,
                'model': self.model,
                'status': 'completed',
                'num_comparisons': len(results),
                'timestamp': datetime.now().isoformat(),
                'results': results
            }
            
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, indent=2)
            
            logging.info(f"Cached {len(results)} results for {problem_id}")
            
        except Exception as e:
            logging.warning(f"Error caching results for {problem_id}: {e}")

    def _process_all_problems(self, metadata_files: List[Path]) -> Tuple[List[Dict], Dict]:
        """Process all problems and return results with statistics."""
        all_results = []
        stats = {
            'successful': 0,
            'cached': 0,
            'failed': []
        }

        for metadata_file in tqdm(
                metadata_files,
                desc="Problems",
                bar_format="{desc:<10.10}{percentage:3.0f}%|{bar:10}{r_bar}",
                leave=True
        ):
            problem_id = metadata_file.stem
            
            if self._is_problem_cached(problem_id):
                cached_results = self._load_cached_problem_results(problem_id)
                all_results.extend(cached_results)
                stats['cached'] += 1
                continue
            
            logging.info(f"Processing {problem_id}...")

            try:
                results = self._process_problem(problem_id, metadata_file)
                
                self._save_problem_to_cache(problem_id, results)
                
                all_results.extend(results)
                stats['successful'] += 1
                logging.info(f"Completed {len(results)} comparisons for {problem_id}")
                
            except Exception as e:
                logging.error(f"Error processing {problem_id}: {e}")
                stats['failed'].append((problem_id, str(e)))

        return all_results, stats

    def _process_problem(self, problem_id: str, metadata_file: Path) -> List[Dict]:
        """Process a single problem for clone detection."""
        submissions = self._read_metadata_file(metadata_file)
        categorized = self._categorize_submissions(submissions)

        self._log_submission_counts(problem_id, categorized)

        file_groups = self._load_submission_files(problem_id, categorized)
        pairs = self._create_comparison_pairs(problem_id, file_groups)

        return self._compare_all_pairs(problem_id, pairs)

    def _categorize_submissions(self, submissions: List[Dict]) -> Dict[str, List[Dict]]:
        """Categorize submissions by language and clone type."""
        categorized = {
            'java_clones': [],
            'python_clones': [],
            'python_non_clones': []
        }

        for sub in submissions:
            language = sub.get('language', '')
            clone_type = sub.get('clone_type', '')

            if clone_type == CLONE_LABEL:
                if language == LANGUAGE_JAVA:
                    categorized['java_clones'].append(sub)
                elif language == LANGUAGE_PYTHON:
                    categorized['python_clones'].append(sub)
            elif clone_type == NON_CLONE_LABEL and language == LANGUAGE_PYTHON:
                categorized['python_non_clones'].append(sub)

        for key in categorized:
            categorized[key].sort(key=lambda x: x.get('filename', ''))

        return categorized

    def _log_submission_counts(self, problem_id: str, categorized: Dict):
        """Log submission counts for a problem."""
        logging.info(
            f"{problem_id}: Found {len(categorized['java_clones'])} Java clones, "
            f"{len(categorized['python_clones'])} Python clones, "
            f"{len(categorized['python_non_clones'])} Python non-clones"
        )

    def _load_submission_files(self, problem_id: str, categorized: Dict) -> Dict[str, List[Tuple]]:
        """Load submission files and return file groups."""
        file_groups = {
            'java_clones': [],
            'python_clones': [],
            'python_non_clones': []
        }

        for group_name, submissions in categorized.items():
            for sub in submissions:
                try:
                    file_path = self._get_code_file_path(problem_id, sub)
                    file_groups[group_name].append((file_path, sub))
                except (FileNotFoundError, ValueError) as e:
                    logging.warning(f"{problem_id}: Skipping {group_name}: {e}")

        logging.info(
            f"{problem_id}: Loaded {len(file_groups['java_clones'])} Java files, "
            f"{len(file_groups['python_clones'])} Python clone files, "
            f"{len(file_groups['python_non_clones'])} Python non-clone files"
        )

        return file_groups

    def _create_comparison_pairs(self, problem_id: str, file_groups: Dict) -> List[Dict]:
        """Create balanced pairs for clone and non-clone comparisons."""
        pairs = []
        num_pairs = min(
            len(file_groups['java_clones']),
            len(file_groups['python_clones']),
            len(file_groups['python_non_clones'])
        )

        if num_pairs == 0:
            logging.warning(
                f"{problem_id}: Cannot create pairs - "
                f"Java: {len(file_groups['java_clones'])}, "
                f"Python clones: {len(file_groups['python_clones'])}, "
                f"Python non-clones: {len(file_groups['python_non_clones'])}"
            )
            return pairs

        logging.info(f"{problem_id}: Creating {num_pairs} clone and {num_pairs} non-clone pairs")

        for i in range(num_pairs):
            java_file, java_meta = file_groups['java_clones'][i]
            python_clone_file, python_clone_meta = file_groups['python_clones'][i]
            python_non_clone_file, python_non_clone_meta = file_groups['python_non_clones'][i]

            pairs.append({
                'code1': (java_file, java_meta),
                'code2': (python_clone_file, python_clone_meta),
                'expected_label': CLONE_LABEL
            })

            pairs.append({
                'code1': (java_file, java_meta),
                'code2': (python_non_clone_file, python_non_clone_meta),
                'expected_label': NON_CLONE_LABEL
            })

        return pairs

    def _compare_all_pairs(self, problem_id: str, pairs: List[Dict]) -> List[Dict]:
        """Compare all pairs and return results."""
        results = []

        for pair in pairs:
            code1_path, code1_meta = pair['code1']
            code2_path, code2_meta = pair['code2']
            expected_label = pair['expected_label']

            result = self._compare_pair(
                problem_id,
                code1_path, code1_meta,
                code2_path, code2_meta,
                expected_label
            )
            results.append(result)

        logging.info(f"{problem_id}: Generated {len(results)} comparisons")
        return results

    def _compare_pair(
            self,
            problem_id: str,
            code1_path: Path,
            code1_meta: Dict,
            code2_path: Path,
            code2_meta: Dict,
            expected_label: str
    ) -> Dict:
        """Compare a pair using the selected detection mode."""
        if self.detection_mode == DETECTION_MODE_DIRECT:
            return self._compare_pair_direct(
                problem_id, code1_path, code1_meta,
                code2_path, code2_meta, expected_label
            )
        elif self.detection_mode == DETECTION_MODE_ALGORITHM:
            return self._compare_pair_algorithm_based(
                problem_id, code1_path, code1_meta,
                code2_path, code2_meta, expected_label
            )
        else:
            raise ValueError(f"Unknown detection mode: {self.detection_mode}")

    def _compare_pair_direct(
            self,
            problem_id: str,
            code1_path: Path,
            code1_meta: Dict,
            code2_path: Path,
            code2_meta: Dict,
            expected_label: str
    ) -> Dict:
        """Direct comparison: Send code snippets directly to LLM."""
        code1_content = self._read_code_file(code1_path)
        code2_content = self._read_code_file(code2_path)

        message = [
            {"role": "system", "content": get_system_prompt_for_direct_clone_detection()},
            {"role": "user", "content": get_prompt_for_direct_clone_detection(
                code1_content, code1_meta.get('language', ''),
                code2_content, code2_meta.get('language', '')
            )}
        ]

        llm_response = self._generate_llm_response(message, max_tokens=800)
        parsed = self._parse_structured_response(llm_response)

        result = self._create_result_dict(
            problem_id, code1_path, code1_meta, code2_path, code2_meta,
            code1_content, code2_content, expected_label, parsed, llm_response,
            mode=DETECTION_MODE_DIRECT
        )

        logging.info(
            f"  [DIRECT] {code1_path.name} vs {code2_path.name} -> "
            f"Expected: {expected_label}, Predicted: {parsed['decision']}"
        )

        return result

    def _compare_pair_algorithm_based(
            self,
            problem_id: str,
            code1_path: Path,
            code1_meta: Dict,
            code2_path: Path,
            code2_meta: Dict,
            expected_label: str
    ) -> Dict:
        """Algorithm-based comparison: Generate algorithms first, then compare."""
        code1_content = self._read_code_file(code1_path)
        code2_content = self._read_code_file(code2_path)

        conversation = [
            {"role": "system", "content": get_system_prompt_for_algorithm_based_clone_detection()}
        ]

        algorithm1 = self._generate_algorithm(conversation, code1_content, code1_meta, code1_path.name)
        algorithm2 = self._generate_algorithm(conversation, code2_content, code2_meta, code2_path.name)

        logging.info("  Comparing algorithms...")
        conversation.append({
            "role": "user",
            "content": get_prompt_for_algorithm_based_clone_detection(algorithm1, algorithm2)
        })

        llm_response = self._generate_llm_response(conversation, max_tokens=800)
        parsed = self._parse_structured_response(llm_response)

        result = self._create_result_dict(
            problem_id, code1_path, code1_meta, code2_path, code2_meta,
            code1_content, code2_content, expected_label, parsed, llm_response,
            mode=DETECTION_MODE_ALGORITHM,
            algorithm1=algorithm1,
            algorithm2=algorithm2,
            conversation=conversation
        )

        logging.info(
            f"  [ALGORITHM] {code1_path.name} vs {code2_path.name} -> "
            f"Expected: {expected_label}, Predicted: {parsed['decision']}"
        )

        return result

    def _generate_algorithm(
            self,
            conversation: List[Dict],
            code_content: str,
            code_meta: Dict,
            filename: str
    ) -> str:
        """Generate algorithm description for code."""
        logging.info(f"  Generating algorithm for {filename}...")

        conversation.append({
            "role": "user",
            "content": get_prompt_to_generate_algorithm_from_code(
                code_content, code_meta.get('language', '')
            )
        })

        algorithm = self._generate_llm_response(conversation, max_tokens=1000)

        conversation.append({
            "role": "assistant",
            "content": algorithm
        })

        return algorithm

    def _create_result_dict(
            self,
            problem_id: str,
            code1_path: Path,
            code1_meta: Dict,
            code2_path: Path,
            code2_meta: Dict,
            code1_content: str,
            code2_content: str,
            expected_label: str,
            parsed: Dict,
            llm_response: str,
            mode: str,
            algorithm1: Optional[str] = None,
            algorithm2: Optional[str] = None,
            conversation: Optional[List[Dict]] = None
    ) -> Dict:
        """Create result dictionary."""
        result = {
            'problem_id': problem_id,
            'code1_file': code1_path.name,
            'code2_file': code2_path.name,
            'code1_language': code1_meta.get('language', ''),
            'code2_language': code2_meta.get('language', ''),
            'code1_submission_id': code1_meta.get('submission_id', ''),
            'code2_submission_id': code2_meta.get('submission_id', ''),
            'code2_original_problem': code2_meta.get('original_problem_id', problem_id),
            'code1_content': code1_content,
            'code2_content': code2_content,
            'expected_label': expected_label,
            'predicted_label': parsed['decision'],
            'is_correct': parsed['decision'] == expected_label,
            'detection_mode': mode,
            'rationale': parsed['rationale'],
            'llm_response': llm_response
        }

        if mode == DETECTION_MODE_ALGORITHM:
            result['algorithm1'] = algorithm1
            result['algorithm2'] = algorithm2
            result['conversation'] = conversation

        return result

    def _get_code_file_path(self, problem_id: str, metadata: Dict) -> Path:
        """Get file path for a code submission."""
        filename = metadata.get('filename', '')

        if not filename:
            error_msg = f"Missing filename for submission {metadata.get('submission_id', 'unknown')}"
            logging.error(error_msg)
            raise ValueError(error_msg)

        problem_dir = self.data_dir.joinpath(problem_id)

        if not problem_dir.exists():
            error_msg = f"Problem directory {problem_dir} does not exist"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        file_path = problem_dir.joinpath(filename)

        if not file_path.exists():
            available = list(problem_dir.glob('*'))
            logging.error(f"File not found: {file_path}")
            logging.error(f"Available files: {[f.name for f in available[:10]]}")
            raise FileNotFoundError(f"File not found: {file_path}")

        logging.debug(f"Found file: {filename}")
        return file_path

    def _read_code_file(self, file_path: Path) -> str:
        """Read code file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            raise

    def _read_metadata_file(self, metadata_file: Path) -> List[Dict]:
        """Read metadata CSV file."""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                submissions = list(reader)
            logging.debug(f"Read {len(submissions)} submissions from {metadata_file.name}")
            return submissions
        except Exception as e:
            logging.error(f"Error reading {metadata_file}: {e}")
            raise

    def _generate_llm_response(self, messages: List[Dict], max_tokens: int = 100) -> str:
        """Generate response from LLM with retry logic."""
        client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        attempts_remaining = MAX_RETRY_ATTEMPTS

        while attempts_remaining > 0:
            try:
                self._enforce_rate_limit()

                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

            except Exception as e:
                if self._is_rate_limit_error(e):
                    self._handle_rate_limit_error()
                    continue
                else:
                    attempts_remaining -= 1
                    logging.info(f"Attempt {MAX_RETRY_ATTEMPTS - attempts_remaining} failed: {e}")

                    if attempts_remaining == 0:
                        logging.error(f"Max retries reached. Last error: {e}")
                        raise

        return "exceptional case"

    def _is_rate_limit_error(self, error: Exception) -> bool:
        """Check if error is a rate limit error."""
        error_str = str(error)
        return "429" in error_str or "rate limit" in error_str.lower()

    def _handle_rate_limit_error(self):
        """Handle rate limit error by waiting."""
        logging.warning(f"Rate limit hit. Waiting {RATE_LIMIT_WAIT_SECONDS}s...")
        print(f"Rate limit hit. Waiting {RATE_LIMIT_WAIT_SECONDS}s...", flush=True)
        time.sleep(RATE_LIMIT_WAIT_SECONDS)

    def _enforce_rate_limit(self):
        """Enforce rate limiting for free models."""
        if not self.is_free_model:
            return

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=1)

        while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
            self.request_timestamps.popleft()

        if len(self.request_timestamps) >= FREE_MODEL_MAX_REQUESTS_PER_MINUTE:
            wait_until = self.request_timestamps[0] + timedelta(minutes=1)
            wait_seconds = (wait_until - current_time).total_seconds() + 1.0

            if wait_seconds > 0:
                logging.info(f"Rate limit reached. Waiting {wait_seconds:.2f}s...")
                time.sleep(wait_seconds)
                self._enforce_rate_limit()
                return

        self.request_timestamps.append(current_time)

    def _parse_structured_response(self, response: str) -> Dict[str, str]:
        """Parse structured LLM response to extract rationale and decision."""
        if not response:
            logging.warning("Empty LLM response")
            return {'rationale': '', 'decision': 'unknown'}

        try:
            response_upper = response.upper()
            rationale = self._extract_rationale(response, response_upper)
            decision = self._extract_decision(response, response_upper)

            return {'rationale': rationale, 'decision': decision}

        except Exception as e:
            logging.warning(f"Error parsing structured response: {e}")
            return {
                'rationale': response,
                'decision': self._parse_simple_response(response)
            }

    def _extract_rationale(self, response: str, response_upper: str) -> str:
        """Extract rationale section from response."""
        if 'RATIONALE:' not in response_upper:
            return ''

        start = response_upper.index('RATIONALE:') + len('RATIONALE:')
        end = response_upper.index('DECISION:') if 'DECISION:' in response_upper else len(response)
        return response[start:end].strip()

    def _extract_decision(self, response: str, response_upper: str) -> str:
        """Extract decision from response."""
        if 'DECISION:' not in response_upper:
            return self._parse_simple_response(response)

        start = response_upper.index('DECISION:') + len('DECISION:')
        decision_text = response[start:].strip().upper()

        decision_lines = [line.strip() for line in decision_text.split('\n') if line.strip()]
        if not decision_lines:
            return 'unknown'

        decision_word = decision_lines[0]
        if 'CLONE' in decision_word and 'NOT' not in decision_word:
            return CLONE_LABEL
        elif 'NOT_CLONE' in decision_word or 'NOT CLONE' in decision_word:
            return NON_CLONE_LABEL

        return 'unknown'

    def _parse_simple_response(self, response: str) -> str:
        """Parse simple LLM response to extract label."""
        if not response:
            return 'unknown'

        response_upper = response.upper().strip()
        lines = [line.strip() for line in response_upper.split('\n') if line.strip()]

        if lines:
            last_line = lines[-1]
            if last_line == "CLONE":
                return CLONE_LABEL
            elif last_line in ["NOT_CLONE", "NOT CLONE", "NOTCLONE"]:
                return NON_CLONE_LABEL

        if "NOT_CLONE" in response_upper.replace(" ", "_") or "NOT CLONE" in response_upper:
            return NON_CLONE_LABEL
        elif "CLONE" in response_upper and "NOT" not in response_upper:
            return CLONE_LABEL

        logging.warning(f"Could not parse response: {response[:200]}")
        return 'unknown'

    def _save_all_results(self, results: List[Dict]):
        """Save all results to CSV, JSON, and Excel."""
        if results:
            self._save_csv_results(results)
            self._save_json_results(results)
            self._save_excel_results(results)

        self._calculate_and_save_metrics(results)

    def _save_csv_results(self, results: List[Dict]):
        """Save results to CSV file."""
        csv_output = self.output_root.joinpath(f"clone_detection_results_{self.detection_mode}.csv")

        fieldnames = [
            'problem_id', 'code1_file', 'code2_file',
            'code1_language', 'code2_language',
            'code1_submission_id', 'code2_submission_id',
            'code2_original_problem',
            'expected_label', 'predicted_label', 'is_correct',
            'detection_mode', 'rationale'
        ]

        with open(csv_output, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(results)

        logging.info(f"CSV results saved to {csv_output}")

    def _save_json_results(self, results: List[Dict]):
        """Save results to JSON file."""
        json_output = self.output_root.joinpath(f"clone_detection_results_{self.detection_mode}.json")

        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)

        logging.info(f"JSON results saved to {json_output}")

    def _save_excel_results(self, results: List[Dict]):
        """Save detailed results to Excel."""
        excel_output = self.output_root.joinpath(f"clone_detection_detailed_{self.detection_mode}.xlsx")

        if not results:
            logging.warning("No results to save to Excel")
            return

        df = self._create_results_dataframe(results)

        with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Clone Detection Results', index=False)
            self._format_excel_worksheet(writer.sheets['Clone Detection Results'], df)

        logging.info(f"Excel results saved to {excel_output}")
        print(f"Detailed results exported to: {excel_output}")

    def _create_results_dataframe(self, results: List[Dict]) -> pd.DataFrame:
        """Create DataFrame from results."""
        excel_data = []

        for result in results:
            row = {
                'Problem ID': result['problem_id'],
                'Code 1 File': result['code1_file'],
                'Code 2 File': result['code2_file'],
                'Code 1 Language': result['code1_language'],
                'Code 2 Language': result['code2_language'],
                'Expected Label': result['expected_label'],
                'Predicted Label': result['predicted_label'],
                'Is Correct': result['is_correct'],
                'Code 1': result.get('code1_content', ''),
                'Code 2': result.get('code2_content', ''),
                'Rationale': result.get('rationale', ''),
                'Code 1 Submission ID': result['code1_submission_id'],
                'Code 2 Submission ID': result['code2_submission_id'],
                'Code 2 Original Problem': result['code2_original_problem'],
                'Detection Mode': result['detection_mode'],
                'Full LLM Response': result['llm_response']
            }

            if self.detection_mode == DETECTION_MODE_ALGORITHM:
                row['Algorithm 1'] = result.get('algorithm1', '')
                row['Algorithm 2'] = result.get('algorithm2', '')

            excel_data.append(row)

        df = pd.DataFrame(excel_data)
        return df[self._get_column_order()]

    def _get_column_order(self) -> List[str]:
        """Get column order for Excel output."""
        base_columns = [
            'Problem ID', 'Code 1 File', 'Code 2 File',
            'Code 1 Language', 'Code 2 Language',
            'Expected Label', 'Predicted Label', 'Is Correct',
            'Code 1', 'Code 2'
        ]

        if self.detection_mode == DETECTION_MODE_ALGORITHM:
            base_columns.extend(['Algorithm 1', 'Algorithm 2'])

        base_columns.extend([
            'Rationale', 'Full LLM Response',
            'Code 1 Submission ID', 'Code 2 Submission ID',
            'Code 2 Original Problem', 'Detection Mode'
        ])

        return base_columns

    def _format_excel_worksheet(self, worksheet, df: pd.DataFrame):
        """Format Excel worksheet."""
        code_columns = ['Code 1', 'Code 2', 'Algorithm 1', 'Algorithm 2']
        text_columns = ['Rationale', 'Full LLM Response']

        for idx, col in enumerate(df.columns, 1):
            column_letter = worksheet.cell(1, idx).column_letter

            if col in code_columns:
                worksheet.column_dimensions[column_letter].width = 50
            elif col in text_columns:
                worksheet.column_dimensions[column_letter].width = 80
            else:
                worksheet.column_dimensions[column_letter].width = 20

        worksheet.freeze_panes = 'A2'

    def _calculate_and_save_metrics(self, results: List[Dict]):
        """Calculate and save evaluation metrics."""
        if not results:
            return

        metrics = self._calculate_metrics(results)
        self._save_metrics(metrics)

    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate evaluation metrics."""
        total = len(results)

        confusion_matrix = self._calculate_confusion_matrix(results)
        overall_metrics = self._calculate_overall_metrics(confusion_matrix, total)
        per_label_metrics = self._calculate_per_label_metrics(results)

        return {
            'detection_mode': self.detection_mode,
            'total_comparisons': total,
            **confusion_matrix,
            **overall_metrics,
            **per_label_metrics
        }

    def _calculate_confusion_matrix(self, results: List[Dict]) -> Dict:
        """Calculate confusion matrix values."""
        return {
            'true_positives': sum(
                1 for r in results
                if r['expected_label'] == CLONE_LABEL and r['predicted_label'] == CLONE_LABEL
            ),
            'false_positives': sum(
                1 for r in results
                if r['expected_label'] == NON_CLONE_LABEL and r['predicted_label'] == CLONE_LABEL
            ),
            'true_negatives': sum(
                1 for r in results
                if r['expected_label'] == NON_CLONE_LABEL and r['predicted_label'] == NON_CLONE_LABEL
            ),
            'false_negatives': sum(
                1 for r in results
                if r['expected_label'] == CLONE_LABEL and r['predicted_label'] == NON_CLONE_LABEL
            )
        }

    def _calculate_overall_metrics(self, confusion_matrix: Dict, total: int) -> Dict:
        """Calculate overall performance metrics."""
        tp = confusion_matrix['true_positives']
        fp = confusion_matrix['false_positives']
        tn = confusion_matrix['true_negatives']
        fn = confusion_matrix['false_negatives']

        correct = tp + tn
        accuracy = correct / total if total > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'correct_predictions': correct,
            'overall_accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4)
        }

    def _calculate_per_label_metrics(self, results: List[Dict]) -> Dict:
        """Calculate per-label metrics."""
        clone_results = [r for r in results if r['expected_label'] == CLONE_LABEL]
        non_clone_results = [r for r in results if r['expected_label'] == NON_CLONE_LABEL]

        clone_correct = sum(1 for r in clone_results if r['is_correct'])
        non_clone_correct = sum(1 for r in non_clone_results if r['is_correct'])

        return {
            'clone_comparisons': len(clone_results),
            'clone_correct': clone_correct,
            'clone_accuracy': round(clone_correct / len(clone_results), 4) if clone_results else 0,
            'non_clone_comparisons': len(non_clone_results),
            'non_clone_correct': non_clone_correct,
            'non_clone_accuracy': round(non_clone_correct / len(non_clone_results), 4) if non_clone_results else 0
        }

    def _save_metrics(self, metrics: Dict):
        """Save metrics to JSON file."""
        metrics_file = self.output_root.joinpath(f"metrics_{self.detection_mode}.json")

        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"Metrics saved to {metrics_file}")

    def _log_completion_stats(self, stats: Dict):
        """Log completion statistics."""
        if stats['failed']:
            logging.warning(f"Failed problems ({len(stats['failed'])}):")
            for prob_id, error in stats['failed']:
                logging.warning(f"  - {prob_id}: {error}")

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Detect code clones using Large Language Models"
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='LLM model name (e.g., gpt-4o, deepseek-r1)'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=[DETECTION_MODE_DIRECT, DETECTION_MODE_ALGORITHM],
        required=True,
        help=f'Detection mode: {DETECTION_MODE_DIRECT} or {DETECTION_MODE_ALGORITHM}'
    )

    return parser.parse_args()

def main():
    args = parse_arguments()

    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    dataset_root = project_root.joinpath("preprocessed_dataset")
    output_root = project_root.joinpath("reports")

    with LLMCloneDetector(
        dataset_root=dataset_root,
        output_root=output_root,
        model=args.model,
        detection_mode=args.mode
    ) as detector:
        detector.detect_clones()

if __name__ == "__main__":
    main()
