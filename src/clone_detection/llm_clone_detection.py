import pandas as pd
import csv
import json
import logging
import os
import time
from collections import deque
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from prompts import (
    get_system_prompt_for_direct_clone_detection, 
    get_prompt_for_direct_clone_detection,
    get_system_prompt_for_algorithm_based_clone_detection,
    get_prompt_to_generate_algorithm_from_code,
    get_prompt_for_algorithm_based_clone_detection
)
from tqdm import tqdm

load_dotenv(override=True)

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename="logs/llm_clone_detection.log",
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class LLMCloneDetector:
    def __init__(self, dataset_root, output_root, detection_mode='direct'):
        """
        Initialize LLM Clone Detector.
        
        Args:
            dataset_root: Path to preprocessed dataset
            output_root: Path to save results
            detection_mode: 'direct' or 'algorithm_based'
                - 'direct': Directly compare code snippets
                - 'algorithm_based': Generate algorithms first, then compare
        """
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(output_root)
        self.base_url = "https://openrouter.ai/api/v1"
        self.api_key = os.getenv("API_KEY")
        self.model = "openai/gpt-oss-20b:free"
        self.detection_mode = detection_mode
        
        self.is_open_source_model = ":free" in self.model
        self.open_source_model_max_requests_per_minute = 16
        self.request_timestamps = deque()

    def __enter__(self):
        self.data_dir = self.dataset_root.joinpath("data")
        self.metadata_dir = self.dataset_root.joinpath("metadata")

        if not self.dataset_root.exists():
            logging.error(f"Directory {self.dataset_root} does not exist")
            raise FileNotFoundError(f"Directory {self.dataset_root} does not exist")

        if not self.data_dir.exists():
            logging.error(f"Data directory {self.data_dir} does not exist")
            raise FileNotFoundError(f"Data directory {self.data_dir} does not exist")

        if not self.metadata_dir.exists():
            logging.error(f"Metadata directory {self.metadata_dir} does not exist")
            raise FileNotFoundError(f"Metadata directory {self.metadata_dir} does not exist")

        if not self.api_key:
            logging.error("API_KEY not found in environment variables")
            raise ValueError("API_KEY not found in environment variables. Please set it in .env file")

        self.output_root.mkdir(parents=True, exist_ok=True)
        logging.info(f"Initialized LLMCloneDetector with mode: {self.detection_mode}")

        return self

    def detect_clones(self):
        """Process all problems for clone detection."""
        metadata_files = sorted([
            f for f in self.metadata_dir.glob("p*.csv")
        ])

        metadata_files = metadata_files[:1]  # Test with 1 problem

        logging.info(f"Found {len(metadata_files)} problems to process")

        all_results = []
        successful_problems = 0
        failed_problems = []

        for metadata_file in tqdm(
            metadata_files, 
            desc="Problems",
            bar_format="{desc:<10.10}{percentage:3.0f}%|{bar:10}{r_bar}",
            leave=True
        ):
            problem_id = metadata_file.stem

            logging.info(f"Processing {problem_id}...")

            try:
                results = self._process_problem(problem_id, metadata_file)
                all_results.extend(results)
                successful_problems += 1
                logging.info(f"Completed {len(results)} comparisons for {problem_id}")
            except Exception as e:
                logging.error(f"Error processing {problem_id}: {e}")
                failed_problems.append((problem_id, str(e)))

        self._save_results(all_results)

        return all_results

    def _process_problem(self, problem_id, metadata_file):
        """Process a single problem for clone detection with balanced pairs."""
        submissions = self._read_metadata_file(metadata_file)

        java_clones = []
        python_clones = []
        python_non_clones = []

        for sub in submissions:
            language = sub.get('language', '')
            clone_type = sub.get('clone_type', '')

            if clone_type == 'clone':
                if language == 'Java':
                    java_clones.append(sub)
                elif language == 'Python':
                    python_clones.append(sub)
            elif clone_type == 'non_clone' and language == 'Python':
                python_non_clones.append(sub)

        logging.info(f"{problem_id}: Found {len(java_clones)} Java clones, {len(python_clones)} Python clones, {len(python_non_clones)} Python non-clones in metadata")

        # Sort by filename to ensure sequential pairing (_1 with _1, _2 with _2, etc.)
        java_clones.sort(key=lambda x: x.get('filename', ''))
        python_clones.sort(key=lambda x: x.get('filename', ''))
        python_non_clones.sort(key=lambda x: x.get('filename', ''))

        # Get file paths using filenames from metadata
        java_clone_files = []
        for java_sub in java_clones:
            try:
                file_path = self._get_code_file_path(problem_id, java_sub)
                java_clone_files.append((file_path, java_sub))
            except (FileNotFoundError, ValueError) as e:
                logging.warning(f"{problem_id}: Skipping Java clone: {e}")

        python_clone_files = []
        for python_sub in python_clones:
            try:
                file_path = self._get_code_file_path(problem_id, python_sub)
                python_clone_files.append((file_path, python_sub))
            except (FileNotFoundError, ValueError) as e:
                logging.warning(f"{problem_id}: Skipping Python clone: {e}")

        python_non_clone_files = []
        for python_sub in python_non_clones:
            try:
                file_path = self._get_code_file_path(problem_id, python_sub)
                python_non_clone_files.append((file_path, python_sub))
            except (FileNotFoundError, ValueError) as e:
                logging.warning(f"{problem_id}: Skipping Python non-clone: {e}")

        logging.info(f"{problem_id}: Successfully loaded {len(java_clone_files)} Java files, {len(python_clone_files)} Python clone files, {len(python_non_clone_files)} Python non-clone files")

        results = []
        num_pairs = min(len(java_clone_files), len(python_clone_files), len(python_non_clone_files))

        if num_pairs == 0:
            logging.warning(f"{problem_id}: Cannot create pairs - Java: {len(java_clone_files)}, Python clones: {len(python_clone_files)}, Python non-clones: {len(python_non_clone_files)}")
            return results

        logging.info(f"{problem_id}: Creating {num_pairs} clone pairs and {num_pairs} non-clone pairs")

        for i in range(num_pairs):
            java_file, java_meta = java_clone_files[i]
            python_clone_file, python_clone_meta = python_clone_files[i]
            python_non_clone_file, python_non_clone_meta = python_non_clone_files[i]

            result_clone = self._compare_pair(
                problem_id,
                java_file, java_meta,
                python_clone_file, python_clone_meta,
                expected_label="clone"
            )
            results.append(result_clone)

            result_non_clone = self._compare_pair(
                problem_id,
                java_file, java_meta,
                python_non_clone_file, python_non_clone_meta,
                expected_label="non_clone"
            )
            results.append(result_non_clone)

        logging.info(f"{problem_id}: Generated {len(results)} comparisons ({num_pairs} clones, {num_pairs} non-clones)")

        return results

    def _compare_pair(self, problem_id, code1_path, code1_meta, code2_path, code2_meta, expected_label):
        """Compare a pair of code snippets using the selected detection mode."""
        if self.detection_mode == 'direct':
            return self._compare_pair_direct(problem_id, code1_path, code1_meta, code2_path, code2_meta, expected_label)
        elif self.detection_mode == 'algorithm_based':
            return self._compare_pair_algorithm_based(problem_id, code1_path, code1_meta, code2_path, code2_meta, expected_label)
        else:
            raise ValueError(f"Unknown detection mode: {self.detection_mode}")

    def _compare_pair_direct(self, problem_id, code1_path, code1_meta, code2_path, code2_meta, expected_label):
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

        llm_response = self._generate_response_using_llm(message, max_tokens=800)  # Increased for rationale
        parsed = self._parse_structured_llm_response(llm_response)

        result = {
            'problem_id': problem_id,
            'code1_file': code1_path.name,
            'code2_file': code2_path.name,
            'code1_language': code1_meta.get('language', ''),
            'code2_language': code2_meta.get('language', ''),
            'code1_submission_id': code1_meta.get('submission_id', ''),
            'code2_submission_id': code2_meta.get('submission_id', ''),
            'code2_original_problem': code2_meta.get('original_problem_id', problem_id),
            'code1_content': code1_content,  # Store for Excel
            'code2_content': code2_content,  # Store for Excel
            'expected_label': expected_label,
            'predicted_label': parsed['decision'],
            'is_correct': parsed['decision'] == expected_label,
            'detection_mode': 'direct',
            'rationale': parsed['rationale'],
            'llm_response': llm_response
        }

        logging.info(f"  [DIRECT] Compared: {code1_path.name} vs {code2_path.name} -> Expected: {expected_label}, Predicted: {parsed['decision']}")

        return result

    def _compare_pair_algorithm_based(self, problem_id, code1_path, code1_meta, code2_path, code2_meta, expected_label):
        """Algorithm-based comparison: Generate algorithms first, then compare."""
        code1_content = self._read_code_file(code1_path)
        code2_content = self._read_code_file(code2_path)

        # Initialize conversation with system prompt
        conversation = [
            {"role": "system", "content": get_system_prompt_for_algorithm_based_clone_detection()}
        ]

        # Step 1: Generate algorithm for code1
        logging.info(f"  Generating algorithm for {code1_path.name}...")
        conversation.append({
            "role": "user",
            "content": get_prompt_to_generate_algorithm_from_code(
                code1_content, code1_meta.get('language', '')
            )
        })
        
        algorithm1 = self._generate_response_using_llm(conversation, max_tokens=1000)
        
        # Add assistant's response to conversation
        conversation.append({
            "role": "assistant",
            "content": algorithm1
        })

        # Step 2: Generate algorithm for code2
        logging.info(f"  Generating algorithm for {code2_path.name}...")
        conversation.append({
            "role": "user",
            "content": get_prompt_to_generate_algorithm_from_code(
                code2_content, code2_meta.get('language', '')
            )
        })
        
        algorithm2 = self._generate_response_using_llm(conversation, max_tokens=1000)
        
        # Add assistant's response to conversation
        conversation.append({
            "role": "assistant",
            "content": algorithm2
        })

        # Step 3: Compare algorithms
        logging.info(f"  Comparing algorithms...")
        conversation.append({
            "role": "user",
            "content": get_prompt_for_algorithm_based_clone_detection(
                algorithm1, algorithm2
            )
        })
        
        llm_response = self._generate_response_using_llm(conversation, max_tokens=800)  # Increased for rationale
        parsed = self._parse_structured_llm_response(llm_response)

        result = {
            'problem_id': problem_id,
            'code1_file': code1_path.name,
            'code2_file': code2_path.name,
            'code1_language': code1_meta.get('language', ''),
            'code2_language': code2_meta.get('language', ''),
            'code1_submission_id': code1_meta.get('submission_id', ''),
            'code2_submission_id': code2_meta.get('submission_id', ''),
            'code2_original_problem': code2_meta.get('original_problem_id', problem_id),
            'code1_content': code1_content,  # Store for Excel
            'code2_content': code2_content,  # Store for Excel
            'algorithm1': algorithm1,
            'algorithm2': algorithm2,
            'expected_label': expected_label,
            'predicted_label': parsed['decision'],
            'is_correct': parsed['decision'] == expected_label,
            'detection_mode': 'algorithm_based',
            'rationale': parsed['rationale'],
            'llm_response': llm_response,
            'conversation': conversation
        }

        logging.info(f"  [ALGORITHM] Compared: {code1_path.name} vs {code2_path.name} -> Expected: {expected_label}, Predicted: {parsed['decision']}")

        return result

    def _get_code_file_path(self, problem_id, metadata):
        """Get the file path for a code submission using filename from metadata."""
        filename = metadata.get('filename', '')
        
        if not filename:
            logging.error(f"No filename found in metadata for submission {metadata.get('submission_id', 'unknown')}")
            raise ValueError(f"Missing filename in metadata")
        
        problem_dir = self.data_dir.joinpath(problem_id)
        
        if not problem_dir.exists():
            logging.error(f"Problem directory {problem_dir} does not exist")
            raise FileNotFoundError(f"Problem directory {problem_dir} does not exist")
        
        file_path = problem_dir.joinpath(filename)
        
        if not file_path.exists():
            available = list(problem_dir.glob('*'))
            logging.error(f"File not found: {file_path}")
            logging.error(f"Available files in {problem_dir}: {[f.name for f in available[:10]]}")
            raise FileNotFoundError(f"File not found: {file_path}")
        
        logging.debug(f"Found file: {filename}")
        return file_path

    def _read_code_file(self, file_path):
        """Read code file content."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading {file_path}: {e}")
            raise

    def _wait_if_request_limit_reached(self):
        """Wait if request limit is reached for free models."""
        if not self.is_open_source_model:
            return

        current_time = datetime.now()
        cutoff_time = current_time - timedelta(minutes=1)
        
        while self.request_timestamps and self.request_timestamps[0] < cutoff_time:
            self.request_timestamps.popleft()

        if len(self.request_timestamps) >= self.open_source_model_max_requests_per_minute:
            time_to_wait = (self.request_timestamps[0] + timedelta(minutes=1) - current_time).total_seconds() + 1.0
            if time_to_wait > 0:
                logging.info(f"Per minute request limit reached for free model. Waiting for {time_to_wait:.2f} seconds...")
                
                time.sleep(time_to_wait)
                self._wait_if_request_limit_reached()
                return

        self.request_timestamps.append(current_time)

    def _generate_response_using_llm(self, message_log, max_tokens=100):
        """Generate response from LLM with retry logic and rate limiting."""
        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )

        response = "exceptional case"
        is_success = False
        max_attempts = 5
        
        while max_attempts > 0:
            try:
                self._wait_if_request_limit_reached()

                response = client.chat.completions.create(
                    model=self.model,
                    messages=message_log,
                    temperature=0,
                    max_tokens=max_tokens,
                )
                is_success = True
                break
            except Exception as e:
                error_str = str(e)
                
                # ✅ Check if it's a rate limit error (HTTP 429)
                if "429" in error_str or "rate limit" in error_str.lower():
                    # Don't decrement attempts for rate limits
                    logging.warning(f"Rate limit hit during request. Waiting 65 seconds before retry...")
                    print(f"⏳ Rate limit hit. Waiting 65s...", flush=True)
                    time.sleep(65)  # Wait for rate limit to reset
                    # Don't decrement max_attempts - just retry
                    continue
                else:
                    # For other errors, decrement attempts
                    max_attempts -= 1
                    logging.info(f"Attempt {5 - max_attempts} failed: {e}. Retrying...")
                    
                if max_attempts == 0:
                    logging.error(f"Maximum retry attempts reached. Last error: {e}")
                    raise

        if not is_success:
            return response

        return response.choices[0].message.content

    def _parse_llm_response(self, response):
        """Parse LLM response to extract label from potentially verbose output."""
        if not response:
            logging.warning("Empty LLM response")
            return "unknown"
        
        response_upper = response.upper().strip()
        
        # Try to find answer in last line first
        lines = [line.strip() for line in response_upper.split('\n') if line.strip()]
        
        if lines:
            last_line = lines[-1]
            if last_line == "CLONE":
                return "clone"
            elif last_line in ["NOT_CLONE", "NOT CLONE", "NOTCLONE"]:
                return "non_clone"
        
        # Fallback: check entire response
        if "NOT_CLONE" in response_upper.replace(" ", "_") or "NOT CLONE" in response_upper:
            return "non_clone"
        elif "CLONE" in response_upper and "NOT" not in response_upper:
            return "clone"
        
        # Log for debugging
        logging.warning(f"Could not parse response: {response[:200]}")
        return "unknown"

    def _parse_structured_llm_response(self, response):
        """Parse structured LLM response to extract rationale and decision."""
        if not response:
            logging.warning("Empty LLM response")
            return {
                'rationale': '',
                'decision': 'unknown'
            }
        
        # Initialize default values
        rationale = ''
        decision = 'unknown'
        
        try:
            response_upper = response.upper()
            
            # Extract RATIONALE section
            if 'RATIONALE:' in response_upper:
                rationale_start = response_upper.index('RATIONALE:') + len('RATIONALE:')
                rationale_end = response_upper.index('DECISION:') if 'DECISION:' in response_upper else len(response)
                rationale = response[rationale_start:rationale_end].strip()
            
            # Extract DECISION section
            if 'DECISION:' in response_upper:
                decision_start = response_upper.index('DECISION:') + len('DECISION:')
                decision_text = response[decision_start:].strip().upper()
                
                # Extract just the decision word
                decision_lines = [line.strip() for line in decision_text.split('\n') if line.strip()]
                if decision_lines:
                    decision_word = decision_lines[0]
                    if 'CLONE' in decision_word and 'NOT' not in decision_word:
                        decision = 'clone'
                    elif 'NOT_CLONE' in decision_word or 'NOT CLONE' in decision_word:
                        decision = 'non_clone'
            
            # Fallback parsing if sections not found
            if decision == 'unknown':
                decision = self._parse_llm_response(response)
                if not rationale:
                    rationale = response  # Use full response as rationale
            
        except Exception as e:
            logging.warning(f"Error parsing structured response: {e}")
            decision = self._parse_llm_response(response)
            rationale = response
        
        return {
            'rationale': rationale,
            'decision': decision
        }

    def _read_metadata_file(self, metadata_file):
        """Read metadata CSV file."""
        submissions = []

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    submissions.append(row)
            logging.debug(f"Successfully read {len(submissions)} submissions from {metadata_file.name}")
        except Exception as e:
            logging.error(f"Error reading {metadata_file}: {e}")
            raise

        return submissions

    def _save_results(self, results):
        """Save detection results to CSV, JSON, and Excel."""
        csv_output = self.output_root.joinpath(f"clone_detection_results_{self.detection_mode}.csv")
        json_output = self.output_root.joinpath(f"clone_detection_results_{self.detection_mode}.json")

        if results:
            fieldnames = [
                'problem_id', 'code1_file', 'code2_file',
                'code1_language', 'code2_language',
                'code1_submission_id', 'code2_submission_id',
                'code2_original_problem',
                'expected_label', 'predicted_label', 'is_correct',
                'detection_mode', 'rationale'
            ]

            with open(csv_output, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for result in results:
                    row = {k: v for k, v in result.items() if k in fieldnames}
                    writer.writerow(row)

            with open(json_output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

            logging.info(f"Results saved to {csv_output} and {json_output}")
            
            self._save_results_to_excel(results)

        self._calculate_metrics(results)

    def _calculate_metrics(self, results):
        """Calculate and save evaluation metrics including precision, recall, and F1 score."""
        if not results:
            return

        total = len(results)
        
        # Calculate True Positives, False Positives, True Negatives, False Negatives
        true_positives = sum(1 for r in results if r['expected_label'] == 'clone' and r['predicted_label'] == 'clone')
        false_positives = sum(1 for r in results if r['expected_label'] == 'non_clone' and r['predicted_label'] == 'clone')
        true_negatives = sum(1 for r in results if r['expected_label'] == 'non_clone' and r['predicted_label'] == 'non_clone')
        false_negatives = sum(1 for r in results if r['expected_label'] == 'clone' and r['predicted_label'] == 'non_clone')
        
        # Calculate overall metrics
        correct = true_positives + true_negatives
        accuracy = correct / total if total > 0 else 0
        
        # Precision: TP / (TP + FP)
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # Recall: TP / (TP + FN)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Calculate per-label metrics
        clone_results = [r for r in results if r['expected_label'] == 'clone']
        non_clone_results = [r for r in results if r['expected_label'] == 'non_clone']

        clone_correct = sum(1 for r in clone_results if r['is_correct'])
        non_clone_correct = sum(1 for r in non_clone_results if r['is_correct'])

        clone_accuracy = clone_correct / len(clone_results) if clone_results else 0
        non_clone_accuracy = non_clone_correct / len(non_clone_results) if non_clone_results else 0

        metrics = {
            'detection_mode': self.detection_mode,
            'total_comparisons': total,
            
            # Confusion Matrix
            'true_positives': true_positives,
            'false_positives': false_positives,
            'true_negatives': true_negatives,
            'false_negatives': false_negatives,
            
            # Overall Performance Metrics
            'correct_predictions': correct,
            'overall_accuracy': round(accuracy, 4),
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1_score': round(f1_score, 4),
            
            # Per-Label Metrics
            'clone_comparisons': len(clone_results),
            'clone_correct': clone_correct,
            'clone_accuracy': round(clone_accuracy, 4),
            
            'non_clone_comparisons': len(non_clone_results),
            'non_clone_correct': non_clone_correct,
            'non_clone_accuracy': round(non_clone_accuracy, 4)
        }

        metrics_file = self.output_root.joinpath(f"metrics_{self.detection_mode}.json")
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"Metrics saved to {metrics_file}")
        
        # Print metrics summary to console
        print(f"\n{'='*60}")
        print(f"EVALUATION METRICS ({self.detection_mode.upper()} MODE)")
        print(f"{'='*60}")
        print(f"Total Comparisons: {total}")
        print(f"\nConfusion Matrix:")
        print(f"  True Positives (TP):  {true_positives:4d}  (Correctly identified clones)")
        print(f"  False Positives (FP): {false_positives:4d}  (Non-clones incorrectly labeled as clones)")
        print(f"  True Negatives (TN):  {true_negatives:4d}  (Correctly identified non-clones)")
        print(f"  False Negatives (FN): {false_negatives:4d}  (Clones incorrectly labeled as non-clones)")
        print(f"\nOverall Performance:")
        print(f"  Accuracy:  {accuracy:.4f} ({correct}/{total})")
        print(f"  Precision: {precision:.4f} (TP / (TP + FP))")
        print(f"  Recall:    {recall:.4f} (TP / (TP + FN))")
        print(f"  F1 Score:  {f1_score:.4f}")
        print(f"\nPer-Label Accuracy:")
        print(f"  Clone Detection:     {clone_accuracy:.4f} ({clone_correct}/{len(clone_results)})")
        print(f"  Non-Clone Detection: {non_clone_accuracy:.4f} ({non_clone_correct}/{len(non_clone_results)})")
        print(f"{'='*60}\n")

    def _save_results_to_excel(self, results):
        """Save detailed results to Excel for analysis."""
        excel_output = self.output_root.joinpath(f"clone_detection_detailed_{self.detection_mode}.xlsx")
        
        if not results:
            logging.warning("No results to save to Excel")
            return
        
        # Prepare data for Excel
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
            
            # Add algorithm columns if algorithm-based mode
            if self.detection_mode == 'algorithm_based':
                row['Algorithm 1'] = result.get('algorithm1', '')
                row['Algorithm 2'] = result.get('algorithm2', '')
            
            excel_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(excel_data)
        
        # Reorder columns for better readability
        if self.detection_mode == 'direct':
            column_order = [
                'Problem ID', 'Code 1 File', 'Code 2 File',
                'Code 1 Language', 'Code 2 Language',
                'Expected Label', 'Predicted Label', 'Is Correct',
                'Code 1', 'Code 2',
                'Rationale', 'Full LLM Response',
                'Code 1 Submission ID', 'Code 2 Submission ID', 
                'Code 2 Original Problem', 'Detection Mode'
            ]
        else:  # algorithm_based
            column_order = [
                'Problem ID', 'Code 1 File', 'Code 2 File',
                'Code 1 Language', 'Code 2 Language',
                'Expected Label', 'Predicted Label', 'Is Correct',
                'Code 1', 'Code 2',
                'Algorithm 1', 'Algorithm 2',
                'Rationale', 'Full LLM Response',
                'Code 1 Submission ID', 'Code 2 Submission ID',
                'Code 2 Original Problem', 'Detection Mode'
            ]
        
        df = df[column_order]
        
        # Write to Excel with formatting
        with pd.ExcelWriter(excel_output, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Clone Detection Results', index=False)
            
            # Get the worksheet
            worksheet = writer.sheets['Clone Detection Results']
            
            # Adjust column widths
            for idx, col in enumerate(df.columns, 1):
                if col in ['Code 1', 'Code 2', 'Algorithm 1', 'Algorithm 2']:
                    worksheet.column_dimensions[worksheet.cell(1, idx).column_letter].width = 50
                elif col in ['Rationale', 'Full LLM Response']:
                    worksheet.column_dimensions[worksheet.cell(1, idx).column_letter].width = 80
                else:
                    worksheet.column_dimensions[worksheet.cell(1, idx).column_letter].width = 20
            
            # Freeze the header row
            worksheet.freeze_panes = 'A2'
        
        logging.info(f"Detailed results saved to Excel: {excel_output}")
        print(f"Detailed results exported to: {excel_output}")

    def __exit__(self, exception, _, __):
        if exception is None:
            print(f"Clone detection ({self.detection_mode} mode) completed successfully")
            logging.info(f"Clone detection ({self.detection_mode} mode) completed successfully")
        else:
            print(f"Clone detection failed due to {exception}")
            logging.error(f"Clone detection failed due to {exception}")


def main():
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent

    DATASET_ROOT = PROJECT_ROOT.joinpath("preprocessed_dataset")
    OUTPUT_ROOT = PROJECT_ROOT.joinpath("reports")

    # Choose detection mode: 'direct' or 'algorithm_based'
    DETECTION_MODE = 'direct'

    with LLMCloneDetector(DATASET_ROOT, OUTPUT_ROOT, detection_mode=DETECTION_MODE) as detector:
        detector.detect_clones()

if __name__ == "__main__":
    main()
