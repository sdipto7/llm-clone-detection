import csv
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/meta_data_organizer.log",
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

DEFAULT_TARGET_COUNT = 100
DEFAULT_SEED = 42
MIN_SUBMISSIONS_PER_LANGUAGE = 5
NON_CLONE_COUNT = 5
ACCEPTED_STATUS = 'Accepted'
JAVA_LANGUAGE = 'Java'
PYTHON_LANGUAGE = 'Python'

class MetadataOrganizer:
    """Organizes and curates metadata for code clone detection dataset."""

    def __init__(self, dataset_root: Path, output_root: Path):
        self.dataset_root = dataset_root
        self.output_root = output_root
        self.metadata_dir = self.dataset_root.joinpath("metadata")
        self.output_metadata_dir = self.output_root.joinpath("metadata")

    def __enter__(self):
        self._validate_directories()
        self.output_metadata_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            f"Initialized MetadataOrganizer with dataset_root={self.dataset_root}, "
            f"output_root={self.output_root}"
        )

        return self

    def __exit__(self, exception, _, __):
        if exception is None:
            print("Metadata organization completed successfully")
            logging.info("Metadata organization completed successfully")
        else:
            print(f"Metadata organization failed due to {exception}")
            logging.error(f"Metadata organization failed due to {exception}")

    def process_metadata(self, target_count: int = DEFAULT_TARGET_COUNT, seed: int = DEFAULT_SEED) -> List[Dict]:
        """Process metadata files and create curated dataset."""
        random.seed(seed)
        logging.info(f"Starting metadata processing with target_count={target_count}, seed={seed}")

        suitable_problems = self._collect_suitable_problems(target_count)
        self._assign_non_clones(suitable_problems, seed)
        self._create_curated_metadata(suitable_problems)
        self._create_summary_report(suitable_problems)

        return suitable_problems

    def _validate_directories(self):
        """Validate required directories exist."""
        if not self.dataset_root.exists():
            error_msg = f"Directory {self.dataset_root} does not exist"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

        if not self.metadata_dir.exists():
            error_msg = f"Metadata directory {self.metadata_dir} does not exist"
            logging.error(error_msg)
            raise FileNotFoundError(error_msg)

    def _collect_suitable_problems(self, target_count: int) -> List[Dict]:
        """Collect problems with sufficient submissions."""
        metadata_files = sorted(self.metadata_dir.glob("p*.csv"))
        logging.info(f"Found {len(metadata_files)} problem metadata files")

        suitable_problems = []
        for metadata_file in metadata_files:
            problem_id = metadata_file.stem
            logging.info(f"Processing problem: {problem_id}")

            submissions = self._extract_sufficient_submissions(metadata_file)
            if submissions:
                suitable_problems.append({
                    'problem_id': problem_id,
                    'metadata_file': metadata_file,
                    'submissions': submissions
                })
                logging.info(
                    f"{problem_id}: Found sufficient submissions "
                    f"(Java: {len(submissions['java_accepted'])}, "
                    f"Python: {len(submissions['python_accepted'])})"
                )

            if len(suitable_problems) >= target_count:
                logging.info(f"Reached target count of {target_count} problems")
                break

        logging.info(f"Total suitable problems found: {len(suitable_problems)}")

        return suitable_problems

    def _extract_sufficient_submissions(self, metadata_file: Path) -> Optional[Dict]:
        """Extract submissions if problem meets criteria."""
        submissions = self._read_metadata_file(metadata_file)
        java_accepted, python_accepted = self._categorize_submissions(submissions)

        if len(java_accepted) >= MIN_SUBMISSIONS_PER_LANGUAGE and len(python_accepted) >= MIN_SUBMISSIONS_PER_LANGUAGE:
            return {
                'java_accepted': random.sample(java_accepted, MIN_SUBMISSIONS_PER_LANGUAGE),
                'python_accepted': random.sample(python_accepted, MIN_SUBMISSIONS_PER_LANGUAGE)
            }

        return None

    def _categorize_submissions(self, submissions: List[Dict]) -> tuple:
        """Categorize submissions by language and acceptance status."""
        java_accepted = []
        python_accepted = []

        for submission in submissions:
            if submission.get('status') == ACCEPTED_STATUS:
                language = submission.get('language', '')
                if language == JAVA_LANGUAGE:
                    java_accepted.append(submission)
                elif language == PYTHON_LANGUAGE:
                    python_accepted.append(submission)

        return java_accepted, python_accepted

    def _assign_non_clones(self, suitable_problems: List[Dict], seed: int):
        """Assign non-clone problems (hard negatives) to each problem."""
        random.seed(seed)
        logging.info("Starting non-clone assignment")

        for i, prob_data in enumerate(suitable_problems):
            problem_id = prob_data['problem_id']
            other_indices = [j for j in range(len(suitable_problems)) if j != i]

            selected_indices = self._select_non_clone_indices(other_indices, problem_id)
            non_clone_data = self._collect_non_clones(suitable_problems, selected_indices)

            prob_data['non_clones'] = non_clone_data
            logging.info(
                f"{problem_id}: Assigned {len(non_clone_data['python'])} Python non-clones "
                f"from problems: {', '.join(non_clone_data['source_problem_ids'])}"
            )

    def _select_non_clone_indices(self, other_indices: List[int], problem_id: str) -> List[int]:
        """Select indices for non-clone problems."""
        if len(other_indices) < NON_CLONE_COUNT:
            logging.warning(
                f"Not enough other problems for {problem_id} "
                f"(available: {len(other_indices)})"
            )
            return other_indices

        return random.sample(other_indices, NON_CLONE_COUNT)

    def _collect_non_clones(self, suitable_problems: List[Dict], selected_indices: List[int]) -> Dict:
        """Collect non-clone submissions from selected problems."""
        non_clone_python = []
        non_clone_problem_ids = []

        for idx in selected_indices:
            other_prob = suitable_problems[idx]
            other_problem_id = other_prob['problem_id']
            non_clone_problem_ids.append(other_problem_id)

            if len(non_clone_python) < NON_CLONE_COUNT:
                python_subs = other_prob['submissions']['python_accepted']
                if python_subs:
                    selected_python = random.choice(python_subs).copy()
                    selected_python['original_problem_id'] = other_problem_id
                    non_clone_python.append(selected_python)

        return {
            'python': non_clone_python,
            'source_problem_ids': non_clone_problem_ids
        }

    def _read_metadata_file(self, metadata_file: Path) -> List[Dict]:
        """Read metadata CSV file and return list of submissions."""
        submissions = []

        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                submissions = list(reader)
        except Exception as e:
            logging.error(f"Error reading {metadata_file}: {e}")

        return submissions

    def _create_curated_metadata(self, suitable_problems: List[Dict]):
        """Create curated metadata CSV files."""
        logging.info("Starting curated metadata file creation")

        for prob_data in suitable_problems:
            problem_id = prob_data['problem_id']
            all_submissions = self._prepare_submissions(prob_data)
            self._assign_filenames(all_submissions, problem_id)
            self._write_metadata_file(problem_id, all_submissions)

        logging.info(f"Successfully created {len(suitable_problems)} curated metadata files")

    def _prepare_submissions(self, prob_data: Dict) -> List[Dict]:
        """Prepare and label all submissions for a problem."""
        submissions = prob_data['submissions']
        non_clones = prob_data['non_clones']
        problem_id = prob_data['problem_id']

        all_submissions = (
            submissions['java_accepted'] +
            submissions['python_accepted'] +
            non_clones['python']
        )

        for sub in submissions['java_accepted'] + submissions['python_accepted']:
            sub['clone_type'] = 'clone'
            sub['original_problem_id'] = sub.get('problem_id', problem_id)

        for sub in non_clones['python']:
            sub['clone_type'] = 'non_clone'
            sub['problem_id'] = problem_id

        return all_submissions

    def _assign_filenames(self, submissions: List[Dict], problem_id: str):
        """Assign unique filenames to submissions."""
        counters = defaultdict(int)

        for sub in submissions:
            language = sub.get('language', '')
            clone_type = sub.get('clone_type', '')
            extension = sub.get('filename_ext', '')

            key = (language, clone_type)
            counters[key] += 1
            count = counters[key]

            sub['filename'] = f"{problem_id}_{clone_type}_{count}.{extension}"

        submissions.sort(key=lambda x: x.get('filename', ''))

    def _write_metadata_file(self, problem_id: str, submissions: List[Dict]):
        """Write submissions to metadata CSV file."""
        output_file = self.output_metadata_dir / f"{problem_id}.csv"

        if not submissions:
            return

        fieldnames = [
            'submission_id',
            'problem_id',
            'language',
            'filename_ext',
            'clone_type',
            'original_problem_id',
            'filename'
        ]

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(submissions)

        logging.info(f"Created {output_file.name}: {len(submissions)} submissions")

    def _create_summary_report(self, suitable_problems: List[Dict]):
        """Create a summary report of the curated dataset."""
        summary_file = self.output_root / "dataset_summary.csv"
        logging.info(f"Creating summary report: {summary_file}")

        with open(summary_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'problem_id',
                'java_clones',
                'python_clones',
                'python_non_clones',
                'total_submissions',
                'non_clone_source_problems'
            ])

            for prob_data in suitable_problems:
                writer.writerow(self._generate_summary_row(prob_data))

        logging.info(f"Summary report created with {len(suitable_problems)} problems")

    def _generate_summary_row(self, prob_data: Dict) -> List:
        """Generate a summary row for a problem."""
        problem_id = prob_data['problem_id']
        submissions = prob_data['submissions']
        non_clones = prob_data['non_clones']

        java_count = len(submissions['java_accepted'])
        python_count = len(submissions['python_accepted'])
        non_clone_count = len(non_clones['python'])
        total = java_count + python_count + non_clone_count

        return [
            problem_id,
            java_count,
            python_count,
            non_clone_count,
            total,
            '; '.join(non_clones['source_problem_ids'])
        ]


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    dataset_root = project_root.joinpath("dataset")
    output_root = project_root.joinpath("preprocessed_dataset")

    with MetadataOrganizer(dataset_root, output_root) as organizer:
        organizer.process_metadata()


if __name__ == "__main__":
    main()
