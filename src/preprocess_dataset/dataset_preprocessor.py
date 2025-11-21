import csv
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

Path('logs').mkdir(exist_ok=True)
logging.basicConfig(
    filename="logs/prepare_dataset.log",
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

CLONE_TYPE = 'clone'
NON_CLONE_TYPE = 'non_clone'

class DatasetPreprocessor:
    """Preprocesses dataset by copying files according to curated metadata."""

    def __init__(self, dataset_root: Path, preprocessed_dataset_root: Path):
        self.dataset_root = dataset_root
        self.preprocessed_dataset_root = preprocessed_dataset_root
        self.source_data_dir = self.dataset_root.joinpath("data")
        self.preprocessed_metadata_dir = self.preprocessed_dataset_root.joinpath("metadata")
        self.preprocessed_data_dir = self.preprocessed_dataset_root.joinpath("data")

    def __enter__(self):
        self._validate_directories()
        self.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        logging.info(
            f"Initialized DatasetPreprocessor with dataset_root={self.dataset_root}, "
            f"preprocessed_dataset_root={self.preprocessed_dataset_root}"
        )
        return self

    def __exit__(self, exception, _, __):
        if exception is None:
            print("Dataset preparation completed successfully")
            logging.info("Dataset preparation completed successfully")
        else:
            print(f"Dataset preparation failed due to {exception}")
            logging.error(f"Dataset preparation failed due to {exception}")

    def process_all_metadata(self):
        """Process all metadata files and copy corresponding submission files."""
        metadata_files = sorted(self.preprocessed_metadata_dir.glob("p*.csv"))
        logging.info(f"Found {len(metadata_files)} metadata files to process")

        stats = self._process_metadata_files(metadata_files)
        self._log_processing_summary(stats)

    def _validate_directories(self):
        """Validate required directories exist."""
        required_dirs = {
            'dataset_root': self.dataset_root,
            'source_data': self.source_data_dir,
            'preprocessed_metadata': self.preprocessed_metadata_dir
        }

        for name, directory in required_dirs.items():
            if not directory.exists():
                error_msg = f"Directory {directory} does not exist"
                logging.error(error_msg)
                raise FileNotFoundError(error_msg)

    def _process_metadata_files(self, metadata_files: List[Path]) -> Dict:
        """Process all metadata files and return processing statistics."""
        stats = {
            'total_clones': 0,
            'total_non_clones': 0,
            'successful_problems': 0,
            'failed_problems': []
        }

        for idx, metadata_file in enumerate(metadata_files, 1):
            problem_id = metadata_file.stem
            logging.info(f"[{idx}/{len(metadata_files)}] Processing {problem_id}...")

            self._process_single_problem(problem_id, metadata_file, stats)

        return stats

    def _process_single_problem(self, problem_id: str, metadata_file: Path, stats: Dict):
        """Process a single problem and update statistics."""
        try:
            clones, non_clones = self._process_problem(problem_id, metadata_file)
            stats['total_clones'] += clones
            stats['total_non_clones'] += non_clones
            stats['successful_problems'] += 1
            logging.info(f"{problem_id}: Copied {clones} clones and {non_clones} non-clones")
        except Exception as e:
            logging.error(f"{problem_id}: Error - {e}")
            stats['failed_problems'].append((problem_id, str(e)))

    def _process_problem(self, problem_id: str, metadata_file: Path) -> Tuple[int, int]:
        """Process a single problem's metadata and copy files."""
        submissions = self._read_metadata_file(metadata_file)
        clone_subs, non_clone_subs = self._categorize_submissions(submissions)

        logging.info(
            f"{problem_id}: Found {len(clone_subs)} clones and {len(non_clone_subs)} non-clones"
        )

        target_dir = self.preprocessed_data_dir.joinpath(problem_id)
        target_dir.mkdir(parents=True, exist_ok=True)

        clones_copied = self._copy_submissions(problem_id, clone_subs, target_dir)
        non_clones_copied = self._copy_submissions(problem_id, non_clone_subs, target_dir)

        return clones_copied, non_clones_copied

    def _categorize_submissions(self, submissions: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Categorize submissions into clones and non-clones."""
        clone_subs = []
        non_clone_subs = []

        for sub in submissions:
            clone_type = sub.get('clone_type', '')
            if clone_type == CLONE_TYPE:
                clone_subs.append(sub)
            elif clone_type == NON_CLONE_TYPE:
                non_clone_subs.append(sub)

        return clone_subs, non_clone_subs

    def _copy_submissions(self, problem_id: str, submissions: List[Dict], target_dir: Path) -> int:
        """Copy submission files using pre-generated filenames from metadata."""
        copied_count = 0

        for sub in submissions:
            if self._copy_single_submission(problem_id, sub, target_dir):
                copied_count += 1

        return copied_count

    def _copy_single_submission(self, problem_id: str, submission: Dict, target_dir: Path) -> bool:
        """Copy a single submission file. Returns True if successful."""
        if not self._validate_submission_metadata(problem_id, submission):
            return False

        source_file = self._get_source_file_path(problem_id, submission)
        if not source_file.exists():
            logging.warning(f"{problem_id}: Source file not found: {source_file}")
            return False

        return self._copy_file(problem_id, source_file, target_dir, submission['filename'])

    def _validate_submission_metadata(self, problem_id: str, submission: Dict) -> bool:
        """Validate submission has required metadata fields."""
        required_fields = ['submission_id', 'language', 'filename_ext', 'filename']

        if not all(submission.get(field) for field in required_fields):
            logging.warning(f"{problem_id}: Missing metadata for submission")
            return False

        return True

    def _get_source_file_path(self, problem_id: str, submission: Dict) -> Path:
        """Construct source file path from submission metadata."""
        submission_id = submission['submission_id']
        extension = submission['filename_ext']
        language = submission['language']
        original_problem_id = submission.get('original_problem_id', problem_id)

        source_filename = f"{submission_id}.{extension}"
        return self.source_data_dir.joinpath(original_problem_id, language, source_filename)

    def _copy_file(self, problem_id: str, source_file: Path, target_dir: Path, new_filename: str) -> bool:
        """Copy file from source to target directory. Returns True if successful."""
        target_file = target_dir.joinpath(new_filename)

        try:
            shutil.copy2(source_file, target_file)
            logging.debug(f"{problem_id}: Copied {source_file.name} -> {new_filename}")
            return True
        except Exception as e:
            logging.error(f"{problem_id}: Failed to copy {source_file.name}: {e}")
            return False

    def _read_metadata_file(self, metadata_file: Path) -> List[Dict]:
        """Read metadata CSV file and return list of submissions."""
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                submissions = list(reader)
            logging.debug(f"Successfully read {len(submissions)} submissions from {metadata_file.name}")
            return submissions
        except Exception as e:
            logging.error(f"Error reading {metadata_file}: {e}")
            return []

    def _log_processing_summary(self, stats: Dict):
        """Log summary of processing results."""
        logging.info(
            f"Processing complete: {stats['successful_problems']} problems, "
            f"{stats['total_clones']} clones, {stats['total_non_clones']} non-clones"
        )

        if stats['failed_problems']:
            logging.warning(f"Failed problems ({len(stats['failed_problems'])}):")
            for prob_id, error in stats['failed_problems']:
                logging.warning(f"  - {prob_id}: {error}")


def main():
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    dataset_root = project_root.joinpath("dataset")
    preprocessed_dataset_root = project_root.joinpath("preprocessed_dataset")

    with DatasetPreprocessor(dataset_root, preprocessed_dataset_root) as preprocessor:
        preprocessor.process_all_metadata()


if __name__ == "__main__":
    main()
