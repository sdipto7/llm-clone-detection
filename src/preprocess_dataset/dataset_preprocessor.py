import os
import csv
import shutil
import logging
from pathlib import Path
from collections import defaultdict

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename="logs/prepare_dataset.log",
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class DatasetPreprocessor:
    def __init__(self, dataset_root, preprocessed_dataset_root):
        self.dataset_root = Path(dataset_root)
        self.preprocessed_dataset_root = Path(preprocessed_dataset_root)
        self.source_data_dir = self.dataset_root.joinpath("data")
        self.preprocessed_metadata_dir = self.preprocessed_dataset_root.joinpath("metadata")
        self.preprocessed_data_dir = self.preprocessed_dataset_root.joinpath("data")
        
    def __enter__(self):
        if not self.dataset_root.exists():
            logging.error(f"Directory {self.dataset_root} does not exist")
            raise FileNotFoundError(f"Directory {self.dataset_root} does not exist")
        
        if not self.source_data_dir.exists():
            logging.error(f"Source data directory {self.source_data_dir} does not exist")
            raise FileNotFoundError(f"Source data directory {self.source_data_dir} does not exist")
        
        if not self.preprocessed_metadata_dir.exists():
            logging.error(f"Curated metadata directory {self.preprocessed_metadata_dir} does not exist")
            raise FileNotFoundError(f"Curated metadata directory {self.preprocessed_metadata_dir} does not exist")
        
        self.preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Initialized DatasetPreprocessor with dataset_root={self.dataset_root}, preprocessed_dataset_root={self.preprocessed_dataset_root}")
        
        return self
        
    def process_all_metadata(self):
        """Process all metadata files and copy corresponding submission files."""
        metadata_files = sorted([
            f for f in self.preprocessed_metadata_dir.glob("p*.csv")
        ])
        
        logging.info(f"Found {len(metadata_files)} metadata files to process")
        
        total_clones = 0
        total_non_clones = 0
        successful_problems = 0
        failed_problems = []
        
        for idx, metadata_file in enumerate(metadata_files, 1):
            problem_id = metadata_file.stem
            
            logging.info(f"[{idx}/{len(metadata_files)}] Processing {problem_id}...")
            
            try:
                clones, non_clones = self._process_problem(problem_id, metadata_file)
                total_clones += clones
                total_non_clones += non_clones
                successful_problems += 1
                logging.info(f"{problem_id}: Copied {clones} clones and {non_clones} non-clones")
            except Exception as e:
                logging.error(f"{problem_id}: Error - {e}")
                failed_problems.append((problem_id, str(e)))
        
        if failed_problems:
            logging.warning(f"Failed problems ({len(failed_problems)}):")
            for prob_id, error in failed_problems:
                logging.warning(f"  - {prob_id}: {error}")
        
    def _process_problem(self, problem_id, metadata_file):
        """Process a single problem's metadata and copy files."""
        submissions = self._read_metadata_file(metadata_file)
        
        clone_subs = []
        non_clone_subs = []
        
        for sub in submissions:
            clone_type = sub.get('clone_type', '')
            if clone_type == 'clone':
                clone_subs.append(sub)
            elif clone_type == 'non_clone':
                non_clone_subs.append(sub)
        
        logging.info(f"{problem_id}: Found {len(clone_subs)} clones and {len(non_clone_subs)} non-clones")
        
        target_dir = self.preprocessed_data_dir.joinpath(problem_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        
        clones_copied = self._copy_submissions(
            problem_id, 
            clone_subs, 
            target_dir, 
            "clone"
        )
        
        non_clones_copied = self._copy_submissions(
            problem_id, 
            non_clone_subs, 
            target_dir, 
            "non_clone"
        )
        
        return clones_copied, non_clones_copied
    
    def _copy_submissions(self, problem_id, submissions, target_dir, label):
        """Copy submission files with new naming convention."""
        lang_counters = defaultdict(int)
        copied_count = 0
        
        for sub in submissions:
            submission_id = sub.get('submission_id', '')
            language = sub.get('language', '')
            extension = sub.get('filename_ext', '')
            original_problem_id = sub.get('original_problem_id', problem_id)
            
            if not all([submission_id, language, extension]):
                logging.warning(f"{problem_id}: Missing metadata for submission")
                continue
            
            filename = f"{submission_id}.{extension}"
            
            source_file = self.source_data_dir.joinpath(original_problem_id, language, filename)
            
            if not source_file.exists():
                logging.warning(f"{problem_id}: Source file not found: {source_file}")
                continue
            
            file_ext = source_file.suffix
            
            lang_counters[language] += 1
            count = lang_counters[language]

            new_filename = f"{problem_id}_{label}_{count}{file_ext}"
            
            target_file = target_dir.joinpath(new_filename)
            
            try:
                shutil.copy2(source_file, target_file)
                copied_count += 1
                logging.debug(f"{problem_id}: Copied {source_file.name} -> {new_filename}")
            except Exception as e:
                logging.error(f"{problem_id}: Failed to copy {filename}: {e}")
        
        return copied_count
    
    def _read_metadata_file(self, metadata_file):
        """Read metadata CSV file and return list of submissions."""
        submissions = []
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    submissions.append(row)
            logging.debug(f"Successfully read {len(submissions)} submissions from {metadata_file.name}")
        except Exception as e:
            logging.error(f"Error reading {metadata_file}: {e}")
        
        return submissions
    
    def __exit__(self, exception, _, __):
        if exception is None:
            print("Dataset preparation completed successfully")
            logging.info("Dataset preparation completed successfully")
        else:
            print(f"Dataset preparation failed due to {exception}")
            logging.error(f"Dataset preparation failed due to {exception}")

def main():
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    
    DATASET_ROOT = PROJECT_ROOT.joinpath("dataset")
    PREPROCESSED_DATASET_ROOT = PROJECT_ROOT.joinpath("preprocessed_dataset")
    
    with DatasetPreprocessor(DATASET_ROOT, PREPROCESSED_DATASET_ROOT) as preprocessor:
        preprocessor.process_all_metadata()

if __name__ == "__main__":
    main()
