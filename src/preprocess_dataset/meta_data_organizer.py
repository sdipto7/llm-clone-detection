import csv
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    filename="logs/meta_data_organizer.log",
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

class MetadataOrganizer:
    def __init__(self, dataset_root, preprocessed_dataset_root):
        self.dataset_root = Path(dataset_root)
        self.output_root = Path(preprocessed_dataset_root)
        self.metadata_dir = self.dataset_root.joinpath("metadata")
        self.output_metadata_dir = self.output_root.joinpath("metadata")
        
    def __enter__(self):
        if not self.dataset_root.exists():
            logging.error(f"Directory {self.dataset_root} does not exist")
            raise FileNotFoundError(f"Directory {self.dataset_root} does not exist")
        
        if not self.metadata_dir.exists():
            logging.error(f"Metadata directory {self.metadata_dir} does not exist")
            raise FileNotFoundError(f"Metadata directory {self.metadata_dir} does not exist")
        
        self.output_metadata_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Initialized MetadataOrganizer with dataset_root={self.dataset_root}, output_root={self.output_root}")
        
        return self
        
    def process_metadata(self, target_count=100, seed=42):
        """
        Process metadata files and create curated dataset.
        
        Args:
            target_count: Number of problems to select (100)
            seed: Random seed for reproducibility
        """
        random.seed(seed)
        logging.info(f"Starting metadata processing with target_count={target_count}, seed={seed}")

        metadata_files = sorted([
            f for f in self.metadata_dir.glob("p*.csv")
        ])
        
        logging.info(f"Found {len(metadata_files)} problem metadata files")
        
        suitable_problems = []
        
        for metadata_file in metadata_files:
            problem_id = metadata_file.stem
            logging.info(f"Processing problem: {problem_id}")

            result = self._check_and_extract_submissions(metadata_file)
            
            if result:
                suitable_problems.append({
                    'problem_id': problem_id,
                    'metadata_file': metadata_file,
                    'submissions': result
                })
                
                logging.info(f"{problem_id}: Found sufficient submissions (Java: {len(result['java_accepted'])}, Python: {len(result['python_accepted'])})")

            if len(suitable_problems) >= target_count:
                logging.info(f"Reached target count of {target_count} problems")
                break
        
        logging.info(f"Total suitable problems found: {len(suitable_problems)}")
        
        self._assign_non_clones(suitable_problems, seed)
        self._create_curated_metadata(suitable_problems)
        self._create_summary_report(suitable_problems)
        
        return suitable_problems
    
    def _check_and_extract_submissions(self, metadata_file):
        """
        Check if problem has sufficient submissions and extract them.
        
        Returns:
            Dictionary with selected submissions or None if criteria not met
        """
        submissions = self._read_metadata_file(metadata_file)

        java_accepted = []
        python_accepted = []
        
        for sub in submissions:
            lang = sub.get('language', '')
            status = sub.get('status', '')

            if status == 'Accepted':
                if lang == 'Java':
                    java_accepted.append(sub)
                elif lang == 'Python':
                    python_accepted.append(sub)

        if len(java_accepted) >= 5 and len(python_accepted) >= 5:
            selected = {
                'java_accepted': random.sample(java_accepted, 5),
                'python_accepted': random.sample(python_accepted, 5)
            }

            return selected

        return None
    
    def _assign_non_clones(self, suitable_problems, seed):
        """
        Assign non-clone problems (hard negatives) to each problem.
        Each problem gets 5 Python non-clones from other problems.
        """
        random.seed(seed)
        logging.info("Starting non-clone assignment")
        
        num_problems = len(suitable_problems)
        
        for i, prob_data in enumerate(suitable_problems):
            problem_id = prob_data['problem_id']
            
            other_indices = [j for j in range(num_problems) if j != i]
            
            if len(other_indices) < 5:
                logging.warning(f"Not enough other problems for {problem_id} (available: {len(other_indices)})")
                selected_indices = other_indices
            else:
                selected_indices = random.sample(other_indices, 5)
            
            non_clone_python = []
            non_clone_problem_ids = []
            
            for idx in selected_indices:
                other_prob = suitable_problems[idx]
                other_problem_id = other_prob['problem_id']
                non_clone_problem_ids.append(other_problem_id)
                
                if len(non_clone_python) < 5:
                    python_subs = other_prob['submissions']['python_accepted']
                    if python_subs:
                        selected_python = random.choice(python_subs).copy()
                        selected_python['original_problem_id'] = other_problem_id
                        non_clone_python.append(selected_python)
            
            prob_data['non_clones'] = {
                'python': non_clone_python,
                'source_problem_ids': non_clone_problem_ids
            }
            
            logging.info(f"{problem_id}: Assigned {len(non_clone_python)} Python non-clones from problems: {', '.join(non_clone_problem_ids)}")
    
    def _read_metadata_file(self, metadata_file):
        """Read metadata CSV file and return list of submissions."""
        submissions = []
        
        try:
            with open(metadata_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    submissions.append(row)
        except Exception as e:
            logging.error(f"Error reading {metadata_file}: {e}")
        
        return submissions
    
    def _create_curated_metadata(self, suitable_problems):
        """Create curated metadata CSV files."""
        logging.info("Starting curated metadata file creation")
        
        for prob_data in suitable_problems:
            problem_id = prob_data['problem_id']
            submissions = prob_data['submissions']
            non_clones = prob_data['non_clones']
            
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
            
            counters = defaultdict(int)
            
            for sub in all_submissions:
                language = sub.get('language', '')
                clone_type = sub.get('clone_type', '')
                extension = sub.get('filename_ext', '')
                
                key = (language, clone_type)
                counters[key] += 1
                count = counters[key]
                
                new_filename = f"{problem_id}_{clone_type}_{count}.{extension}"
                sub['filename'] = new_filename
            
            all_submissions.sort(key=lambda x: x.get('filename', ''))
            
            output_file = self.output_metadata_dir.joinpath(f"{problem_id}.csv")
            
            if all_submissions:
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
                    writer = csv.DictWriter(
                        f, 
                        fieldnames=fieldnames,
                        extrasaction='ignore'
                    )
                    writer.writeheader()
                    writer.writerows(all_submissions)
                
                logging.info(f"Created {output_file.name}: {len(all_submissions)} submissions")
        
        logging.info(f"Successfully created {len(suitable_problems)} curated metadata files")
    
    def _create_summary_report(self, suitable_problems):
        """Create a summary report of the curated dataset."""
        summary_file = self.output_root.joinpath("dataset_summary.csv")
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
                problem_id = prob_data['problem_id']
                submissions = prob_data['submissions']
                non_clones = prob_data['non_clones']
                
                writer.writerow([
                    problem_id,
                    len(submissions['java_accepted']),
                    len(submissions['python_accepted']),
                    len(non_clones['python']),
                    len(submissions['java_accepted']) + len(submissions['python_accepted']) + len(non_clones['python']),
                    '; '.join(non_clones['source_problem_ids'])
                ])
    
        logging.info(f"Summary report created with {len(suitable_problems)} problems")
    
    def __exit__(self, exception, _, __):
        if exception is None:
            print("Metadata organization completed successfully")
            logging.info("Metadata organization completed successfully")
        else:
            print(f"Metadata organization failed due to {exception}")
            logging.error(f"Metadata organization failed due to {exception}")

def main():
    SCRIPT_DIR = Path(__file__).parent
    PROJECT_ROOT = SCRIPT_DIR.parent.parent
    
    DATASET_ROOT = PROJECT_ROOT.joinpath("dataset")
    PREPROCESSED_DATASET_ROOT = PROJECT_ROOT.joinpath("preprocessed_dataset")
    
    with MetadataOrganizer(DATASET_ROOT, PREPROCESSED_DATASET_ROOT) as organizer:
        organizer.process_metadata()

if __name__ == "__main__":
    main()
