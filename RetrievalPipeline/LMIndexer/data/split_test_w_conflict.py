import json
import argparse
import os
from collections import Counter

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def extract_prefix(target):
    """Extract the prefix of the target format: <id_0_10><id_1_46>... (excluding last part)"""
    parts = target.strip('<>').split('><')
    prefix = '><'.join(parts[:-1]) if len(parts) > 1 else None
    return prefix

def load_id_counts(ids_path):
    """Load IDs from ids.txt and count prefix occurrences."""
    with open(ids_path, 'r', encoding='utf-8') as f:
        prefixes = [extract_prefix(line.strip()) for line in f if extract_prefix(line.strip())]
    return Counter(prefixes)

def categorize_test_cases(test_path, ids_path, output_path, noconflict_test, noconflict_output):
    test_data = load_data(test_path)
    id_counts = load_id_counts(ids_path)
    noconflict_test_data = None

    if noconflict_test:
        noconflict_test_data = load_data(noconflict_test)
    
    conflict_tests = []
    unique_tests = []

    noconflict_conflict_tests = []
    noconflict_unique_tests = []
    
    for index, entry in enumerate(test_data):
        prefix = extract_prefix(entry["target"])
        if prefix and id_counts[prefix] > 1:
            conflict_tests.append(entry)
            if noconflict_test_data is not None:
                noconflict_conflict_tests.append(noconflict_test_data[index])
        else:
            unique_tests.append(entry)
            if noconflict_test_data is not None:
                noconflict_unique_tests.append(noconflict_test_data[index])
    
    conflict_file = os.path.join(output_path, "conflict_test_cases.json")
    unique_file = os.path.join(output_path, "unique_test_cases.json")
    
    save_data(conflict_file, conflict_tests)
    save_data(unique_file, unique_tests)

    if noconflict_test_data is not None:
        noconflict_conflict_file = os.path.join(noconflict_output, "conflict_test_cases.json")
        noconflict_unique_file = os.path.join(noconflict_output, "unique_test_cases.json")

        save_data(noconflict_conflict_file, noconflict_conflict_tests)
        save_data(noconflict_unique_file, noconflict_unique_tests)
    
    
    print(f"Conflict test cases saved to {conflict_file}")
    print(f"Unique test cases saved to {unique_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Categorize test cases based on ID prefix conflicts.")
    parser.add_argument("--test", type=str, required=True, help="Path to test.json file")
    parser.add_argument("--ids", type=str, required=True, help="Path to ids.txt file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the categorized test cases")
    parser.add_argument("--noconflict_test", type=str, required=False, help="Path to noconflict test.json file")
    parser.add_argument("--noconflict_output", type=str, required=True, help="Directory to save the categorized test cases")
    
    args = parser.parse_args()
    categorize_test_cases(args.test, args.ids, args.output, args.noconflict_test, args.noconflict_output)
