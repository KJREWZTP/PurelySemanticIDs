import json
import argparse
from collections import Counter

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_data(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

def find_cold_start_targets(train_path, test_path, output_path, threshold, noconflict_test, noconflict_output):
    train_data = load_data(train_path)
    test_data = load_data(test_path)
    noconflict_test_data = None
    if noconflict_test:
        noconflict_test_data = load_data(noconflict_test)

    target_counts = Counter(entry["target"] for entry in train_data)

    # select cold-start test cases (<= threshold)
    cold_start_tests = []
    noconflict_cold_start_tests = []
    for index, entry in enumerate(test_data):
        if target_counts[entry["target"]] <= threshold:
            cold_start_tests.append(entry)
            if noconflict_test_data is not None:
                noconflict_cold_start_tests.append(noconflict_test_data[index])

    output_file = f"{output_path}/cold_start_test_threshold_{threshold}.json"
    
    save_data(output_file, cold_start_tests)
    print(f"Cold-start test cases saved to {output_file}")

    if noconflict_test_data is not None:
        noconflict_output_file = f"{noconflict_output}/cold_start_test_threshold_{threshold}.json"
    
        save_data(noconflict_output_file, noconflict_cold_start_tests)
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter cold-start test cases based on occurrence in train dataset.")

    parser.add_argument("--train", type=str, required=True, help="Path to train.json file")
    parser.add_argument("--test", type=str, required=True, help="Path to test.json file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save the filtered test cases")
    parser.add_argument("--threshold", type=int, required=True, help="Threshold number for cold-start filtering")
    parser.add_argument("--noconflict_test", type=str, required=False, help="Path to noconflict test.json file")
    parser.add_argument("--noconflict_output", type=str, required=False, help="Directory to save the filtered test cases")

    args = parser.parse_args()

    find_cold_start_targets(args.train, args.test, args.output, args.threshold, args.noconflict_test, args.noconflict_output)
