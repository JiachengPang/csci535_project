import json
import argparse

def compute_accuracy(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    correct = sum(1 for item in data if item.get("match_result", {}).get("is_match") is True)

    accuracy = correct / total if total > 0 else 0.0
    print(f"Total Samples: {total}")
    print(f"Correct Matches: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the JSON file")
    args = parser.parse_args()

    compute_accuracy(args.path)
    print(f'Path: {args.path}')
