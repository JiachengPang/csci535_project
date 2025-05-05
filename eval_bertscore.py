import json
import argparse
from bert_score import score

def load_captions(json_file):
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    candidates = [item['generated_caption'].strip() for item in data]
    references = [item['ground_truth'].strip() for item in data]
    return candidates, references

def evaluate_bertscore(candidates, references, lang='en', verbose=True):
    P, R, F1 = score(candidates, references, lang=lang, verbose=verbose)
    print(f"\nBERTScore Results:\n"
          f"  Precision:  {P.mean().item():.4f}\n"
          f"  Recall:     {R.mean().item():.4f}\n"
          f"  F1 Score:   {F1.mean().item():.4f}")
    return P, R, F1

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True, help="Path to the JSON file")
    args = parser.parse_args()

    candidates, references = load_captions(args.path)
    evaluate_bertscore(candidates, references)
    print(f'Path: {args.path}')

if __name__ == "__main__":
    main()
