from comet import download_model, load_from_checkpoint
import argparse
import json

def load_data(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        {
            "pair_index": item["pair_index"], 
            "src": "",
            "mt": item["generated_caption"], 
            "ref": item["ground_truth"], 
            "is_match": item["match_result"]["is_match"]}
        for item in data
    ]

def main(json_path):
    print("Loading COMET model...")
    model_path = download_model("wmt20-comet-da")
    model = load_from_checkpoint(model_path)

    print("Loading data...")
    samples = load_data(json_path)

    print("Running evaluation...")
    outputs = model.predict(samples, batch_size=8)
    scores = outputs["scores"]

    avg_score = sum(scores) / len(scores)
    print(f"\nAverage COMET Score: {avg_score:.4f}")

    # Optional: Save individual scores
    for i, sample in enumerate(samples):
        sample["comet_score"] = scores[i]

    save_path = "./results/generation/comet_scored_output.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=2)

    print(f"Saved detailed scores to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Path to the JSON file")
    args = parser.parse_args()

    main(args.path)
