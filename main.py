import torch
from transformers import (
    RobertaTokenizer,
    Wav2Vec2FeatureExtractor,
)
from tqdm import tqdm
import argparse
from decoder import ProjectionLayer, MultimodalDecoder
from decoder_main import load_encoder
from utils import (
    get_iemocap_caption_data_loaders,
    collate_fn_caption,
    collate_fn_caption_precomputed,
)
import json

text_checkpoint = "roberta-base"
audio_checkpoint = "facebook/hubert-base-ls960"

DEFAULT_PROMPT = "Describe the emotion expressed in this speech by focusing on both the speaker's words and vocal characteristics. Your response:"

def load_projector_decoder(model_choice, from_pretrained=None):
    print(
        f"Loading projector/decoder: model: {model_choice}, from_pretrained: {from_pretrained}"
    )
    if model_choice == "xnorm":
        encoder_dim = 1536
    elif model_choice == "mbt":
        encoder_dim = 768
    else: # early, late
        encoder_dim = 512

    if from_pretrained:
        pretrained = torch.load(from_pretrained, map_location="cpu")
        projector_state_dict = pretrained.get("projector_state_dict")
        decoder_state_dict = pretrained.get("decoder_state_dict")
        projector = ProjectionLayer(
            encoder_dim, 2048, pretrained_weights=projector_state_dict
        )
        decoder = MultimodalDecoder(pretrained_weights=decoder_state_dict)
    else:
        projector = ProjectionLayer(encoder_dim, 2048)
        decoder = MultimodalDecoder()
    return projector, decoder

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Decoder generation on {device}")
    emotion_labels = ["angry", "frustrated", "happy", "sad", "neutral"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="xnorm", choices=["xnorm", "early", "late", "mbt"]
    )
    args = parser.parse_args()
    encoder_choice = args.model

    encoder_ckpt = f"./results/{encoder_choice}_checkpoint.pth"
    decoder_ckpt = f"./results/{encoder_choice}_best_captioning_model.pth"
    encoder = load_encoder(
        encoder_choice, num_classes=len(emotion_labels), from_pretrained=encoder_ckpt
    )
    projector, decoder = load_projector_decoder(
        encoder_choice, from_pretrained=decoder_ckpt
    )

    caption_tokenizer = decoder.tokenizer

    if encoder_choice in ("xnorm", "mbt"):
        text_tokenizer = RobertaTokenizer.from_pretrained(text_checkpoint)
        audio_processor = Wav2Vec2FeatureExtractor.from_pretrained(audio_checkpoint)
        _, _, test_loader = get_iemocap_caption_data_loaders(
            ds_path="./iemocap",
            precomputed=False,
            collate_fn=lambda batch: collate_fn_caption(
                batch,
                text_tokenizer=text_tokenizer,
                audio_processor=audio_processor,
                caption_tokenizer=caption_tokenizer,
            ),
            batch_size=16,
        )
    else: # early, late
        _, _, test_loader = get_iemocap_caption_data_loaders(
            ds_path="./iemocap_precomputed",
            precomputed=True,
            collate_fn=lambda batch: collate_fn_caption_precomputed(
                batch, caption_tokenizer=caption_tokenizer
            ),
            batch_size=16,
        )

    generation_outputs_path = f"./results/{encoder_choice}_test_generations.json"

    encoder.to(device)
    projector.to(device)
    decoder.to(device)

    encoder.eval()
    projector.eval()
    decoder.eval()

    generations = []
    progress_bar = tqdm(test_loader, desc="Generating Captions")

    with torch.no_grad():
        for batch in progress_bar:
            batch_wav_ids = batch["ids"]

            if encoder_choice == "xnorm":
                text_inputs = {
                    "input_ids": batch["text_inputs"]["input_ids"].to(device),
                    "attention_mask": batch["text_inputs"]["attention_mask"].to(device),
                }
                audio_inputs = {
                    "input_values": batch["audio_inputs"]["input_values"].to(device),
                }
                features = encoder(
                    text_inputs, audio_inputs, return_features=True
                )
            elif encoder_choice == "mbt":
                a = batch["audio_inputs"]["input_values"].to(device)
                t = batch["text_inputs"]["input_ids"].to(device)
                m = batch["text_inputs"]["attention_mask"].to(device)
                features = encoder(a, t, return_features=True, text_mask=m)
            else: # early, late
                text_embs = batch["text_embs"].to(device)
                audio_embs = batch["audio_embs"].to(device)
                features = encoder(
                    audio_emb=audio_embs, text_emb=text_embs, return_features=True
                )

            prefix_emb = projector(features)

            prompts = [DEFAULT_PROMPT] * prefix_emb.size(0)
            prompt_inputs = caption_tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).input_ids.to(device)

            generated_ids = decoder.generate(
                prefix_emb=prefix_emb,
                input_ids=prompt_inputs,
                max_new_tokens=50,
            )

            generated_texts = decoder.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )

            ground_truths = batch["labels"].clone()
            ground_truths[ground_truths == -100] = caption_tokenizer.pad_token_id
            ground_truth_texts = decoder.tokenizer.batch_decode(
                ground_truths, skip_special_tokens=True
            )

            for i in range(len(generated_texts)):
                gen_text = generated_texts[i]
                gt_text = ground_truth_texts[i]
                wav_id = batch_wav_ids[i]
                generations.append(
                    {
                        "wav_filename": wav_id,
                        "generated_caption": gen_text.strip(),
                        "ground_truth": gt_text.strip(),
                    }
                )
    with open(generation_outputs_path, "w") as f:
        json.dump(generations, f, indent=2)

    print(
        f"{encoder_choice} decoder model generated captions saved to '{generation_outputs_path}'"
    )

if __name__ == "__main__":
    main()