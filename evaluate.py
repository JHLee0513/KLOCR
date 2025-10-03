from datasets import load_metric
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor
from train import IAMDataset
import numpy as np
from kloser import Pipeline
import argparse
np.random.seed(0)

BS = 1
NW = 10
SAMPLE_SIZE = 5000
out_dir = None
DATA_DIR = None
DATA_LIST_NAME = out_dir.split("/")[-1]
DATA_LIST_SUFFIX = "_filelist_Validation.txt"

def load_dataest(sample=-1):
    eval_dataset = IAMDataset(
        '',
        filelist=f'{DATA_DIR}/{DATA_LIST_NAME}{DATA_LIST_SUFFIX}',
        processor=None,
        max_target_length=None,
        split='val'
    )
    print(f"Validation samples: {len(eval_dataset)}")
    if sample > 0:
        c = np.random.choice(np.arange(len(eval_dataset)), SAMPLE_SIZE)
        evalset = torch.utils.data.Subset(eval_dataset, c)
        return evalset
    return eval_dataset

def load_model(name):
    return Pipeline(f"configs/{name}.yaml")

def compute_word_accuracy(predictions, references):
    """
    Compute word accuracy between predictions and references.
    Word accuracy is the percentage of words that exactly match.
    """
    if isinstance(predictions, str):
        predictions = [predictions]
    if isinstance(references, str):
        references = [references]
    
    total_accuracy = 0.0
    for pred, ref in zip(predictions, references):
        pred_words = pred.split()
        ref_words = ref.split()
        
        if len(ref_words) == 0:
            # If reference is empty, accuracy is 1 if prediction is also empty
            accuracy = 1.0 if len(pred_words) == 0 else 0.0
        else:
            # Count exact word matches
            matches = sum(1 for p, r in zip(pred_words, ref_words) if p == r)
            # Word accuracy = correct words / total reference words
            accuracy = matches / len(ref_words)
        
        total_accuracy += accuracy
    
    return total_accuracy / len(predictions)

def run_evaluation(model, eval_dataloader):
    cer_metric = load_metric("cer")
    cntr = 0
    valid_cer = 0.0
    valid_word_acc = 0.0
    with torch.no_grad():
        pbar=tqdm(total=len(eval_dataloader))
        for batch in eval_dataloader:
            pbar.update(1)
            im = batch['pixel_values'][0]
            h_,w_,c_ = im.shape
            outputs = model.run(
                {
                    'image': im,
                    'roi': np.array([[[0,0], [0,w_], [h_,w_], [h_,0]]])
                }
            )
            if batch["labels"][0] == "":
                continue
            cer = cer_metric.compute(
                predictions=outputs['text'],
                references=batch["labels"])
            word_acc = compute_word_accuracy(
                predictions=outputs['text'],
                references=batch["labels"])
            valid_cer += cer
            valid_word_acc += word_acc
            cntr += 1

    valid_cer = valid_cer / cntr
    valid_word_acc = valid_word_acc / cntr
    print(f"Character Error Rate (max: 1 min: 0): {valid_cer}")
    print(f"Word Accuracy (max: 1 min: 0): {valid_word_acc}")

def main(args):
    dataset = load_dataest(sample=SAMPLE_SIZE)
    eval_dataloader = DataLoader(dataset, batch_size=BS, num_workers=NW)
    model = load_model(args.model)
    if model is not None:
        run_evaluation(model, eval_dataloader)
            
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', choices=['klocr','trocr'])
    args = parser.parse_args()
    main(args)
