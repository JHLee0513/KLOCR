# launch with OMP_NUM_THREADS=1 accelerate launch --gpu_ids="0,1" train.py for distributed training
# launch with OMP_NUM_THREADS=1 python train.py for single GPU training

import os
import torch
from PIL import Image
import numpy as np
import albumentations as A
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from accelerate.utils.tqdm import tqdm
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
use_amp = True
SAVE_ITER = 100000  # Save checkpoint every SAVE_ITER iterations
out_dir = "tmp/experiment-dir"
DATA_DIR = "data"
DATA_LIST_NAME = out_dir.split("/")[-1]
os.makedirs(out_dir, exist_ok=True)
writer = SummaryWriter(out_dir)
MAX_LENGTH = 192
BATCHSIZE = 80
NW = 8

def get_files_recursive(file_list=[], input_path=""):
    for x in os.listdir(input_path):
        x = os.path.join(input_path, x)
        if os.path.isdir(x):
            file_list = get_files_recursive(file_list=file_list, input_path=x)
        else:
            if os.path.isfile(x.replace('images', 'label').replace('.jpg', '.txt').replace('.png', '.txt')):
                file_list.append(x)
            else:
                print(f"Found issue with {x}, could not find corresponding label!")
    return file_list


# TODO: you can first run this to generate filelist, and reuse it
# for split in ['Training', 'Validation']:
#     if not os.path.isfile(f"{DATA_DIR}/{DATA_LIST_NAME}_filelist_{split}.txt"):
#         print(f"Generating file list for split {split}..")
#         image_files = []
#         for d in [
#                 f"{DATA_DIR}/ko-ocr-logistics-processed",
#                 f"{DATA_DIR}/synthtiger_v1.1-processed",
#                 f"{DATA_DIR}/ko-font-processed",
#                 f"{DATA_DIR}/ko-ocr-public-processed",
#                 f"{DATA_DIR}/ko-ocr-public2-processed",
#                 f"{DATA_DIR}/idl-wds-processed",
#                 f"{DATA_DIR}/ko-font-itw-processed",
#                 f"{DATA_DIR}/synthtiger-kloser-v0.1-10-processed",
#                 f"{DATA_DIR}/synthtiger-kloser-v0.1-20-processed",
#                 f"{DATA_DIR}/synthtiger-kloser-v0.1-30-processed",
#                 f"{DATA_DIR}/ko-handwriting-processed",
#                 f"{DATA_DIR}/ko-finance-processed",
#                 f"{DATA_DIR}/ko-medical-processed",
#                 f"{DATA_DIR}/Uber-Text-processed",
#                 f"{DATA_DIR}/ko-math-processed",
#                 f"{DATA_DIR}/textocr-processed",
#                 f"{DATA_DIR}/cocotext-processed",
#                 f"{DATA_DIR}/Union14M-L-processed",
#                 f"{DATA_DIR}/multilingual-ocr-processed"
#             ]:
        
#             image_files += get_files_recursive(file_list=[], input_path=os.path.join(d, split, "images"))
#         with open(f"{DATA_DIR}/{DATA_LIST_NAME}_filelist_{split}.txt", "w") as f:
#             f.write("\n".join(image_files) + "\n")

#     else:
#         print("existing file list found, skipping file generation..")
# assert False

class IAMDataset(Dataset):
    def __init__(self, root_dir, processor=None, max_target_length=128, verify=False, filelist=None, split='train'):
        self.root_dir = root_dir
        if filelist is not None:
            with open(filelist, "r") as f:
                self.image_files = f.read().split("\n")[:-1]
        else:
            self.image_files = get_files_recursive(file_list=[], input_path=os.path.join(root_dir, "images"))
        self.processor = processor
        self.max_target_length = max_target_length
        self.split = split
        if self.split == 'train':

            self.transform = A.Compose([
                A.Resize(384,384),
                A.Blur(blur_limit=(3,11), p=0.05),
                A.CoarseDropout(
                    num_holes_range=(5,30),
                    hole_height_range=(0.01, 0.3),
                    hole_width_range=(0.01, 0.03),
                    fill_value = 255,
                    p=0.05
                )
            ])
        else:
            self.transform = A.Compose([
                A.Resize(384,384),
            ])
    
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            img_name = os.path.join(self.root_dir, "images", self.image_files[idx])
            lbl_name = img_name.replace("images", "label").replace(".jpg", ".txt").replace(".png", ".txt").replace(".jpeg", ".txt")
            image = Image.open(img_name).convert("RGB")
            
            with open(lbl_name, "r") as f:
                text = f.read()

            image = self.transform(image=np.array(image))['image']
        except Exception as e:
            print(f"ran into issue with {img_name}")
            image = torch.rand(1, 3, 384, 384)
            text = ""
        if self.processor is not None:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            labels = self.processor.tokenizer(text, 
                                            padding="max_length", 
                                            max_length=self.max_target_length,
                                            truncation=True).input_ids
            labels = [label if label != self.processor.tokenizer.pad_token_id else -100 for label in labels]
            encoding = {"pixel_values": pixel_values.squeeze(), "labels": torch.tensor(labels)}
        else:
            encoding = {
                "pixel_values": image.squeeze(),
                "labels": text}

        return encoding

def training_loop(mixed_precision="fp16", seed: int = 42, batch_size: int = 88):
    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    
    model = VisionEncoderDecoderModel.from_pretrained("team-lucid/trocr-small-korean")
    processor = TrOCRProcessor.from_pretrained("team-lucid/trocr-small-korean")
    
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    # make sure vocab size is set correctly
    model.config.vocab_size = model.config.decoder.vocab_size
    
    # set beam search parameters
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.max_length = MAX_LENGTH
    model.config.early_stopping = False
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 1.0
    model.config.num_beams = 1
    
    train_dt = []
    val_dt = []
    
    filelist = f"{DATA_DIR}/{DATA_LIST_NAME}_filelist_Training.txt"
    filelist_val = f"{DATA_DIR}/{DATA_LIST_NAME}_filelist_Validation.txt"
    train_dataset = IAMDataset('', filelist=f'{filelist}', processor=processor, max_target_length=MAX_LENGTH) 
    eval_dataset = IAMDataset('', filelist=f'{filelist_val}', processor=processor, max_target_length=MAX_LENGTH, split='val')
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    c = np.random.choice(np.arange(len(eval_dataset)), 1000000)
    evalset = torch.utils.data.Subset(eval_dataset, c)
    eval_dataloader = DataLoader(evalset, batch_size=batch_size, num_workers=NW)
    trainset = train_dataset
    train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=NW)

    model, optimizer, eval_dataloader = accelerator.prepare(
        model, optimizer, eval_dataloader
    )
    train_dataloader = accelerator.prepare(train_dataloader)

    global_iteration = 0
    running_loss = 0.0
    log_interval = 1000
    for epoch in range(3):
        model.module.decoder.train()
        train_loss = 0.0
        pbar=tqdm(total=len(train_dataloader))
        for i, batch in enumerate(train_dataloader):
            if accelerator.is_main_process:
                pbar.update(1)
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            
            train_loss += loss.item()
            global_iteration += 1
            
            running_loss = 0.9 * running_loss + 0.1 * loss.item()
            if global_iteration % log_interval == 0 and accelerator.is_main_process:
                writer.add_scalar('loss/train_batch_running_avg', running_loss, global_iteration)
            
            if global_iteration % SAVE_ITER == 0 and accelerator.is_main_process:
                iter_dir = os.path.join(out_dir, f"iter_{str(global_iteration).zfill(6)}")
                os.makedirs(iter_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
                unwrapped_model.save_pretrained(
                    iter_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
                )
                accelerator.print(f"Saved checkpoint at iteration {global_iteration}")
        pbar.close()
        
        # garbage collection
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        
        accelerator.print(f"Loss after epoch {epoch}:", float(train_loss)/len(train_dataloader))
        writer.add_scalar('loss/train', float(train_loss)/len(train_dataloader), epoch)
        
        with torch.no_grad():
            model.eval()
            if accelerator.is_main_process:
                epoch_dir = os.path.join(out_dir, str(epoch).zfill(3))        
                os.makedirs(epoch_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model, keep_fp32_wrapper=True)
                unwrapped_model.save_pretrained(
                    epoch_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save,
                )
            
            # validation loop
            valid_loss = 0.0
            valid_batch_count = 0
            pbar=tqdm(total=len(eval_dataloader))
            for batch in eval_dataloader:
                if accelerator.is_main_process:
                    pbar.update(1)
                outputs = model(**batch)
                valid_loss += outputs.loss.item()
                valid_batch_count += 1
            pbar.close()
            
            # cleanup
            torch.cuda.empty_cache()
            gc.collect()
            
        # use batch count instead of labels length
        avg_valid_loss = float(valid_loss) / valid_batch_count
        accelerator.print(f"epoch {epoch} Validation Loss: {avg_valid_loss}")
        writer.add_scalar('loss/valid', avg_valid_loss, epoch)

if __name__ == "__main__":
    training_loop(batch_size=BATCHSIZE)
