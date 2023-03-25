import argparse
import os
import torch
import config
import json
import evaluate
from tqdm import tqdm
from datapath import data_path
from new_dataloader import newDataset
from model_copy import ModelClass
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description="sentiment analysis")
parser.add_argument(
    "--result_path", type=str, required=True,
    help="result_path"
)
parser.add_argument(
    "--epoch", type=int, default=30, required=True,
    help="epoch"
)

parser.add_argument(
    "--cuda", type=int, default=0,
    help="cuda"
)


args = parser.parse_args()
device = "cuda:" + str(args.cuda)

config = json.load(open(os.path.join(args.result_path, "config.json")))

model_path = os.path.join(args.result_path, "checkpoints", "{}_mymodel.pt".format(args.epoch))
model = ModelClass(config)

model.load_state_dict(torch.load(model_path))
model.to(device)



train_path, val_path = data_path()

eval_dataset = newDataset(val_path, 'val')
eval_dataloader = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=eval_dataset.collate_fn)



bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

model.eval()
with torch.no_grad():
    # 모델 불러올 때 일치하지 않는 키들을 무시 
    # model.load_state_dict(torch.load("/home/nlplab/hdd1/eunseo/AutoVQA/results/0309__real_dl_cls_latent_768/checkpoints/0_mymodel.pt", map_location=device), strict=False)
    # model.to(device)

    eval_bleu = []
    eval_rouge1 = []
    eval_rougeL = []
    eval_meteor = [] # machine translation metric


    for i, batch in enumerate(tqdm(eval_dataloader)):

        mode = "eval"
        img_path = batch[0]
        images = batch[1]
        image_ids = batch[2]
        ids = batch[3]
        captions = batch[4]

        # captions = [f"<|endoftext|>{caption}<|endoftext|>" for caption in captions]
    
        logits = model.generate(device, images)
        print(logits.shape)

        # print(logits.shape) # [64, 3, 50261]
        label = list(captions)
        # label = model.tokenizer.batch_decode([:,:], skip_special_tokens=True)
        pred = model.tokenizer.batch_decode(logits, skip_special_tokens=False)
        print(pred)
        pred = list(map(lambda x:x.split("<|endoftext|>")[1], pred))
        
        print(captions)
        print(pred)

        # print("pred: ", pred)
        
        # if label is not None and pred is not None:
        #     answer_label = list(map(lambda x:x.split("\nAnswer: ")[1], label))
        #     answer_pred = list(map(lambda x:x.split("\nAnswer: ")[1], pred))
        
        # print(answer_label)
        # print(answer_pred)
        
        eval_bleu.append(bleu.compute(predictions=pred, references=label)["bleu"])
        eval_rouge1.append(rouge.compute(predictions=pred, references=label)["rouge1"])
        eval_rougeL.append(rouge.compute(predictions=pred, references=label)["rougeL"])
        eval_meteor.append(meteor.compute(predictions=pred, references=label)['meteor'])
        print(eval_bleu)
        print(eval_rouge1)
        print(eval_rougeL)
        print(eval_meteor)

        for img_path, label, pred in zip(img_path, label, pred):
            result[epoch][i]['img_path'] = img_path
            result[epoch][i]['label'] = label
            result[epoch][i]['pred'] = pred
            
               
    