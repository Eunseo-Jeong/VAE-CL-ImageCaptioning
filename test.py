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

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = parser.parse_args()
device = "cuda:" + str(args.cuda)
print(device)

config = json.load(open(os.path.join(args.result_path, "config.json")))

model_path = os.path.join(args.result_path, "checkpoints", "{}_mymodel.pt".format(args.epoch))
model = ModelClass(config)

# model.load_state_dict(torch.load(model_path, map_location=device))
# model.to(device)



train_path, val_path = data_path()

eval_dataset = newDataset(val_path, 'val')
eval_dataloader = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=eval_dataset.collate_fn)

test_performance_save = os.path.join(os.getcwd(), 'test_performance.json')
test_result_save = os.path.join(os.getcwd(), 'test_result.json')


bleu = evaluate.load('bleu')
rouge = evaluate.load('rouge')
meteor = evaluate.load('meteor')

performance = {}
result = {}

model.eval()
with torch.no_grad():
    # 모델 불러올 때 일치하지 않는 키들을 무시 
    # model.load_state_dict(torch.load("/home/nlplab/hdd1/eunseo/AutoVQA/results/0309__real_dl_cls_latent_768/checkpoints/0_mymodel.pt", map_location=device), strict=False)
    # model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)    

    eval_bleu = []
    eval_rouge1 = []
    eval_rougeL = []
    eval_meteor = [] # machine translation metric


    for i, batch in enumerate(tqdm(eval_dataloader)):
        result[i] = {}

        mode = "eval"
        img_path = batch[0]
        images = batch[1]
        image_ids = batch[2]
        ids = batch[3]
        captions = batch[4]

        # captions = [f"<|endoftext|>{caption}<|endoftext|>" for caption in captions]
    
        logits = model.generate(device, images)
        # print(logits.shape) # [64, 3, 50261]
        label = list(captions)
        # label = model.tokenizer.batch_decode([:,:], skip_special_tokens=True)
        pred = model.tokenizer.batch_decode(logits, skip_special_tokens=False)
        pred = list(map(lambda x:x.split("<|endoftext|>")[1], pred))
        
        # print(captions)
        # print(pred)

        eval_bleu.append(bleu.compute(predictions=pred, references=label)["bleu"])
        eval_rouge1.append(rouge.compute(predictions=pred, references=label)["rouge1"])
        eval_rougeL.append(rouge.compute(predictions=pred, references=label)["rougeL"])
        eval_meteor.append(meteor.compute(predictions=pred, references=label)['meteor'])
        
        for img_path, label, pred in zip(img_path, label, pred):
            result[i]['img_path'] = img_path
            result[i]['label'] = label
            result[i]['pred'] = pred

    # print("eval BLEU in epoch {}: {}".format(str(epoch), str(sum(eval_bleu)/len(eval_bleu))))
    # print("eval Rouge1 in epoch {}: {}".format(str(epoch), str(sum(eval_rouge1)/len(eval_rouge1))))
    # print("eval RougeL in epoch {}: {}".format(str(epoch), str(sum(eval_rougeL)/len(eval_rougeL))))
    # print("eval METEOR in epoch {}: {}".format(str(epoch), str(sum(eval_meteor)/len(eval_meteor))))
    
    performance['eval_bleu'] = str(sum(eval_bleu)/len(eval_bleu))
    performance['eval_rouge1'] = str(sum(eval_rouge1)/len(eval_rouge1))
    performance['eval_rougeL'] = str(sum(eval_rougeL)/len(eval_rougeL))
    performance['eval_meteor'] = str(sum(eval_meteor)/len(eval_meteor))
    
    
    with open(test_performance_save, "w") as file:
        json.dump(performance, file, indent=4)
    
    with open(test_result_save, "w") as file:
        json.dump(result, file, indent=4)
        
    