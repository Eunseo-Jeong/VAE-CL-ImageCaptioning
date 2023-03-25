import os
import json
import torch
# from model_copy import generate
import evaluate
# from new_dataloader import collate_fn
from torch import nn
from torch.utils.data import Dataset, DataLoader
import os
# from encoderDecoder import EncoderDecoder

from tqdm import tqdm
from config import config_data
from datapath import data_path
from model_copy import ModelClass
from new_dataloader import newDataset

import time
import random
import numpy as np
import torch.backends.cudnn as cudnn



def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    # cudnn의 random seed를 고정해줌 (학습 속도가 느려짐)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

    
def kld_weight(config, start=0.0, stop=1, n_cycle=1, ratio=1, linear_ratio=1):
    # if self.fixed_reg_weight is not None:
    #     return self.fixed_reg_weight
    # cycle_size = self.iterations_per_training_epoch // n_cycle
    cycle_size = config['epoch'] * 100 # iterations_per_training_epoch
    vae_steps = int(cycle_size * ratio) # epoch * 100 
    ae_steps = cycle_size - vae_steps  # epoch * 100 - epoch * 100 = 0
    linear_steps = int(vae_steps * linear_ratio)  # 25% # epoch * 100 
    full_steps = cycle_size - ae_steps - linear_steps  # 25% # epoch * 100 - epoch * 100 = 0
    step = config['epoch'] % cycle_size # global step # epoch % epoch * 100 = epoch
    if step <= ae_steps: 
        return 0
    vae_step = step - ae_steps # epoch - 0
    weight = (
        vae_step / linear_steps * (stop - start) # epoch / epoch * 100 *(1-0)
        if vae_step <= linear_steps # epoch * 100 <=  epoch * 100
        else stop
    )
    return weight
 
def write_json(mode, i, epoch, path, time_str, reg_weight, decoder_loss, contrastive_loss, image_reg_loss, loss):
    
    log = {
        "state" : mode,
        "i": i, 
        "epoch": epoch, 
        "reg_weight": reg_weight,
        "decoder_loss": decoder_loss.item(),
        "reg_weight * image_reg_loss": reg_weight * image_reg_loss.item(),
        "loss": loss.item()
    }
    if contrastive_loss is not None:
        log["contrastive_loss"]: contrastive_loss.item()
        

    txt_file = os.path.join(path, time_str)
    txt_file = txt_file + str(".json")
    
    with open(txt_file, "a") as file:
        json.dump(log, file, indent=4)


def time_path():
    now = time.localtime()
    month = str(now.tm_mon); day = str(now.tm_mday)
    
    if len(month) < 2 and len(day) < 2:
        day_str = "0" + month + "0" + day
    elif len(month) < 2 and len(day) >= 2:
        day_str = "0" + month + day
    elif len(month) >= 2 and day < 2:
        day_str = month + "0" + day
    else:
        day_str = month + day
    
    time_str = str(now.tm_hour) + ":" + str(now.tm_min) + ":" + str(now.tm_sec)
        
    return day_str, time_str
        
        
    
def main():
    
    config = config_data()
    config = vars(config) # class -> dictionary로 바뀜 
    
    print(config)
    
    day_str, time_str = time_path()
    
    path = os.path.join("results", str(day_str)+"_"+str(config["result_path"]))
    performance_save = os.path.join(path,"performance.json")
    result_save = os.path.join(path, "result.json")
    
    # if os.path.exists(path):
    #     print("The dicrectory is already exist")
    #     quit()
        
    os.makedirs(path, exist_ok=True)
        
    json.dump(config, open(os.path.join(path,"config.json"), "w"), indent=2)

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"    

    cuda = 'cuda:' + str(config['cuda'])
    device = torch.device(cuda if torch.cuda.is_available() else 'cpu') 
    
    seed_everything(42)

    ########################################################
    contrastive = config['contrastive']
    latent_vector = config['cls_latent_vector']
    dd = config['dd']; dc = config['dc']; di = config['di']
    ########################################################
    
    train_path, val_path = data_path()

    train_dataset = newDataset(train_path, 'train')
    eval_dataset = newDataset(val_path, 'val')
    # test_dataset = 

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=train_dataset.collate_fn)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config['batch_size'], shuffle=False, collate_fn=eval_dataset.collate_fn)

    model = ModelClass(config)
    model.to(device)

    # criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6) # Cosine annealing
    
    bleu = evaluate.load('bleu')
    rouge = evaluate.load('rouge')
    meteor = evaluate.load('meteor')
    
    performance = {}
    result = {}
    
    for epoch in range(config['epoch']):
        os.makedirs(os.path.join(path,"checkpoints"), exist_ok=True)
    
        performance[epoch] = {}
        result[epoch] = {}
    
        train_total_loss = []
        reg_weight_list = []
        log = {}
        
        
        model.train() 
        for i, batch in enumerate(tqdm(train_dataloader)):
            mode = "train"
            img_path = batch[0]
            images = batch[1]
            image_ids = batch[2]
            ids = batch[3]
            captions = batch[4]

            captions = [f"<|endoftext|>{caption}<|endoftext|>" for caption in captions]
            
            optimizer.zero_grad()
            
            reg_weight = kld_weight(config)
            reg_weight_list.append(reg_weight)
            
            
            if contrastive == True:
                decoder_loss, contrastive_loss, image_reg_loss = model(mode, device, images, captions)
                loss = dd * decoder_loss + dc * contrastive_loss + di * image_reg_loss
                log.update({"decoder_loss": decoder_loss, "contrastive_loss": contrastive_loss, "(reg_weight * image_reg_loss)": (reg_weight * image_reg_loss)})
            
            else:
                contrastive_loss = None
                decoder_loss, image_reg_loss = model(mode, device, images, captions)
                loss = dd* decoder_loss + di * image_reg_loss
                log.update({"decoder_loss": decoder_loss, "(reg_weight * image_reg_loss)": (reg_weight * image_reg_loss)})
            
            write_json(mode, i, epoch, path, time_str, reg_weight, decoder_loss, contrastive_loss, image_reg_loss, loss)
            
            # loss = criterion(output, label.squeeze(dim=-1)) # 2522, label
            
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_total_loss.append(loss.item())

        torch.save(model.state_dict(), os.path.join(path, "checkpoints", str(epoch)+"_mymodel.pt"))
        print("epoch", epoch, "log", log)
        avg_loss = sum(train_total_loss) / len(train_dataloader)
        print("avg_loss", avg_loss)
        
        
        performance[epoch]['loss'] = avg_loss
        
        
        
        model.eval()
        with torch.no_grad():
            # 모델 불러올 때 일치하지 않는 키들을 무시 
            # model.load_state_dict(torch.load("/home/nlplab/hdd1/eunseo/AutoVQA/results/0309__real_dl_cls_latent_768/checkpoints/0_mymodel.pt", map_location=device), strict=False)
            # model.to(device)

            eval_bleu = []
            eval_rouge1 = []
            eval_rougeL = []
            eval_meteor = [] # machine translation metric

            # eval_log = {}

            for i, batch in enumerate(tqdm(eval_dataloader)):
                result[epoch][i] = {}
                
                mode = "eval"
                img_path = batch[0]
                images = batch[1]
                image_ids = batch[2]
                ids = batch[3]
                captions = batch[4]

                captions = [f"<|endoftext|>{caption}<|endoftext|>" for caption in captions]
            
                logits, caps_len, caps_input_ids, tokenizer = model(mode, device, images, captions)
                # print(logits.shape) # [64, 3, 50261]

                label = tokenizer.batch_decode(caps_input_ids[:,:], skip_special_tokens=True)
                pred = tokenizer.batch_decode(torch.argmax(logits[:,:,:], dim=-1), skip_special_tokens=False)
                pred = list(map(lambda x:x.split("<|endoftext|>")[0], pred))
                
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
                
                for img_path, label, pred in zip(img_path, label, pred):
                    result[epoch][i]['img_path'] = img_path
                    result[epoch][i]['label'] = label
                    result[epoch][i]['pred'] = pred
                    
               
    
        print("eval BLEU in epoch {}: {}".format(str(epoch), str(sum(eval_bleu)/len(eval_bleu))))
        print("eval Rouge1 in epoch {}: {}".format(str(epoch), str(sum(eval_rouge1)/len(eval_rouge1))))
        print("eval RougeL in epoch {}: {}".format(str(epoch), str(sum(eval_rougeL)/len(eval_rougeL))))
        print("eval METEOR in epoch {}: {}".format(str(epoch), str(sum(eval_meteor)/len(eval_meteor))))
        
        performance[epoch]['eval_bleu'] = str(sum(eval_bleu)/len(eval_bleu))
        performance[epoch]['eval_rouge1'] = str(sum(eval_rouge1)/len(eval_rouge1))
        performance[epoch]['eval_rougeL'] = str(sum(eval_rougeL)/len(eval_rougeL))
        performance[epoch]['eval_meteor'] = str(sum(eval_meteor)/len(eval_meteor))
        
        
        with open(performance_save, "w") as file:
            json.dump(performance, file, indent=4)
        
        with open(result_save, "w") as file:
            json.dump(result, file, indent=4)
        

if __name__ == "__main__":
    main()
