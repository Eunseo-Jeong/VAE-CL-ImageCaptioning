import json
import argparse

parser = argparse.ArgumentParser(description="result analysis")
parser.add_argument(
    "--out_file", type=str, required=True,
    help="json_path"
)
args = parser.parse_args()

result_dict = {}
epoch = 0
with open(args.out_file, "r") as fp:
    a = json.load(fp)

    for epoch, data in a.items():
        epoch = int(epoch)

        loss = data['loss']
        bleu = data['eval_bleu']
        rouge1 = data['eval_rouge1']
        rougeL = data['eval_rougeL']
        meteor = data['eval_meteor']

        try:
            result_dict[epoch]['loss'] = loss
            result_dict[epoch]['bleu'] = bleu
            result_dict[epoch]['rouge1'] = rouge1
            result_dict[epoch]['rougeL'] = rougeL
            result_dict[epoch]['meteor'] = meteor

        except:
            result_dict[epoch] = {}
            result_dict[epoch]['loss'] = loss
            result_dict[epoch]['bleu'] = bleu
            result_dict[epoch]['rouge1'] = rouge1
            result_dict[epoch]['rougeL'] = rougeL
            result_dict[epoch]['meteor'] = meteor

print(result_dict)

print("\n\nbest score")
max_value = {}
best_epoch = {}

for e, i in result_dict.items():
    for p_t, p in i.items():
        try:
            if max_value[p_t] < p:
                max_value[p_t] = p
                best_epoch[p_t] = e
        except:
            max_value[p_t] = p
            best_epoch[p_t] = e
            
print(max_value)
print(best_epoch)

for i in set(best_epoch.values()):
    print("\n\n",i)
    print(result_dict[i])