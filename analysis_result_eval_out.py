import json
import argparse

parser = argparse.ArgumentParser(description="sentiment analysis")
parser.add_argument(
    "--out_file", type=str, required=True,
    help="json_path"
)
args = parser.parse_args()

result_dict = {}
epoch = 0
with open(args.out_file, "r") as fp:
    a = fp.readlines()
    for l in a:
        if "eval" in l and "in epoch" in l:
            # print(l)
            tmp = l.split(":")
            epoch = int(tmp[0].split(" ")[-1])
            performance_type = l.split()[1]
            performance = float(tmp[-1])
            
            try:
                result_dict[epoch][performance_type] = performance
            except:
                result_dict[epoch] = {}
                result_dict[epoch][performance_type] = performance

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