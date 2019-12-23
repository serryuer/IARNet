import json
import os
from tqdm import tqdm
import pandas as pd

data_dir = '/sdd/yujunshuai/data/weibo/Weibo'
files = os.listdir(data_dir)

# with open('has_image_sample.txt', mode='w') as w1:
#     with open('image_url.txt', mode='w') as w2:
#         for file in tqdm(files):
#             with open(os.path.join(data_dir, file), mode='r') as f:
#                 data = json.loads(f.read())
#                 image_url = data[0]['picture']
#                 if image_url:
#                     w1.write(file + '\n')
#                     w2.write(data[0]['picture'] + "\n")
#                 else:
#                     continue
all_data = []
all_comment = []
for file in tqdm(files):
    with open(os.path.join(data_dir, file), mode='r') as f:
        data = json.loads(f.read())
        # all_data.append(data[0])
        for i in range(1, len(data)):
            all_comment.append(data[i])

# all_data = pd.DataFrame(all_data)
# all_data.to_csv('all_data.csv', index=False, sep='\t')
all_comment = pd.DataFrame(all_comment)
all_comment.to_csv('all_comment.csv', index=False, sep='\t')

