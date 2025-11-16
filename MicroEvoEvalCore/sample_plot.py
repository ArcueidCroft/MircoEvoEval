import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as gridspec
data_path = "your_path"
data = np.load(data_path)['data']
data = np.squeeze(data)
print(data.shape)
total_sample = data.shape[0]
data_name = 'plane_test23.png'
save_path = '/data/'
if os.path.isdir(save_path) == False:
    os.makedirs(save_path)

sample_size = 100
sample_id = 23
frames_id = [0,9,19,29,39,49,59,69,79,89,99]
frames_name = [1,10,20,30,40,50,60,70,80,90,100]
row = 1
col = 12

plot_samples = []


bit = 0
d = 0
min_val = 0
max_val = 0
for i in range(total_sample):
    img_matrix = data[i, -100, :, :]
    _min_val = np.min(img_matrix)
    _max_val = np.max(img_matrix)
    _d = _max_val - _min_val
    if _d > d:
        bit = i
        d = _d
        max_val = _max_val
        min_val = _min_val
        print(bit, "   ", d, "  ", max_val, "   ", min_val)
        

print(bit, "   ", d, "  ", max_val, "   ", min_val)




for i in frames_id:
    plot_samples.append(data[sample_id, i, :, :])
    # plot_samples.append(data[sample_id, i, :, :])
fig, axes = plt.subplots(row, col, figsize=(9.5,1), constrained_layout=True,
                       gridspec_kw={'width_ratios': [1,1,0.1,1,1,1,1,1,1,1,1,1],
                                    # 'wspace': 0.1
                                    })
# fig = plt.figure(figsize=(12,1))
i = 0
for j in range(col):
        if j != 2:  
            img_matrix = plot_samples[i]
            # ax = axes[i,j]
            ax = axes[j]
            ax.axis('off')  
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            print("max = ", np.max(img_matrix), ",   min = ", np.min(img_matrix))
            ax.imshow(img_matrix, 
                        cmap='gray',
                        interpolation='nearest',
                        # vmin=np.min(img_matrix),
                        # vmax=np.max(img_matrix))
                        vmin=0,
                        vmax=1)
            if j == 0:
                ax.set_title("t = 1", fontsize = 10)
            else:
                # ax.set_title(f"{i*10}", fontsize = 10)
                ax.set_title(f"{frames_name[i]}", fontsize = 10)
            i=i+1
        else:
            ax = axes[j]
            ax.axis('off')  
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.plot([3,3], [-1,1], linestyle='--', color='black', linewidth=1.5)       
plt.savefig(save_path+data_name, 
            bbox_inches='tight', 
            pad_inches=0.1)

        
    