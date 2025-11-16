import numpy as np
import matplotlib.pyplot as plt
import os
data_file = np.load('./preds.npy')
true_file = np.load('./trues.npy')
is_error = True
sample_num = data_file.shape[0]
sample_num = 1
frame_num = data_file.shape[1]
print(sample_num, frame_num)
save_path = './plane_data/'
if os.path.isdir(save_path) == False:
    os.mkdir(save_path)
for i in range(sample_num):
    print(i)
    max_error = 0
    min_error = 1
    if is_error == True:
        for j in range(frame_num):
            img_matrix = data_file[i, j, :, :]
            img_true = true_file[i, j, :, :]
            img_matrix = img_matrix-img_true
            img_matrix = np.squeeze(img_matrix)
            _max = np.max(img_matrix)
            _min = np.min(img_matrix)
            if _max > max_error:
                max_error = _max
            if _min < min_error:
                min_error = _min
    path = os.path.join(save_path, f"sample{i}/")
    if os.path.isdir(os.path.join(save_path,f"sample{i}/")) == False:
        os.mkdir(os.path.join(save_path, f"sample{i}/"))
    for j in range(frame_num):
        img_matrix = data_file[i, j, :, :]
        img_true = true_file[i, j, :, :]
        if is_error == True:

            img_matrix = img_matrix-img_true
            img_matrix = np.squeeze(img_matrix)
            print(img_matrix.shape)
            print(f"max = {max_error}, min = {min_error}")
            fig = plt.figure(figsize=(6, 6), facecolor='white')  
            ax = fig.add_subplot(111)

            ax.axis('off')  
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])

            im = plt.imshow(img_matrix, 
                        cmap='gray',
                        interpolation='nearest',
                        vmin=min_error,
                        vmax=max_error)# 
            cbar = plt.colorbar(im, ax=ax, orientation='vertical',shrink=0.8)  # 
            plt.savefig(
                path+f'error_image_{j}.png',
                bbox_inches='tight',  
                pad_inches=0,         
                transparent=False     
            )
            plt.close(fig)
        else:
            img_matrix = np.squeeze(img_matrix)
            fig = plt.figure(figsize=(4, 4), dpi=300)  
            ax = fig.add_subplot(111)
        
            plt.imshow(img_matrix, 
                        cmap='gray',
                        interpolation='nearest',
                        vmin=np.min(img_matrix),
                        vmax=np.max(img_matrix))
            
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            plt.savefig(
                path+f'image_{j}.png',
                bbox_inches='tight',  
                pad_inches=0,         
                transparent=False     
            )
            plt.close(fig) 
    