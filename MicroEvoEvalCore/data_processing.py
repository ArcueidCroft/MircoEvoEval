import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
import pathlib as Path

def draw_image(img_matrix, save_path, filename, dpi=300, figesize=(6,6)):
    plt.figure(figsize=figesize)
    plt.imshow(img_matrix, 
            cmap='gray',
            interpolation='nearest',
            vmin=np.min(img_matrix),
            vmax=np.max(img_matrix))
    save_path = os.path.join(save_path, filename)
    plt.savefig(save_path, 
            bbox_inches='tight',
            pad_inches=0.1,
            dpi=dpi)
    
def matrix_processing(matrix, tol=125):
    
    for i in range(len(matrix[:, 0])):
        for j in range(len(matrix[0, :])):
            if matrix[i, j] <= tol:
                matrix[i, j] = 0
            else:
                matrix[i, j] = 1
    
    return matrix

def count_areas(gray_matrix, tol=205):
    _min_val = np.min(gray_matrix)
    gray_matrix = gray_matrix - _min_val
    _max_val = np.max(gray_matrix)
    gray_matrix = gray_matrix/_max_val * 255
    if gray_matrix.dtype != np.uint8:
        gray_matrix = gray_matrix.astype(np.uint8)
    if tol < 0 or tol > 255:
        print("wrong tol should in [0,255]")
    gray_matrix = np.pad(gray_matrix, pad_width=1, mode='constant', constant_values=0)
    _, gray_matrix = cv2.threshold(gray_matrix, tol, 255, cv2.THRESH_BINARY )
    morph_iters=1
    inverted = 255 - gray_matrix
    kernel = np.ones((3, 3), np.uint8)
    morph = inverted
    morph = cv2.resize(morph, (256, 256), interpolation=cv2.INTER_LINEAR)
    _, morph = cv2.threshold(morph, 55, 255, cv2.THRESH_BINARY )
    morph = 255 - morph
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=morph_iters)
    cnt, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(cnt)

def count_pixels_above_tol(image_matrix, tol):
    if not 0 <= tol <= 255:
        return 0, f"wrong tol should in [0,255], now is {tol}"
    gray_img = image_matrix
    count = np.sum(gray_img < tol)
    return count

def save_dual_grayscale(matrix_left, matrix_right, save_dir, filename, dpi=300, figsize=(12, 6)):

    matrix_left = matrix_left*255
    # matrix_right = matrix_right* 255
    # proccessing matrix_right
    rmin = np.min(matrix_right)
    matrix_right = matrix_right-rmin
    rmax = np.max(matrix_right)
    matrix_right = matrix_right/rmax * 255

    # print(np.max(matrix_left), "   ", np.max(matrix_right)) 
    # print(np.min(matrix_left), "   ", np.min(matrix_right)) 
    try:
        os.makedirs(save_dir, exist_ok=True)

        plt.figure(figsize=figsize)

        plt.subplot(1, 2, 1)
        plt.imshow(matrix_left, 
                  cmap='gray',
                  interpolation='nearest',
                  vmin=np.min(matrix_left),
                  vmax=np.max(matrix_left))
        plt.title('True Image')
        plt.axis('off')
    
        plt.subplot(1, 2, 2)
        plt.imshow(matrix_right, 
                  cmap='gray',
                  interpolation='nearest',
                  vmin=np.min(matrix_right),
                  vmax=np.max(matrix_right))
        plt.title('Pred. Image')
        plt.axis('off')
        plt.subplots_adjust(wspace=0.05)
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, 
                   bbox_inches='tight',
                   pad_inches=0.1,
                   dpi=dpi)
        plt.close()
        #tol
        cnt_true = count_pixels_above_tol(matrix_left, tol=127)
        cnt_pred = count_pixels_above_tol(matrix_right, tol=127)
        print(cnt_true/matrix_left.size, "   ", cnt_pred/matrix_right.size)   
        return cnt_true/matrix_left.size, cnt_pred/matrix_right.size
    except Exception as e:
        plt.close()
        raise RuntimeError(f"fail to save：{str(e)}")

def count_bw_areas(matrix_left, matrix_right, save_dir, tol=128):

    matrix_left = matrix_left*255
    rmin = np.min(matrix_right)
    matrix_right = matrix_right-rmin
    rmax = np.max(matrix_right)
    matrix_right = matrix_right/rmax * 255

    try:
        # os.makedirs(save_dir, exist_ok=True)
        #tol
        cnt_true = count_pixels_above_tol(matrix_left, tol=tol)
        cnt_pred = count_pixels_above_tol(matrix_right, tol=tol)
        # print(cnt_true/matrix_left.size, "   ", cnt_pred/matrix_right.size)   
        return cnt_true/matrix_left.size, cnt_pred/matrix_right.size
    except Exception as e:
        raise RuntimeError(f"fail to save：{str(e)}")


def save_l2_error(true_l2, pred_l2, save_path, dpi=300,figsize=(6,6)):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=figsize)
    plt.plot(true_l2, color='blue', label='true')
    plt.plot(pred_l2, color='red', label='pred')
    plt.legend()
    _save_path = os.path.join(save_path, 'l2.png')
    save_npy_true = os.path.join(save_path, 'true_l2.npy')
    save_npy_pred = os.path.join(save_path, 'pred_l2.npy')  
    plt.savefig(_save_path, 
            bbox_inches='tight',
            pad_inches=0.1,
            dpi=dpi)
    plt.close()
    np.save(save_npy_pred, pred_l2)
    np.save(save_npy_true, true_l2)
    
def save_area_error(true_area, pred_area, save_path, dpi=300,figsize=(6,6)):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=figsize)
    plt.plot(true_area, color='blue', label='true')
    plt.plot(pred_area, color='red', label='pred')
    plt.legend()
    _save_path = os.path.join(save_path, 'area.png')
    save_npy_true = os.path.join(save_path, 'true_area.npy')
    save_npy_pred = os.path.join(save_path, 'pred_area.npy')  
    plt.savefig(_save_path, 
            bbox_inches='tight',
            pad_inches=0.1,
            dpi=dpi)
    plt.close()
    np.save(save_npy_pred, pred_area)
    np.save(save_npy_true, true_area)
    
def save_sample_error(vector, save_path, dpi=300,figsize=(6,6)):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=figsize)
    plt.plot(vector, color='blue', label='true')
    plt.legend()
    _save_path = os.path.join(save_path, 'total_l2.png')
    save_error_path = os.path.join(save_path, 'total_l2.npy')
    plt.savefig(_save_path, 
            bbox_inches='tight',
            pad_inches=0.1,
            dpi=dpi)
    np.save(save_error_path, vector)

#**********************************#
#*            parameters          *#
#**********************************#



def start_test(path_true, path_pred, path,img_row=64, img_col=64, is_grain=False):

    print("\n", flush=True)
    print("\n", flush=True)
    print("\n", flush=True)
    print("%"*30, "  start  ", "%"*30, flush=True)
    
    print(path_true)
    print(path_pred)
    if is_grain == True:
        print("Grain phy", flush=True)
    pred = np.load(path_pred)
    trues = np.load(path_true)
    num_sample = len(pred) 
    
    print(f"total samples = {num_sample}", flush=True)
    total_l2 = np.zeros(num_sample)
    
    for i in range(num_sample):
        # print(f"now processing sample {i}", flush=True)
        if i%100 == 0:
            print(f"proccessing sample {i}", flush=True)
        sample_pred = pred[i]
        sample_true = trues[i]
        # print(os.path.join(path,f"sample_{i}"))
        if os.path.isdir(os.path.join(path,f"sample_{i}")) == False:
            os.mkdir(os.path.join(path,f"sample_{i}"))
        path_save = os.path.join(path,f"sample_{i}")
        true_l2 = np.zeros(len(sample_pred))
        pred_l2 = np.zeros(len(sample_pred))
        areas_pred = np.zeros(len(sample_pred))
        areas_true = np.zeros(len(sample_pred))    
        for j in range(len(sample_pred)):
            img_pred = sample_pred[j]
            img_true = sample_true[j]
            img_pred = img_pred.reshape(img_row, img_col)
            img_true = img_true.reshape(img_row, img_col)
            if is_grain == False:
                cnt_true, cnt_pred = count_bw_areas(img_true, img_pred, path_save)
                # cnt_true, cnt_pred = save_dual_grayscale(img_true, img_pred, path_save, f"step_{j}.png")
                true_l2[j] = cnt_true
                pred_l2[j] = cnt_pred
            else:
                image_size = img_true.size
                ave_area_pred = 1/count_areas(img_pred)
                ave_area_true = 1/count_areas(img_true)
                areas_pred[j] = ave_area_pred
                areas_true[j] = ave_area_true
        
        if is_grain == False:
            # save_l2_error(true_l2, pred_l2, path_save)
            total_l2[i] = np.linalg.norm(true_l2-pred_l2)
        else:
            # save_area_error(areas_true, areas_pred, path_save)
            total_l2[i] = np.linalg.norm(areas_true -areas_pred)
    if is_grain == False:
        save_sample_error(total_l2, path)
        err_l2 = np.linalg.norm(total_l2)
        print(err_l2)
        np.savetxt(os.path.join(path, 'err_l2.txt'), np.array([err_l2]))
    else:
        err_l2 = np.linalg.norm(total_l2)
        np.savetxt(os.path.join(path, 'err_l2.txt'), np.array([err_l2]))
        print(err_l2)
    # print(metrics)
    # metrics = metrics.reshape(1, -1)
    print("%"*30, "  end  ", "%"*30)



