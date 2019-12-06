
'''
convert img_to_gif script
Copyright (c) Xiangzi Dai, 2019
'''
import imageio
import os
import sys
def create_gif(or_path,source, name, duration):
    """
    Only support .png format image
    """
    frames = []
    for img in source:
        frames.append(imageio.imread(os.path.join(or_path,img)))
    imageio.mimsave(name, frames, 'GIF', duration=duration)
    print("ok")

def get_list(file_path):
    """
    get file order by file create time
    """
    dir_list = os.listdir(file_path)
    if not dir_list:
        return
    else:
        dir_list = sorted(dir_list,key=lambda x: os.path.getmtime(os.path.join(file_path, x)))
        return dir_list

def main(paths,start=None,end=None):
    """
    paths: target dir
    """
    pic_list = get_list(paths)
    gif_name = "result.gif" 
    duration_time = 0.2 #duration time
    # create gif
    create_gif(paths,pic_list[start:end], gif_name, duration_time)

if __name__ == '__main__':
    """
    default: all images on ./data_dir
    example: python3 convert_gif.py ./data_dir   
                or
             python3 convert_gif.py ./data_dir 80 600
    """
    nums = len(sys.argv)
    if nums==2:
        main(sys.argv[1])
    elif nums==3:
        main(sys.argv[1],int(sys.argv[2]))
    elif nums==4:
        if int(sys.argv[2])<=int(sys.argv[3]):
            main(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]))
        else:
            print("start_num should less than end_num")
    else:
        print("please input vaild param")

