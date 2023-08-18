import os
from tqdm import tqdm

#Get all dataset frames, empty frames, frames with annotations, frames with litter, and instances of litter
#Works in already pooled txt files in folder "all_txt"
if __name__ == "__main__":
    cwd = r"C:\Users\Computing\OneDrive - University of Lincoln (1)\Documents\external_roads\external_output_blend"
    print(cwd)
    d = os.path.join(cwd,"labels")
    txt_fs = os.listdir(d)
    frame_count= len(txt_fs)
    empty_frame_count = 0
    litter_frame_count = 0
    litter_instances_count = 0

    for f in tqdm(txt_fs):
        if f.endswith('.txt'): # if text file (annotation)
            ann_txt = open(os.path.join(d,f), 'r')
            lines = ann_txt.readlines() # read txt file
            ann_txt.close()

            if lines: # if lines, theres annotations AKA litter present
                litter_frame_count += 1
                for l in lines:
                    litter_instances_count +=1 #count lines AKA litter

            else:
                empty_frame_count += 1 #otherwise count as empty frame

    print("Frame Count:",frame_count)
    print("Empty Frame Count:",empty_frame_count)
    print("litter Frame Count:",litter_frame_count)
    print("litter Instances Count:",litter_instances_count)


