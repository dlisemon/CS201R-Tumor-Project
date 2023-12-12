import os

source_path = "/Users/jhgirald/Desktop/tumorFiles/tumor/"
prefix = "sub-"
t1 = "_T1w.nii.gz"
t2 = "_T2w.nii.gz"
file_div = "/"
destination_path ="/Users/jhgirald/Desktop/tumorFiles/CS201R-Tumor-Project/CNN_Classification/tumor_div/"

for i in range(1, 43):
    formatted_number = "{:03d}".format(i)
    print(source_path + prefix + formatted_number + t1)
    os.system(f"mkdir dataset_div/{formatted_number}")
    # os.system("pwd")
    # os.system(f"cp {source_path + prefix + formatted_number + t1} {destination_path + formatted_number + file_div + prefix + formatted_number+ t1}")
    # os.system(f"cp {source_path + prefix + formatted_number + t2} {destination_path + formatted_number + file_div + prefix + formatted_number+ t2}")
