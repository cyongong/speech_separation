
import torch
from scipy.io import wavfile
from tqdm import tqdm
import numpy as np
import os

from config import args
from models.visual_frontend import VisualFrontend
from utils.preprocessing import preprocess_sample



def main():

    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")



    #declaring the visual frontend module
    vf = VisualFrontend()
    vf.load_state_dict(torch.load("/home/codud9914/s1/deep_avsr/visual_frontend.pt", map_location=device))
    vf.to(device)


    #walking through the data directory and obtaining a list of all files in the dataset
    #filesList = list()
    for root, dirs, files in os.walk("/home/codud9914/s1/deep_avsr/mv3"):
        for file in files:
            if file.endswith("a.npy"):
                #filesList.append(os.path.join(root, file[:-4]))
                print(file.shape())
            if file.endswith("a.wav"):
                print(file.shape())





    #Preprocessing each sample
    print("\nNumber of data samples to be processed = %d" %(len(filesList)))
    print("\n\nStarting preprocessing ....\n")

    params = {"roiSize":args["ROI_SIZE"], "normMean":args["NORMALIZATION_MEAN"], "normStd":args["NORMALIZATION_STD"], "vf":vf}
    for file in tqdm(filesList, leave=True, desc="Preprocess", ncols=75):
        preprocess_sample(file, params)

    print("\nPreprocessing Done.")
    
  if __name__ == "__main__":
    main()
