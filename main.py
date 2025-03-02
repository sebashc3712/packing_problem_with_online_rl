import oskp_rl
import torch

cut_files = ['cut_1.pt', 'cut_2.pt', 'rs.pt']
instances = []

for i in cut_files:
    checkpoint = torch.load("approachesO3DKP\\" + i)
    instances.append(checkpoint)
    print(i)
    print("Total batches: ",len(checkpoint))
    print("boxes in first batch: ",len(checkpoint[0]))


    
oskp_rl.train(instances[0])
