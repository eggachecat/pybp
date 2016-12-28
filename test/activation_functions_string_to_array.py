from pynn import nnaf

original_afs_type = [("purelin", 1), ("tanh", 1), 10]
afs_str = str(original_afs_type)
print(afs_str)

afs_type = eval(afs_str)
nnaf.generateActivationFunctions(afs_type)