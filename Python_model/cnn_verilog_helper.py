# for i in range(28):
#     print("assign outw[{}:{}] = ".format(28*(i+1)*8-1,28*i*8))
#     for j in range(28):
#         if j==0:
#             print("{{weights[{}][{}], ".format(i,27-j),end = " ")
#         elif j==27:
#             print("weights[{}][{}] }} ;".format(i,27-j))
#         else:
#             print("weights[{}][{}], ".format(i, 27 - j),end = " ")

for i in range(784):
    print("I(in{}) + ".format(i+1),end=" ")