import matplotlib.pyplot as plt

ans = [55, 42, 33, 26, 20, 20, 18, 17, 16, 16, 15, 13, 12, 12, 11, 11, 11, 11, 10, 10, 9, 9, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]

print(sum(ans[0:20]))

fig = plt.bar(range(1,len(ans)+1), ans)

#plt.show()