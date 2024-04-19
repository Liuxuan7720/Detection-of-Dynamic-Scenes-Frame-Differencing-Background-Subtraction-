list = [1, 2, 2, 3, 3]
record = [0] * 100
for i in range(len(list)):
    k = list[i]
    while k >= 0:
        record[k] += 1
        k -=1
max_val = 0
for i in range(0, 100):
    max_val = max(max_val, min(i, record[i]))
print(max_val)