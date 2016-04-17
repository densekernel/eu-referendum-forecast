import sys

filename = sys.argv[-1]

with open(filename) as f:
    lines = f.readlines()
    counter = 0
    true_counter = 0.0
    for line in lines:
        first_char = int(line[0])
        if counter < 50 and first_char == 0 or (counter >= 50 and first_char == 4):
            true_counter += 1
        counter += 1
print(true_counter / counter)
