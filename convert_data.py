input_file = "./THUCNews/data/dev.txt"
output_file = "./THUCNews/data/dev_1280.txt"
static = {}
maxSize = 128
with open(output_file, mode="w", encoding="utf-8") as output:
    with open(input_file, mode="r", encoding="utf-8") as input:
        for line in input.readlines():
            label = line.split("\t")[1]
            if static.get(label) is None:
                static[label] = 0
            if static[label] < maxSize:
                output.write(line)
                static[label] = static[label] + 1


