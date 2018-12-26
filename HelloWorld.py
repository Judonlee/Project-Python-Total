from pprint import pprint

if __name__ == '__main__':
    for a in range(2):
        for b in range(2):
            for c in range(2):
                for d in range(2):
                    list = [[a, b, c, d]]
                    for x in range(3):
                        data = []
                        for y in range(4 - x - 1):
                            if list[x][y] == list[x][y + 1]:
                                data.append(1)
                            else:
                                data.append(0)
                        list.append(data)
                    # pprint(list)
                    # exit()

                    counter = 0
                    for x in range(len(list)):
                        for y in range(len(list[x])):
                            counter += list[x][y]
                    # print(counter)
                    if counter == 5:
                        pprint(list)
