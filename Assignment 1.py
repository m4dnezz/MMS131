def check_validity(filename:str):
    with open(file=filename) as file:
        rows = 0
        a = 0
        b = 0
        c = 0
        d = 0
        mydict = dict()
        for line in file:
            rows += 1
            line = line.split("=")
            line[1] = float(line[1])
            mydict[line[0]] = line[1]
            assert 1 >= float(line[1]) >= 0

            if line[0].startswith("a"):
                a += 1
            if line[0].startswith("b"):
                b += 1
            if line[0].startswith("c"):
                c += 1
            if line[0].startswith("d"):
                d += 1

    assert rows == sum([a, b, c, d])
    assert a == 3
    assert b == 9
    assert c == 9
    assert d == 27
    assert mydict["a1"] + mydict["a2"] + mydict["a3"] == 1

    assert mydict["c1|a1"] + mydict["c2|a1"] + mydict["c3|a1"] == 1
    assert mydict["c1|a2"] + mydict["c2|a2"] + mydict["c3|a2"] == 1
    assert mydict["c1|a3"] + mydict["c2|a3"] + mydict["c3|a3"] == 1

    assert mydict["b1|a1"] + mydict["b2|a1"] + mydict["b3|a1"] == 1
    assert mydict["b1|a2"] + mydict["b2|a2"] + mydict["b3|a2"] == 1
    assert mydict["b1|a3"] + mydict["b2|a3"] + mydict["b3|a3"] == 1

    assert mydict["d1|a1,b1"] + mydict["d2|a1,b1"] + mydict["d3|a1,b1"] == 1
    assert mydict["d1|a1,b2"] + mydict["d2|a1,b2"] + mydict["d3|a1,b2"] == 1
    assert mydict["d1|a1,b3"] + mydict["d2|a1,b3"] + mydict["d3|a1,b3"] == 1

    assert mydict["d1|a2,b1"] + mydict["d2|a2,b1"] + mydict["d3|a2,b1"] == 1
    assert mydict["d1|a2,b2"] + mydict["d2|a2,b2"] + mydict["d3|a2,b2"] == 1
    assert mydict["d1|a2,b3"] + mydict["d2|a2,b3"] + mydict["d3|a2,b3"] == 1

    assert mydict["d1|a3,b1"] + mydict["d2|a3,b1"] + mydict["d3|a3,b1"] == 1
    assert mydict["d1|a3,b2"] + mydict["d2|a3,b2"] + mydict["d3|a3,b2"] == 1
    assert mydict["d1|a3,b3"] + mydict["d2|a3,b3"] + mydict["d3|a3,b3"] == 1
    print("Validity of CPT OK")

check_validity("inOK2.txt")