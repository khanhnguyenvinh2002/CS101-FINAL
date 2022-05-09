'''
label rules and back rules for each symbol
'''

class Rules():
    def __init__(self):
        self.symbols = ['dot', 'tan', ')', '(', '+', '-', '|', 'sqrt', '1', '0', '3', '2', '4', '6','8', 'mul', 'pi', 'sin', 'A', 'cube root', 'co', 'os', 'mn', 'frac', 'cos', 'delta', 'a', 'c', 'b', 'bar', 'd', 'f', 'i', 'h', 'k', 'm', 'o', 'n', 'p', 's', 't', 'y', 'x', 'z', 'v', 'l', 'w', 'div', 'z_no_line', 'z_line']

    def get_rules(self):
        rules = {}
        lst = [float(0)] * len(self.symbols) # buckets
        for i in range(0,len(self.symbols)):
            lst[i] = float(1) # filling each symbol in a unique box
            rules[self.symbols[i]] = lst[:] # mapping a symbol to a list of [0,0,1,...0,0,0]
            lst[i] = float(0)

        rules['o'] = rules['0'] # make o's label 0 and frac, bar's label -, mul's label x
        rules['frac'] = rules['-']
        rules['bar'] = rules['-']
        rules['mul'] = rules['x']
        return rules

    # def getbrules(self):
    #     brules = {}
    #     for i in range(0,len(self.symbols)):
    #         brules[i] = self.symbols[i]
    #     return brules # note that 0 and o, frac and bar and -, x and mul are in the same bucket
