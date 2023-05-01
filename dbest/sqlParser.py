from collections import defaultdict
import shlex


class Filter:
    def __init__(self):
        self.lb = None
        self.ub = None
        self.equalities = []
class Parser:
    def parse(self,query, attributes):    
        charset = ['=','>','<','>=','=<', '<=', '=>']
        aggs = ['count', 'sum', 'avg']
        keywords = ['select','where', 'from', 'and']
        query = query.strip().lower()
        attributes = [att.lower() for att in attributes]
        for ch in charset:
            query = query.replace(ch, ' '+ch+' ')
        
            
        tokens_all = shlex.split(query,posix=False)
        tokens = []
        i = 0
        while i < len(tokens_all) - 1:
            if (tokens_all[i]+tokens_all[i+1]) in charset:
                tokens.append(tokens_all[i]+tokens_all[i+1])
                i = i + 1
            else:
                tokens.append(tokens_all[i])
            i = i+1
        tokens.append(tokens_all[-1])

        if tokens[0]!='select':
            print ('query should start with select')
            return False, None, None
        aggregation = tokens[1].lower()        
        if 'count' in aggregation:
            agg = 'count'
        elif 'sum' in aggregation:
            agg = 'sum'
        elif 'avg' in aggregation:
            agg = 'avg'
        else:
            raise ValueError("ValueError exception thrown")("aggregation function {} not implemented!".format(tokens[1]))

        conditions = defaultdict(Filter)
        for ch in charset:
            try:
                #idx = tokens.index(ch)
                indices = [i for i, x in enumerate(tokens) if x == ch]
                for idx in indices:
                    before = tokens[idx-1]
                    next = tokens[idx+1]
                    if ch!='=' and (Parser.is_number(before) == Parser.is_number(next)):
                        print ("error near {}".fromat(ch))
                        return False, None, None
                    
                    
                    if ch=='<':
                        if Parser.is_number(before):
                            conditions[next].lb = float(before) + 1e-10
                        elif Parser.is_number(next):
                            conditions[before].ub = float(next) - 1e-10
                    if ch=='=<' or ch=='<=':
                        if Parser.is_number(before):
                            conditions[next].lb = float(before)
                        elif Parser.is_number(next):
                            conditions[before].ub = float(next)

                    if ch=='>':
                        if Parser.is_number(before):
                            conditions[next].ub = float(before) - 1e-10
                        elif Parser.is_number(next):
                            conditions[before].lb = float(next) + 1e-10
                    if ch=='>=' or ch=='=>':
                        if Parser.is_number(before):
                            conditions[next].ub = float(before)
                        elif Parser.is_number(next):
                            conditions[before].lb = float(next)
                    
                    if ch=='=':

                        if Parser.is_number(before) and Parser.is_number(next):
                            print ("error near {}".format(ch))
                            return False, None, None
                        if Parser.is_number(before):
                            conditions[next].equalities.append(before)
                        if Parser.is_number(next):
                            conditions[before].equalities.append(next)
                        if before.startswith("'") or before.startswith('"'):
                            conditions[next].equalities.append(before.replace("'","").replace('"',''))
                        elif next.startswith("'") or next.startswith('"'):
                            conditions[before].equalities.append(next.replace("'","").replace('"',''))
                        else:
                            print ("error near {}".format(ch))
                            return False, None, None
            except:
                pass

                

        keys = conditions.keys()
        for key in keys:
            if key not in attributes:
                print ('{} is unknown!'.format(key))
                return False, None, None
        
        
        
        return True, conditions, agg

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            return False
                