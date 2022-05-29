#Topic: AR - efficient apriori
#-----------------------------
#libraries

#https://en.wikipedia.org/wiki/Apriori_algorithm
#https://en.wikipedia.org/wiki/Association_rule_learning

!pip install efficient_apriori


from efficient_apriori import apriori

transactions = [('eggs', 'bacon', 'soup', 'milk'), 
                ('eggs', 'bacon', 'apple', 'milk'), 
                ('soup', 'bacon', 'banana')]


transactions



itemsets, rules = apriori(transactions)

print(itemsets)

print(rules)


for rule in rules:
    print (rule)







rules_1 = list(filter(lambda rule: len(rule.lhs) == 1 and len(rule.rhs) == 1, rules))


#rules_1

for rule1 in rules_1:
    print (rule1)
    



'''
# Nested list of student's info in a Science Olympiad
# List elements: (Student's Name, Marks out of 100 , Age)
participant_list = [
    ('Alison', 50, 18),
    ('Terence', 75, 12),
    ('David', 75, 20),
    ('Jimmy', 90, 22),
    ('John', 45, 12)
]

sorted_list = sorted(participant_list, key=lambda item: item[2])
print(sorted_list)
'''

