import unicodedata

text = open("answer.txt", "r").read()
open("n_answer.txt", "wb").write(unicodedata.normalize('NFKD', text).encode('ascii','ignore'))
print(unicodedata.normalize('NFKD', text).encode('ascii','ignore'))
