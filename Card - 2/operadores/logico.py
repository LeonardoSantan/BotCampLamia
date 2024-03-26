b1 = True
b2 = False
b3 = True

print(b1 and b2)
#True se tudo for True

print(b1 or b2 or b3)
#True se algo é True

#Não existe XOR, para este caso utilizar o !=
print(b1 != b2)

print(not b1)
print(not b2)

print(b1 and not b2 and b3)

x = 3
y = 4

print(b1 and not b2 and x < y)