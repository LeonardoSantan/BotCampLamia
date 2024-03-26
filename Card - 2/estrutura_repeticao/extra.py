pessoas = ['Gui', 'Rebeca']
adj = ['Sapeca', 'Inteligente']
for p in pessoas:
	for a in adj:
		print(f'{p} é {a}!',end= ' ')
print()
for i in [1, 2, 3]:
	pass # gerar uma classe vazia

for i in range(1, 11):
	if i % 2:
		continue # continua laço mesmo se false
	print(i,end= ' ')
print()
for i in range(1,11):
	if i == 5:
		break # sai do laço de repetição
	print(i,end= ' ')
print()
print('Fim! ')