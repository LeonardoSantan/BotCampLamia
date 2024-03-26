for i in range(10): #range quantidade de repetições
	print(i, end=',') # 0-9
print()

for i in range(1,11): #range quantidade de repetições
	print(i, end=',') #1-10
print()

for i in range(1,100,7): #range quantidade de repetições
	print(i, end=',') #1-100 de 7 em 7
print()

for i in range(20,0,-3): #range quantidade de repetições
	print(i, end=',') #20-0 de 7 em 7
print()

nums = [2, 4, 6, 8]

for n in nums:
	print(n, end=',') #end consegue colocar como ele vai concluir essa linha '\n'
print()

texto = 'Python é muito massa!'

for letra in texto:
	print(letra, end=' ')
print()
for n in {1, 2, 3, 4, 4, 4}:
	print(n, end=' ') # irá trazer 1-4 devido set não permitir valor duplicados
print()
produto = {
		'nome': 'Caneta',
		'preço': 8.80,
		'desc': 0.5
}



for atrib in produto:
	print(atrib,'-->' , produto[atrib], end=' ')
print()

for atrib, valor in produto.items():
	print(atrib,'-->' , valor)
print()

for valor in produto.values():
	print(valor,end=' ')
print()

for atrib in produto.values():
	print(valor, end=' ')
print()

