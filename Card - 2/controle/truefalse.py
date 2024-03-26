a = 'valor'

if a:
	print('Existe!!')
else:
	print('Não existe ou zero ou vazio')
#Ira retornar como verdadeiro

a = 'valor'
a = 0
if a:
	print('Existe!!')
else:
	print('Não existe ou zero ou vazio')
#irá retornar como falso

a = 'valor'
a = 0
a = -0.000001
if a:
	print('Existe!!')
else:
	print('Não existe ou zero ou vazio')
#irá retornar como verdadeiro

a = 'valor'
a = '' #falso por ser vazio
a = ' ' #verdadeiro devido haver espaço
a = [] #falso por estar vazio
a = {} #falso por estar vazio
if a:
	print('Existe!!')
else:
	print('Não existe ou zero ou vazio')