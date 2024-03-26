x = 0

while x != -1:
	x = float(input('Informe o número ou -1 para sair: '))

print('Fim!')

qtde=0
total=0
nota = 0

while nota != -1:
    nota = float(input('Informe o número ou -1 para sair: '))
    if nota != -1:
        qtde += 1
        total += nota
        print(f'{total} -- {qtde}')
print(f'A média da turma é ({total} / {qtde})')

