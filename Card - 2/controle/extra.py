nota = float(input('Informe o valor da nota do Aluno '))
comportado = True if input('Comportado: (s/n)') == 's' or 'S' else False
if nota >= 9 and comportado:
	print(f'Sua nota: {nota} esta com status Ouro, Parabéns!')
	#tabulação indicada através do TAB
	#espaçamento de aproximadamente 4 espaços
	
elif nota >= 7:
	print(f'Sua nota foi de: {nota}, parabéns você foi aprovado')

elif nota >= 5.5:
	print(f'Sua nota foi de {nota}, deverá realizar recuperação')

elif nota>= 3.0:
	print(f'Sua nota foi de {nota}, deverá além de realizar recuperação, realizar trabalho!')

else:
	print(f'Sua nota foi de {nota}, você foi Reprovado')
