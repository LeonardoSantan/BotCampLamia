from functools import reduce


alunos = [
    {"nome": "Ana", "nota": 7.2},
    {"nome": "Breno", "nota": 8.1},
    {"nome": "Claudia", "nota": 8.7},
    {"nome": "Pedro", "nota": 6.4},
    {"nome": "Rafael", "nota": 6.7},
    {"nome": "Léo", "nota": 5.1},
]


aluno_aprovado = lambda aluno: aluno["nota"] >= 7  # noqa: E731
# aluno_honra = lambda aluno: aluno["nota"] >= 9  # noqa: E731
obter_nota = lambda aluno: aluno["nota"]  # noqa: E731


# trazer somatório de produtos em conjunto com reduce
somar = lambda a, b: a + b  # noqa: E731

alunos_aprovados = list(filter(aluno_aprovado, alunos))


notas_alunos_aprovados = map(obter_nota, alunos_aprovados)

total = reduce(somar, notas_alunos_aprovados, 0)

# Valor 0 ao fim, quer dizer vlaor inicial
# print(list(notas_alunos_aprovados))


print(total)

print(total / len(list(alunos_aprovados)))
