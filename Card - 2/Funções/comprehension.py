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
obter_nota = lambda aluno: aluno["nota"]  # noqa: E731
somar = lambda a, b: a + b  # noqa: E731

alunos_aprovados = [aluno["nota"] for aluno in alunos if aluno["nota"] >= 7]
total = reduce(somar, alunos_aprovados, 0)

# print(total)

print(total / len(list(alunos_aprovados)))
