import functools as fc


def somar_nota(delta):
    def calc(nota):
        return nota + delta

    return calc


notas = [6.4, 7.8, 7.2, 4.8, 8.4]
notas_finais_1 = list(map(somar_nota(1.5), notas))
notas_finais_2 = list(map(somar_nota(1.6), notas))
print(notas_finais_1)
print(notas_finais_2)


def soma(a, b):
    return a + b


# primeiro elemento acomulador e segundo o que será percorrido
total = fc.reduce(soma, notas, 0)
print(total)

# print('Primeiro exemplo: ')
# for i, nota in enumerate(notas):
#     notas[i] = nota + 1.5
#     print(f'({i}) = {nota}')
# print(f' Primeiro exemplo {notas}')
# print('Segundo exemplo: ')
# através do tamanho do array
# for i in range(len(notas)):
#     notas[i] = notas[i] +1.5
