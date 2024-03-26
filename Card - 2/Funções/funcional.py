def soma(a, b):
    return a + b


def sub(a, b):
    return a - b


somar = soma
print(somar(3, 4))


def operacao_aritimetica(fn, op1, op2):
    return fn(op1, op2)


resultado = operacao_aritimetica(soma, 13, 48)
print(resultado)

resultado = operacao_aritimetica(sub, 13, 48)
print(resultado)


def soma_parcial(a):
    def concluir_soma(b):
        return a + b

    return concluir_soma


fn = soma_parcial(10)
resultado_final = fn(12)
print(resultado_final)
