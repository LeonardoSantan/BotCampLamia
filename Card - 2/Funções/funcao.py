# funções são definidas com underline + 1 nome
def saudacao_pela_tarde(nome: any, idade: any):
    print(f"Olá {nome}, como esta? Você nem parece ter {idade} anos")


if __name__ == "__main__":
    saudacao_pela_tarde("Ana", idade=30)


def soma_mult(a, b, x):
    return a + b * x
