class carro:
    def __init__(self):
        self.__velocidade = 0

    @property
    def velocidade(self):
        return self.__velocidade

    def acelerar(self):
        self.__velocidade += 5
        return self.__velocidade

    def frear(self):
        self.__velocidade -= 5
        return self.__velocidade


class uno(carro, nome):
    self.nome = nome


class ferrari(carro):
    def acelerar(self):
        super().acelerar()
        return super().acelerar()


# declarar somente chamando a classe
c1 = carro()
# declarar usando outra classe
c1 = ferrari()
c2 = uno("Uno")


print(c1.acelerar())
print(c1.acelerar())
print(c1.acelerar())
print(c1.frear())
print(c1.frear())
print(c1.frear())

print(c2.nome())
print(c2.acelerar())
print(c2.acelerar())
print(c2.acelerar())
print(c2.frear())
print(c2.frear())
print(c2.frear())
