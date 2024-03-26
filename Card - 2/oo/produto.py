class produto:
    def __init__(self, nome, preco, desc):
        self.nome = nome
        self.__preco = preco
        self.desc = desc
        # decorator preco_final é tratado com variavel e não como um metodo

    @property
    def preco_final(self):  # não é necessário passar nenhum parametro
        return (1 - self.desc) * self.preco

    @property
    def preco(self):
        return self.__preco

    @preco.setter
    def preco(self, novo_preco):
        if novo_preco > 0:
            self.__preco = novo_preco


# def __init__ é inicializador padrão de classe

# se a classe for vazia usar pass

p1 = produto("Caneta", 1.99, 0.1)
p2 = produto("Caderno", 5.99, 0.5)

p1.preco = 15
p2.preco = 18


print(p1.nome, p1.preco, p1.desc, p1.preco_final)
print(p2.nome, p2.preco, p2.desc, p2.preco_final)
