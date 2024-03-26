class contador:
    contador = 0  # atributo de classe

    @classmethod
    def inc(cls):
        cls.contador += 1
        return cls.contador

    @classmethod
    def dec(cls):
        cls.contador -= 1
        return cls.contador

    #  usado para quando não é utilizado nada da classe
    @staticmethod
    def mais_um(n):
        return n + 1


# c1 = contador()
# por se tratar de um metodo de classe, não é necessário iniciar uma instancia


print(contador.inc())
print(contador.inc())
print(contador.inc())
print(contador.inc())
print(contador.dec())
print(contador.dec())
print(contador.dec())

print(contador.mais_um(99))
