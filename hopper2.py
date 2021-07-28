from dataclasses import dataclass


@dataclass
class LinComb:
    """Represents `c + k * E[a]`"""

    c: float
    k: float
    a: int

    def __repr__(self):
        return f"{self.c} + {self.k} * E[{self.a}]"

    def deepest(self):
        """
        c + k * E[a]
        = c + k * (E[a].c + E[a].k * E[a+1])
        = (c + k * E[a].c) + k * E[a].k * E[a+1]
        """
        if self.a >= len(E):
            return self
        else:
            next = LinComb(
                self.c + self.k * E[self.a].c, self.k * E[self.a].k, self.a + 1
            )
            return next.deepest()


E = [LinComb(0, 0, 0)]


def hopper2(depth: int):
    """We divide each state by the smallest value to get [1, 2^a]; denote this as state a.
    Then
      E[0] = 0
      E[a, a≥0] = 1 + 1/(1+2^a) (2^a E[a-1] + E[a+1])
    We're looking for 1 + E[1]
    """

    # Essentially a bidiagonal matrix solution
    for a in range(1, depth):
        """
        given E[a-1] = c + k * E[a]
        E[a] = 1 + 2^a/(1+2^a) E[a-1] + E[a+1] / (1+2^a)
        let u = 2^a/(1+2^a)
        E[a] = 1 + u E[a-1] + E[a+1] / (1+2^a)
        E[a] = 1 + u * c + k * u * E[a] + E[a+1] / (1+2^a)
        (1 - k * u) E[a] = (1+u*c) + (1/(1+2^a))E[a+1]
        E[a] = (1+u*c)/(1-k*u) + (1/((1+2^a)(1-k*u)))E[a+1]
        """
        tot = 1 + 2 ** a
        u = 2 ** a / tot
        denom = 1 - E[a - 1].k * u
        E.append(LinComb((1 + u * E[a - 1].c) / denom, 1 / (tot * denom), a + 1))

    print(1 + E[1].deepest().c)


hopper2(12)
