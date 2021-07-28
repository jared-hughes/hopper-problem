import scipy.sparse
import scipy.sparse.linalg
from dataclasses import dataclass
from multiset import Multiset
import heapq
import numpy as np
from tabulate import tabulate


class State(Multiset):
    def with_val_picked(self, val):
        assert val in self
        copy = self.copy()
        if val == 1:
            mapping = {(2 * v): mult for v, mult in self.items()}
            # One 1 does not turn into a 2
            mapping[1] = 1
            mapping[2] -= 1
            return State(mapping)
        else:
            # Double everything, but divide by 2 since 2 is a common factor
            # Cancels out, so just copy
            copy = self.copy()
            # One val does not turn into a 2*val
            # Divided by 2, so one val/2 does not turn into a val
            copy[val] -= 1
            copy[val // 2] += 1
            return copy


def hopperN(n: int, k: int):
    """Select k states depth-first, then treat all higher states as 0 and solve"""
    final_state = State({1: n})
    root = State({1: 1, 2: n - 1})
    states = [root]
    # The state heap is a min-heap, so in order to get the state
    # with the highest probability, we index using the negative of the probability
    state_heap = [(-1, root)]

    while len(states) < k:
        assert len(state_heap) > 0
        from_prob_neg, from_state = heapq.heappop(state_heap)
        for val in from_state.distinct_elements():
            next_state = from_state.with_val_picked(val)
            if next_state != final_state and next_state not in states:
                states.append(next_state)
                state_heap.append(
                    (from_prob_neg * val / sum(from_state), next_state))
                if len(states) >= k:
                    break

    """ Compute transition matrix """
    # transition_mat[i, j] is the probability of transitioning from state i to state j
    transition_mat = scipy.sparse.dok_matrix((k, k), dtype=np.float32)

    for from_index, from_state in enumerate(states):
        for val in from_state.distinct_elements():
            next_state = from_state.with_val_picked(val)
            for to_index, to_state in enumerate(states):
                if next_state == to_state:
                    prob = val * from_state[val] / sum(from_state)
                    transition_mat[from_index, to_index] = prob
                    break

    """ Solve """

    # Solve Tx+(1 1 ... 1 1)=x where T is the transition matrix
    # (I_n-T)x=(1 1 ... 1 1)
    # Ax=b

    A = scipy.sparse.identity(k) - transition_mat
    b = np.ones((k, 1))

    sol = scipy.sparse.linalg.spsolve(A, b)

    # We're looking for 1+E[(1,2,2,...2)]
    return 1 + sol[0]


def table():
    n_values = list(range(2, 6))
    k_values = [250, 500, 1000]

    data = [["n\k"] + k_values] + [
        [n] + [hopperN(n, k) for k in k_values] for n in n_values
    ]

    print(tabulate(data, headers="firstrow", tablefmt="plain", floatfmt=".13f"))


def bests():
    nk = [
        (2, 1000),
        (3, 1000),
        (4, 1000),
        (5, 1000),
        (6, 2500),
        (7, 10000),
        (8, 15000),
        (9, 20000),
    ]
    print("n", "k", "=>", "hopperN(n, k)")
    for n, k in nk:
        print(n, k, "=>", hopperN(n, k))


if __name__ == "__main__":
    table()
