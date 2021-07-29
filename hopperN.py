import scipy.sparse
import scipy.sparse.linalg
from dataclasses import dataclass
from multiset import FrozenMultiset
import heapq
import numpy as np
from tabulate import tabulate
from functools import lru_cache


def tabulate_print(data, **kwargs):
    print(
        tabulate(data, headers="firstrow", tablefmt="plain", floatfmt=".13f", **kwargs)
    )


class State(FrozenMultiset):
    @lru_cache(maxsize=None)
    def with_val_picked(self, val):
        assert val in self
        if val == 1:
            mapping = {(2 * v): mult for v, mult in self.items()}
            # One 1 does not turn into a 2
            mapping[1] = 1
            mapping[2] -= 1
            return State(mapping)
        else:
            # Double everything, but divide by 2 since 2 is a common factor
            # Cancels out, so just copy
            mapping = {v: mult for v, mult in self.items()}
            # One val does not turn into a 2*val
            # Divided by 2, so one val/2 does not turn into a val
            mapping[val] -= 1
            half = val // 2
            mapping[half] = 1 if half not in mapping else mapping[half] + 1
            return State(mapping)

    def is_final(self):
        # { 1: n }
        return max(self) == 1


def get_states(n: int, k: int):
    root = State({1: 1, 2: n - 1})
    # I had trouble getting OrderedSet to work, so effectively making my own
    states_ordered = [root]
    state_to_index = {root: 0}
    # The state heap is a min-heap, so in order to get the state
    # with the highest probability, we index using the negative of the probability
    state_heap = [(-1, root)]

    while len(states_ordered) < k:
        assert len(state_heap) > 0
        from_prob_neg, from_state = heapq.heappop(state_heap)
        for val in from_state.distinct_elements():
            next_state = from_state.with_val_picked(val)
            if not next_state.is_final() and next_state not in state_to_index:
                states_ordered.append(next_state)
                state_to_index[next_state] = len(states_ordered) - 1
                heap_item = (from_prob_neg * val / sum(from_state), next_state)
                heapq.heappush(state_heap, heap_item)
                if len(states_ordered) >= k:
                    break

    return states_ordered, state_to_index


def get_transition_matrix(k, states_ordered, state_to_index, lower_bound: bool):
    # transition_mat[i, j] is the probability of transitioning from state i to state j
    data = []

    for from_index, from_state in enumerate(states_ordered):
        expected_total_weight = sum(from_state)
        total_weight = 0
        from_data = []
        for val in from_state.distinct_elements():
            to_state = from_state.with_val_picked(val)
            weight = val * from_state[val]
            if to_state in state_to_index:
                total_weight += weight
                to_index = state_to_index[to_state]
                from_data.append((from_index, to_index, weight))
            elif to_state.is_final():
                total_weight += weight
        denom = expected_total_weight if lower_bound else total_weight
        for i, j, weight in from_data:
            data.append((i, j, weight / denom))

    i, j, prob = zip(*data)

    # The square matrix A will be converted into CSC or CSR form
    return scipy.sparse.coo_matrix((prob, (i, j)), shape=(k, k))


def solve_transition_matrix(k, transition_mat):
    # Solve Tx+(1 1 ... 1 1) = x where T is the transition matrix
    # (I_n-T)x = (1 1 ... 1 1)
    # Ax = b
    A = scipy.sparse.identity(k) - transition_mat
    b = np.ones((k, 1))
    return scipy.sparse.linalg.spsolve(A, b)


def hopperN(n: int, k: int, lower_bound: bool = True):
    """If lower_bound is True, guarantee that the result is a lower bound;
    otherwise, the result is a better approximation which may be higher or lower"""
    if n == 1:
        return 1
    assert n > 1
    states_ordered, state_to_index = get_states(n, k)
    transition_mat = get_transition_matrix(
        k, states_ordered, state_to_index, lower_bound
    )
    sol = solve_transition_matrix(k, transition_mat)
    # We're looking for 1+E[(1,2,2,...2)]
    return 1 + sol[0]


def bound_message(lower_bound: bool):
    return "Guaranteed lower bound (≥)" if lower_bound else "Approximation (≈)"


def table(lower_bound: bool):
    n_values = list(range(2, 8))
    k_values = [250, 500, 1000]

    print(bound_message(lower_bound))
    data = [["n\k"] + k_values] + [
        [n] + [hopperN(n, k, lower_bound) for k in k_values] for n in n_values
    ]
    tabulate_print(data)


def bests(lower_bound: bool):
    nk = (
        [
            (2, 1000),
            (3, 1000),
            (4, 1000),
            (5, 1000),
            (6, 2500),
            (7, 10000),
            (8, 15000),
            (9, 20000),
            (10, 20000),
            (11, 20000),
            (12, 20000),
        ]
        if lower_bound
        else [
            (2, 100),
            (3, 100),
            (4, 100),
            (5, 1000),
            (6, 2000),
            (7, 3000),
            (8, 4000),
            (9, 5000),
            (10, 6000),
            (11, 7000),
            (12, 8000),
            (13, 9000),
            (14, 10000),
        ]
    )
    print(bound_message(lower_bound))
    data = [["n", "k", "→", "hopperN(n,k)"]] + [
        [n, k, "→", hopperN(n, k, lower_bound)] for n, k in nk
    ]
    tabulate_print(data)


if __name__ == "__main__":
    table(True)
    table(False)
    # bests(False)
