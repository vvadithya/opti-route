import random
from enum import Enum
from typing import List, Tuple, Union


class MutationOperatorType(str, Enum):
    TWO_OPT = "2-opt"
    SWAP = "swap"
    INSERTION = "insertion"
    RANDOM = "random"


def two_opt_reversal(route: List[str]) -> List[str]:
    """
    2-Opt Edge Reversal: Reverses a sub-sequence of waypoints between index i and j.
    Preserves start node (index 0) and end node (index -1).

    :param route: List of node IDs.
    :return: Mutated route list.
    """
    if len(route) <= 3:
        return list(route)

    new_route = list(route)
    # Pick two distinct indices inside intermediate waypoints
    n = len(route) - 1
    i, j = sorted(random.sample(range(1, n), 2))
    new_route[i : j + 1] = reversed(new_route[i : j + 1])
    return new_route


def node_swap(route: List[str]) -> List[str]:
    """
    Customer Swap: Swaps positions of two intermediate nodes in a route.
    Preserves start and end nodes.

    :param route: List of node IDs.
    :return: Mutated route list.
    """
    if len(route) <= 3:
        return list(route)

    new_route = list(route)
    n = len(route) - 1
    i, j = random.sample(range(1, n), 2)
    new_route[i], new_route[j] = new_route[j], new_route[i]
    return new_route


def node_insertion(route: List[str]) -> List[str]:
    """
    Customer Insertion: Removes an intermediate node from position i and inserts it at position j.
    Preserves start and end nodes.

    :param route: List of node IDs.
    :return: Mutated route list.
    """
    if len(route) <= 3:
        return list(route)

    new_route = list(route)
    n = len(route) - 1
    i, j = random.sample(range(1, n), 2)
    node = new_route.pop(i)
    new_route.insert(j, node)
    return new_route


def multi_route_customer_swap(routes: List[List[str]]) -> List[List[str]]:
    """
    Multi-vehicle Customer Swap: Swaps a customer node between two distinct vehicle routes.
    """
    valid_routes = [r for r in routes if len(r) > 2]
    if len(valid_routes) < 2:
        # Fall back to single-route swap if only 1 route available
        if valid_routes:
            idx = routes.index(valid_routes[0])
            new_routes = [list(r) for r in routes]
            new_routes[idx] = node_swap(new_routes[idx])
            return new_routes
        return [list(r) for r in routes]

    new_routes = [list(r) for r in routes]
    r1_idx, r2_idx = random.sample([i for i, r in enumerate(new_routes) if len(r) > 2], 2)

    r1, r2 = new_routes[r1_idx], new_routes[r2_idx]
    n1_idx = random.randint(1, len(r1) - 2)
    n2_idx = random.randint(1, len(r2) - 2)

    r1[n1_idx], r2[n2_idx] = r2[n2_idx], r1[n1_idx]
    return new_routes


def multi_route_customer_insertion(routes: List[List[str]]) -> List[List[str]]:
    """
    Multi-vehicle Customer Insertion: Moves a customer node from one vehicle route to another.
    """
    valid_routes = [r for r in routes if len(r) > 2]
    if len(routes) < 2 or not valid_routes:
        return [list(r) for r in routes]

    new_routes = [list(r) for r in routes]
    src_idx = random.choice([i for i, r in enumerate(new_routes) if len(r) > 2])
    dst_idx = random.choice([i for i in range(len(new_routes)) if i != src_idx])

    src_route = new_routes[src_idx]
    dst_route = new_routes[dst_idx]

    node_idx = random.randint(1, len(src_route) - 2)
    node = src_route.pop(node_idx)

    insert_idx = random.randint(1, max(1, len(dst_route) - 1))
    dst_route.insert(insert_idx, node)

    return new_routes


def apply_mutation(
    route_or_routes: Union[List[str], List[List[str]]],
    operator_type: Union[str, MutationOperatorType] = MutationOperatorType.RANDOM,
) -> Union[List[str], List[List[str]]]:
    """
    Applies the specified mutation operator to a route (List[str]) or multi-route (List[List[str]]).

    :param route_or_routes: Solution sequence or list of vehicle routes.
    :param operator_type: Operator strategy name ("2-opt", "swap", "insertion", "random").
    :return: Mutated solution copy.
    """
    op_str = str(operator_type).lower()

    if op_str == MutationOperatorType.RANDOM or op_str not in ("2-opt", "swap", "insertion"):
        op_str = random.choice(["2-opt", "swap", "insertion"])

    # Check if single route or multi-route
    if route_or_routes and isinstance(route_or_routes[0], list):
        # Multi-route handling
        routes = route_or_routes  # type: List[List[str]]
        if op_str == "2-opt":
            new_routes = [list(r) for r in routes]
            idx = random.randrange(len(new_routes))
            new_routes[idx] = two_opt_reversal(new_routes[idx])
            return new_routes
        elif op_str == "swap":
            return multi_route_customer_swap(routes)
        else:  # insertion
            return multi_route_customer_insertion(routes)
    else:
        # Single route handling
        route = route_or_routes  # type: List[str]
        if op_str == "2-opt":
            return two_opt_reversal(route)
        elif op_str == "swap":
            return node_swap(route)
        else:  # insertion
            return node_insertion(route)
