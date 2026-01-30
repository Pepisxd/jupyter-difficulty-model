#!/usr/bin/env python3
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


FIELDS = [
    "id",
    "dificultad",
    "tema",
    "enunciado",
    "input_desc",
    "output_desc",
    "restricciones",
    "ejemplo_entrada",
    "ejemplo_salida",
    "codigo",
    "explicacion",
    "tiempo",
    "memoria",
]


@dataclass
class Template:
    name: str
    dificultad: str
    tema: str
    build: Callable[[random.Random, int], dict]


VOWELS = "aeiouáéíóúAEIOUÁÉÍÓÚ"


def _rand_list(rng: random.Random, n: int, low: int, high: int) -> list[int]:
    return [rng.randint(low, high) for _ in range(n)]


def _as_space(nums: list[int]) -> str:
    return " ".join(str(x) for x in nums)


def build_sum_positives(rng: random.Random, idx: int) -> dict:
    contexts = ["ventas diarias", "temperaturas", "puntos", "ganancias", "mediciones"]
    ctx = rng.choice(contexts)
    n = rng.randint(5, 8)
    nums = _rand_list(rng, n, -10, 10)
    s = sum(x for x in nums if x > 0)
    return {
        "tema": "arrays",
        "enunciado": f"Dado un entero n y una lista de {ctx}, imprime la suma de los valores positivos.",
        "input_desc": "Un entero n seguido de n enteros.",
        "output_desc": "La suma de los valores positivos.",
        "restricciones": "1 <= n <= 100000\nLos valores pueden ser negativos.",
        "ejemplo_entrada": f"{n}\n{_as_space(nums)}",
        "ejemplo_salida": str(s),
        "codigo": """import sys

data = list(map(int, sys.stdin.read().split()))
if not data:
    raise SystemExit
n = data[0]
nums = data[1:1+n]
print(sum(x for x in nums if x > 0))
""",
        "explicacion": "Se recorre la lista y se acumulan solo los valores mayores que cero.",
        "tiempo": "O(n)",
        "memoria": "O(1)",
    }


def build_count_vowels(rng: random.Random, idx: int) -> dict:
    words = ["programacion", "computadora", "algoritmo", "jupyter", "datos", "academia"]
    word = rng.choice(words)
    count = sum(1 for c in word if c in VOWELS)
    return {
        "tema": "strings",
        "enunciado": "Cuenta cuántas vocales hay en una cadena.",
        "input_desc": "Una línea con una cadena de texto.",
        "output_desc": "El número total de vocales.",
        "restricciones": "1 <= longitud <= 100000",
        "ejemplo_entrada": word,
        "ejemplo_salida": str(count),
        "codigo": """s = input().strip()
print(sum(1 for c in s if c in "aeiouáéíóúAEIOUÁÉÍÓÚ"))
""",
        "explicacion": "Se recorre la cadena y se cuentan los caracteres que son vocal.",
        "tiempo": "O(n)",
        "memoria": "O(1)",
    }


def build_palindrome(rng: random.Random, idx: int) -> dict:
    options = ["reconocer", "nivel", "radar", "python", "casa"]
    s = rng.choice(options)
    out = "SI" if s == s[::-1] else "NO"
    return {
        "tema": "strings",
        "enunciado": "Determina si una cadena es un palíndromo exacto.",
        "input_desc": "Una línea con una cadena sin espacios.",
        "output_desc": "Imprime SI si es palíndromo, en caso contrario NO.",
        "restricciones": "1 <= longitud <= 100000",
        "ejemplo_entrada": s,
        "ejemplo_salida": out,
        "codigo": """s = input().strip()
print("SI" if s == s[::-1] else "NO")
""",
        "explicacion": "Se compara la cadena con su reverso.",
        "tiempo": "O(n)",
        "memoria": "O(1)",
    }


def build_max_value(rng: random.Random, idx: int) -> dict:
    n = rng.randint(5, 10)
    nums = _rand_list(rng, n, -50, 50)
    return {
        "tema": "arrays",
        "enunciado": "Encuentra el valor máximo de una lista de enteros.",
        "input_desc": "Un entero n seguido de n enteros.",
        "output_desc": "El valor máximo.",
        "restricciones": "1 <= n <= 200000",
        "ejemplo_entrada": f"{n}\n{_as_space(nums)}",
        "ejemplo_salida": str(max(nums)),
        "codigo": """import sys

data = list(map(int, sys.stdin.read().split()))
if not data:
    raise SystemExit
n = data[0]
nums = data[1:1+n]
print(max(nums))
""",
        "explicacion": "Se recorre la lista y se mantiene el máximo encontrado.",
        "tiempo": "O(n)",
        "memoria": "O(1)",
    }


def build_sum_evens(rng: random.Random, idx: int) -> dict:
    n = rng.randint(5, 9)
    nums = _rand_list(rng, n, -10, 20)
    s = sum(x for x in nums if x % 2 == 0)
    return {
        "tema": "arrays",
        "enunciado": "Suma los números pares de una lista de enteros.",
        "input_desc": "Un entero n seguido de n enteros.",
        "output_desc": "La suma de los valores pares.",
        "restricciones": "1 <= n <= 100000",
        "ejemplo_entrada": f"{n}\n{_as_space(nums)}",
        "ejemplo_salida": str(s),
        "codigo": """import sys

data = list(map(int, sys.stdin.read().split()))
if not data:
    raise SystemExit
n = data[0]
nums = data[1:1+n]
print(sum(x for x in nums if x % 2 == 0))
""",
        "explicacion": "Se recorre la lista y se suman los elementos pares.",
        "tiempo": "O(n)",
        "memoria": "O(1)",
    }


def build_count_greater(rng: random.Random, idx: int) -> dict:
    n = rng.randint(6, 10)
    nums = _rand_list(rng, n, -5, 12)
    k = rng.randint(-2, 8)
    cnt = sum(1 for x in nums if x > k)
    return {
        "tema": "arrays",
        "enunciado": "Cuenta cuántos valores de la lista son mayores que k.",
        "input_desc": "Un entero n, luego n enteros y al final k.",
        "output_desc": "La cantidad de números mayores que k.",
        "restricciones": "1 <= n <= 200000",
        "ejemplo_entrada": f"{n}\n{_as_space(nums)}\n{k}",
        "ejemplo_salida": str(cnt),
        "codigo": """import sys

data = list(map(int, sys.stdin.read().split()))
if not data:
    raise SystemExit
n = data[0]
nums = data[1:1+n]
k = data[1+n]
print(sum(1 for x in nums if x > k))
""",
        "explicacion": "Se compara cada valor con k y se incrementa el conteo.",
        "tiempo": "O(n)",
        "memoria": "O(1)",
    }


def build_reverse_string(rng: random.Random, idx: int) -> dict:
    samples = ["hola", "programa", "datos", "python", "curso"]
    s = rng.choice(samples)
    return {
        "tema": "strings",
        "enunciado": "Imprime una cadena en orden inverso.",
        "input_desc": "Una línea con una cadena sin espacios.",
        "output_desc": "La cadena invertida.",
        "restricciones": "1 <= longitud <= 100000",
        "ejemplo_entrada": s,
        "ejemplo_salida": s[::-1],
        "codigo": """s = input().strip()
print(s[::-1])
""",
        "explicacion": "Se usa slicing para invertir la cadena.",
        "tiempo": "O(n)",
        "memoria": "O(n)",
    }


def build_is_prime_small(rng: random.Random, idx: int) -> dict:
    n = rng.randint(2, 50)
    def is_prime(x: int) -> bool:
        if x < 2:
            return False
        i = 2
        while i * i <= x:
            if x % i == 0:
                return False
            i += 1
        return True
    out = "SI" if is_prime(n) else "NO"
    return {
        "tema": "math",
        "enunciado": "Determina si un número es primo.",
        "input_desc": "Un entero n.",
        "output_desc": "SI si n es primo, en caso contrario NO.",
        "restricciones": "2 <= n <= 10^9",
        "ejemplo_entrada": str(n),
        "ejemplo_salida": out,
        "codigo": """n = int(input().strip())
if n < 2:
    print("NO")
else:
    i = 2
    while i * i <= n:
        if n % i == 0:
            print("NO")
            break
        i += 1
    else:
        print("SI")
""",
        "explicacion": "Se prueba divisibilidad hasta la raíz cuadrada.",
        "tiempo": "O(sqrt(n))",
        "memoria": "O(1)",
    }


def build_two_sum(rng: random.Random, idx: int) -> dict:
    n = rng.randint(5, 7)
    nums = _rand_list(rng, n, 1, 9)
    i, j = rng.sample(range(n), 2)
    target = nums[i] + nums[j]
    return {
        "tema": "hash_map",
        "enunciado": "Dado un arreglo y un objetivo, encuentra dos índices con suma igual al objetivo.",
        "input_desc": "Un entero n, luego n enteros, y finalmente el objetivo.",
        "output_desc": "Dos índices (0-based) separados por espacio.",
        "restricciones": "2 <= n <= 200000\nExiste exactamente una solución.",
        "ejemplo_entrada": f"{n}\n{_as_space(nums)}\n{target}",
        "ejemplo_salida": f"{i} {j}",
        "codigo": """import sys

data = list(map(int, sys.stdin.read().split()))
if not data:
    raise SystemExit
n = data[0]
nums = data[1:1+n]
target = data[1+n]
seen = {}
for idx, x in enumerate(nums):
    need = target - x
    if need in seen:
        print(seen[need], idx)
        break
    seen[x] = idx
""",
        "explicacion": "Se usa un mapa para verificar en O(1) si el complemento ya apareció.",
        "tiempo": "O(n)",
        "memoria": "O(n)",
    }


def build_longest_unique_substring(rng: random.Random, idx: int) -> dict:
    options = ["abcaefg", "bbbca", "pwwkew", "abcabcbb", "cadaba"]
    s = rng.choice(options)
    seen = {}
    best = 0
    left = 0
    for right, ch in enumerate(s):
        if ch in seen and seen[ch] >= left:
            left = seen[ch] + 1
        seen[ch] = right
        best = max(best, right - left + 1)
    return {
        "tema": "sliding_window",
        "enunciado": "Calcula la longitud de la subcadena sin caracteres repetidos más larga.",
        "input_desc": "Una cadena s.",
        "output_desc": "La longitud máxima.",
        "restricciones": "1 <= longitud <= 200000",
        "ejemplo_entrada": s,
        "ejemplo_salida": str(best),
        "codigo": """s = input().strip()
last = {}
left = 0
best = 0
for right, ch in enumerate(s):
    if ch in last and last[ch] >= left:
        left = last[ch] + 1
    last[ch] = right
    best = max(best, right - left + 1)
print(best)
""",
        "explicacion": "Ventana deslizante que se ajusta cuando se repite un caracter.",
        "tiempo": "O(n)",
        "memoria": "O(n)",
    }


def build_balanced_parentheses(rng: random.Random, idx: int) -> dict:
    samples = ["()[]{}", "([{}])", "([)]", "((())", "{[()()]}"]
    s = rng.choice(samples)
    def is_balanced(s: str) -> bool:
        pairs = {')': '(', ']': '[', '}': '{'}
        stack = []
        for ch in s:
            if ch in "([{":
                stack.append(ch)
            elif ch in pairs:
                if not stack or stack[-1] != pairs[ch]:
                    return False
                stack.pop()
        return not stack
    out = "SI" if is_balanced(s) else "NO"
    return {
        "tema": "stack",
        "enunciado": "Verifica si una cadena de paréntesis y llaves está balanceada.",
        "input_desc": "Una cadena con caracteres ()[]{}.",
        "output_desc": "SI si está balanceada, de lo contrario NO.",
        "restricciones": "1 <= longitud <= 200000",
        "ejemplo_entrada": s,
        "ejemplo_salida": out,
        "codigo": """s = input().strip()
pairs = {')': '(', ']': '[', '}': '{'}
stack = []
for ch in s:
    if ch in "([{":
        stack.append(ch)
    elif ch in pairs:
        if not stack or stack[-1] != pairs[ch]:
            print("NO")
            break
        stack.pop()
else:
    print("SI" if not stack else "NO")
""",
        "explicacion": "Se usa una pila para validar el cierre correcto de símbolos.",
        "tiempo": "O(n)",
        "memoria": "O(n)",
    }


def build_merge_intervals(rng: random.Random, idx: int) -> dict:
    intervals = [(1, 3), (2, 6), (8, 10), (15, 18)]
    # Keep example deterministic
    merged = [(1, 6), (8, 10), (15, 18)]
    input_lines = ["4"] + [f"{a} {b}" for a, b in intervals]
    output_lines = [str(len(merged))] + [f"{a} {b}" for a, b in merged]
    return {
        "tema": "sorting",
        "enunciado": "Dado un conjunto de intervalos, combina los que se traslapan.",
        "input_desc": "Un entero n y luego n líneas con intervalos [inicio fin].",
        "output_desc": "Primero el número de intervalos resultantes y luego cada intervalo.",
        "restricciones": "1 <= n <= 200000\n1 <= inicio <= fin <= 10^9",
        "ejemplo_entrada": "\n".join(input_lines),
        "ejemplo_salida": "\n".join(output_lines),
        "codigo": """import sys

data = list(map(int, sys.stdin.read().split()))
if not data:
    raise SystemExit
n = data[0]
vals = data[1:]
intervals = [(vals[i], vals[i+1]) for i in range(0, 2*n, 2)]
intervals.sort()
merged = []
for a, b in intervals:
    if not merged or a > merged[-1][1]:
        merged.append([a, b])
    else:
        merged[-1][1] = max(merged[-1][1], b)
print(len(merged))
for a, b in merged:
    print(a, b)
""",
        "explicacion": "Se ordenan los intervalos y se fusionan de izquierda a derecha.",
        "tiempo": "O(n log n)",
        "memoria": "O(n)",
    }


def build_dijkstra(rng: random.Random, idx: int) -> dict:
    n = 5
    edges = [
        (1, 2, 4),
        (1, 3, 2),
        (3, 2, 1),
        (2, 4, 7),
        (3, 4, 3),
        (4, 5, 1),
    ]
    start, end = 1, 5
    # compute shortest path
    import heapq
    graph = {i: [] for i in range(1, n + 1)}
    for u, v, w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    dist = {i: 10**9 for i in range(1, n + 1)}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    ans = dist[end] if dist[end] < 10**9 else -1
    input_lines = [f"{n} {len(edges)}"] + [f"{u} {v} {w}" for u, v, w in edges] + [f"{start} {end}"]
    return {
        "tema": "graph",
        "enunciado": "Encuentra la distancia mínima entre dos nodos en un grafo ponderado no dirigido.",
        "input_desc": "n m, luego m aristas (u v w) y finalmente inicio y fin.",
        "output_desc": "La distancia mínima, o -1 si no hay camino.",
        "restricciones": "1 <= n <= 200000\n1 <= m <= 300000",
        "ejemplo_entrada": "\n".join(input_lines),
        "ejemplo_salida": str(ans),
        "codigo": """import sys, heapq

data = list(map(int, sys.stdin.read().split()))
if not data:
    raise SystemExit
it = iter(data)
n = next(it)
m = next(it)
graph = [[] for _ in range(n+1)]
for _ in range(m):
    u = next(it); v = next(it); w = next(it)
    graph[u].append((v, w))
    graph[v].append((u, w))
start = next(it); end = next(it)
INF = 10**18
dist = [INF]*(n+1)
dist[start] = 0
pq = [(0, start)]
while pq:
    d, u = heapq.heappop(pq)
    if d != dist[u]:
        continue
    for v, w in graph[u]:
        nd = d + w
        if nd < dist[v]:
            dist[v] = nd
            heapq.heappush(pq, (nd, v))
ans = dist[end] if dist[end] < INF else -1
print(ans)
""",
        "explicacion": "Dijkstra con cola de prioridad para obtener la distancia mínima.",
        "tiempo": "O((n+m) log n)",
        "memoria": "O(n+m)",
    }


def build_topological(rng: random.Random, idx: int) -> dict:
    n = 6
    edges = [(1, 2), (1, 3), (3, 4), (2, 4), (4, 5), (5, 6)]
    # compute topo order
    from collections import deque
    indeg = [0] * (n + 1)
    g = [[] for _ in range(n + 1)]
    for u, v in edges:
        g[u].append(v)
        indeg[v] += 1
    q = deque([i for i in range(1, n + 1) if indeg[i] == 0])
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    input_lines = [f"{n} {len(edges)}"] + [f"{u} {v}" for u, v in edges]
    return {
        "tema": "graph",
        "enunciado": "Obtén un orden topológico de un grafo dirigido acíclico.",
        "input_desc": "n m, seguido de m aristas dirigidas (u v).",
        "output_desc": "Un orden topológico válido o CICLO si no existe.",
        "restricciones": "1 <= n <= 200000\n1 <= m <= 300000",
        "ejemplo_entrada": "\n".join(input_lines),
        "ejemplo_salida": " ".join(map(str, order)),
        "codigo": """import sys
from collections import deque

data = list(map(int, sys.stdin.read().split()))
if not data:
    raise SystemExit
it = iter(data)
n = next(it); m = next(it)
indeg = [0]*(n+1)
g = [[] for _ in range(n+1)]
for _ in range(m):
    u = next(it); v = next(it)
    g[u].append(v)
    indeg[v] += 1
q = deque([i for i in range(1, n+1) if indeg[i] == 0])
order = []
while q:
    u = q.popleft()
    order.append(u)
    for v in g[u]:
        indeg[v] -= 1
        if indeg[v] == 0:
            q.append(v)
if len(order) != n:
    print("CICLO")
else:
    print(" ".join(map(str, order)))
""",
        "explicacion": "Kahn: se toman nodos con indegree 0 hasta completar el orden.",
        "tiempo": "O(n+m)",
        "memoria": "O(n+m)",
    }


def build_edit_distance(rng: random.Random, idx: int) -> dict:
    pairs = [("gato", "gata"), ("casa", "caso"), ("barco", "banco")]
    a, b = rng.choice(pairs)
    # compute edit distance
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,
                dp[i][j-1] + 1,
                dp[i-1][j-1] + cost,
            )
    dist = dp[n][m]
    return {
        "tema": "dp",
        "enunciado": "Calcula la distancia de edición entre dos cadenas.",
        "input_desc": "Dos líneas con cadenas a y b.",
        "output_desc": "El número mínimo de operaciones de edición.",
        "restricciones": "1 <= longitud <= 5000",
        "ejemplo_entrada": f"{a}\n{b}",
        "ejemplo_salida": str(dist),
        "codigo": """import sys

a = sys.stdin.readline().strip()
b = sys.stdin.readline().strip()
if not a and not b:
    raise SystemExit
n, m = len(a), len(b)
dp = [[0]*(m+1) for _ in range(n+1)]
for i in range(n+1):
    dp[i][0] = i
for j in range(m+1):
    dp[0][j] = j
for i in range(1, n+1):
    for j in range(1, m+1):
        cost = 0 if a[i-1] == b[j-1] else 1
        dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+cost)
print(dp[n][m])
""",
        "explicacion": "DP clásico donde dp[i][j] es la distancia entre prefijos.",
        "tiempo": "O(n*m)",
        "memoria": "O(n*m)",
    }


def build_coin_change(rng: random.Random, idx: int) -> dict:
    coins = [1, 2, 5]
    target = rng.choice([5, 6, 7])
    # count ways
    ways = [0]*(target+1)
    ways[0] = 1
    for c in coins:
        for v in range(c, target+1):
            ways[v] += ways[v-c]
    return {
        "tema": "dp",
        "enunciado": "Cuenta de cuántas formas se puede formar una cantidad con monedas dadas.",
        "input_desc": "m, luego m valores de monedas y finalmente la cantidad objetivo.",
        "output_desc": "Número de formas distintas de formar la cantidad.",
        "restricciones": "1 <= m <= 100\n1 <= objetivo <= 10000",
        "ejemplo_entrada": f"{len(coins)}\n{_as_space(coins)}\n{target}",
        "ejemplo_salida": str(ways[target]),
        "codigo": """import sys

data = list(map(int, sys.stdin.read().split()))
if not data:
    raise SystemExit
m = data[0]
coins = data[1:1+m]
amount = data[1+m]
ways = [0]*(amount+1)
ways[0] = 1
for c in coins:
    for v in range(c, amount+1):
        ways[v] += ways[v-c]
print(ways[amount])
""",
        "explicacion": "DP de conteo: ways[v] acumula las formas de llegar a v.",
        "tiempo": "O(m*objetivo)",
        "memoria": "O(objetivo)",
    }


def templates() -> list[Template]:
    return [
        Template("sum_positives", "facil", "arrays", build_sum_positives),
        Template("count_vowels", "facil", "strings", build_count_vowels),
        Template("palindrome", "facil", "strings", build_palindrome),
        Template("max_value", "facil", "arrays", build_max_value),
        Template("sum_evens", "facil", "arrays", build_sum_evens),
        Template("count_greater", "facil", "arrays", build_count_greater),
        Template("reverse_string", "facil", "strings", build_reverse_string),
        Template("is_prime_small", "facil", "math", build_is_prime_small),
        Template("two_sum", "intermedio", "hash_map", build_two_sum),
        Template("longest_unique", "intermedio", "sliding_window", build_longest_unique_substring),
        Template("balanced", "intermedio", "stack", build_balanced_parentheses),
        Template("merge_intervals", "intermedio", "sorting", build_merge_intervals),
        Template("dijkstra", "dificil", "graph", build_dijkstra),
        Template("topological", "dificil", "graph", build_topological),
        Template("edit_distance", "dificil", "dp", build_edit_distance),
        Template("coin_change", "dificil", "dp", build_coin_change),
    ]


def main() -> None:
    rng = random.Random(42)
    base = Path(__file__).parent
    src = base / "miDataSet_clean.csv"
    out = base / "miDataSet_250.csv"

    existing = []
    with src.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # normalize difficulty
            diff = (row.get("dificultad") or "").strip().lower()
            if diff == "easy":
                diff = "facil"
            row["dificultad"] = diff
            existing.append(row)

    counts = {"facil": 0, "intermedio": 0, "dificil": 0}
    for row in existing:
        if row.get("dificultad") in counts:
            counts[row["dificultad"]] += 1

    total_target = 250
    per_class = {"facil": 117, "intermedio": 67, "dificil": 66}
    needed = {k: max(0, per_class[k] - counts.get(k, 0)) for k in per_class}

    # determine next id
    max_id = 0
    for row in existing:
        rid = row.get("id", "")
        if rid.startswith("ex_"):
            try:
                max_id = max(max_id, int(rid.split("_")[1]))
            except ValueError:
                pass
    next_id = max_id + 1

    generated = []
    tpls = templates()
    tpl_by_diff = {"facil": [], "intermedio": [], "dificil": []}
    for t in tpls:
        tpl_by_diff[t.dificultad].append(t)

    for diff, n_needed in needed.items():
        for _ in range(n_needed):
            tpl = rng.choice(tpl_by_diff[diff])
            row = tpl.build(rng, next_id)
            row = {**row}
            row["id"] = f"ex_{next_id:04d}"
            row["dificultad"] = diff
            row["tema"] = tpl.tema
            generated.append(row)
            next_id += 1

    all_rows = existing + generated
    all_rows = all_rows[:total_target]

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})

    print(f"Generado: {out}")
    print("Total:", len(all_rows))
    from collections import Counter
    print("Distribucion:", Counter(r["dificultad"] for r in all_rows))


if __name__ == "__main__":
    main()
